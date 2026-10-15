"""Research lifecycle transactions, payload validation, privacy, and Windows DPAPI."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

import manage_templates
from modules.template_store import (
    CredentialNotFound, DPAPIProtector, MAX_CIPHERTEXT_BYTES, MAX_GROUPS,
    ProtectionError, ProtectionUnavailable, TemplateStore,
)
from verify_with_antispoof import ProtectionContext, create_protection_context, slerp_encrypt


class MemoryProtector:
    """Opaque in-memory token map, only for transaction tests; not encryption."""

    def __init__(self):
        self.records = {}
        self.fail_next = False

    def protect(self, clear, entropy):
        if self.fail_next:
            self.fail_next = False
            raise ProtectionError("Injected protection failure.")
        token = secrets.token_bytes(64)
        self.records[token] = (clear, entropy)
        return token

    def unprotect(self, token, entropy):
        record = self.records.get(token)
        if record is None or record[1] != entropy:
            raise ProtectionError("Injected integrity or namespace failure.")
        return record[0]


def material(seed=13):
    features = np.random.default_rng(seed).normal(size=(49, 16))
    context = create_protection_context(features.shape, seed=seed)
    value = slerp_encrypt(features, context=context)
    return value / np.linalg.norm(value, axis=1, keepdims=True), context


@pytest.fixture
def store(tmp_path):
    instance = TemplateStore(tmp_path / "research.db", _test_protector=MemoryProtector())
    assert not instance.path.exists()
    assert instance.initialize()["authenticated"] is False
    return instance


def test_enroll_load_copies_and_never_exposes_subject_or_context_in_public_results(store):
    template, context = material()
    result = store.enroll("private-subject-marker", template, context)
    credential = result["credential_id"]
    loaded = store.load_research(credential)
    template[0] *= -1
    assert loaded.subject == "private-subject-marker"
    assert not np.array_equal(loaded.protected_template, template)
    assert loaded.authenticated is False and not loaded.protected_template.flags.writeable
    assert not loaded.context.key.flags.writeable and not loaded.context.keep_mask.flags.writeable
    assert "array" not in repr(loaded) and "private-subject-marker" not in repr(loaded)
    public = json.dumps([result, store.list_credentials()])
    assert "private-subject-marker" not in public
    assert all(word not in public for word in ("template", "keep_mask", '"key"'))
    for path in store.path.parent.glob("research.db*"):
        assert b"private-subject-marker" not in path.read_bytes()
        assert b'"template"' not in path.read_bytes()


def test_revoke_removes_ciphertext_and_old_id_is_unusable(store):
    credential = store.enroll("subject", *material())["credential_id"]
    assert store.revoke(credential)["status"] == "revoked"
    with pytest.raises(CredentialNotFound):
        store.load_research(credential)
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT payload FROM credentials WHERE id=?", (credential,)).fetchone() == (None,)
    assert store.list_credentials() == [{"credential_id": credential, "status": "revoked", "authenticated": False}]


def test_rotate_atomically_replaces_material_and_preserves_only_encrypted_subject(store):
    old = store.enroll("secret-label", *material())["credential_id"]
    replacement, context = material(22)
    result = store.rotate(old, replacement, context)
    new = result["credential_id"]
    assert old != new and result["revoked_credential_id"] == old
    assert result["authenticated"] is False
    with pytest.raises(CredentialNotFound):
        store.load_research(old)
    record = store.load_research(new)
    assert record.subject == "secret-label"
    np.testing.assert_array_equal(record.protected_template, replacement)
    np.testing.assert_array_equal(record.context.key, context.key)
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT payload FROM credentials WHERE id=?", (old,)).fetchone() == (None,)


def test_rotation_protector_failure_keeps_old_credential(store):
    old = store.enroll("subject", *material())["credential_id"]
    store._protector.fail_next = True
    with pytest.raises(ProtectionError):
        store.rotate(old, *material(30))
    assert store.load_research(old).subject == "subject"
    assert len(store.list_credentials()) == 1


def test_rotation_requires_a_fresh_context(store):
    value, context = material()
    old = store.enroll("subject", value, context)["credential_id"]
    with pytest.raises(ValueError, match="fresh"):
        store.rotate(old, value, context)
    assert store.load_research(old).subject == "subject"


def test_rotation_sql_failure_rolls_back_insert_and_revocation(store):
    old = store.enroll("subject", *material())["credential_id"]
    with sqlite3.connect(store.path) as connection:
        connection.execute("CREATE TRIGGER reject_revoke BEFORE UPDATE ON credentials BEGIN SELECT RAISE(ABORT,'test failure'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.rotate(old, *material(31))
    assert store.load_research(old).subject == "subject"
    assert len(store.list_credentials()) == 1


def test_concurrent_rotation_has_exactly_one_winner(store):
    old = store.enroll("subject", *material())["credential_id"]

    def rotate(seed):
        try:
            return store.rotate(old, *material(seed))["credential_id"]
        except CredentialNotFound:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(rotate, (40, 41)))
    assert sum(value is not None for value in results) == 1
    assert sum(row["status"] == "active" for row in store.list_credentials()) == 1


def test_corrupt_ciphertext_and_swapped_rows_are_rejected(store):
    first = store.enroll("first", *material())["credential_id"]
    second = store.enroll("second", *material(21))["credential_id"]
    with sqlite3.connect(store.path) as connection:
        cipher = connection.execute("SELECT payload FROM credentials WHERE id=?", (first,)).fetchone()[0]
        connection.execute("UPDATE credentials SET payload=? WHERE id=?", (cipher, second))
    with pytest.raises(ProtectionError):
        store.load_research(second)
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE credentials SET payload=? WHERE id=?", (cipher[:-1] + bytes([cipher[-1] ^ 1]), first))
    with pytest.raises(ProtectionError):
        store.load_research(first)


def test_namespace_change_and_cross_store_payload_are_rejected(store, tmp_path):
    identifier = store.enroll("subject", *material())["credential_id"]
    other = TemplateStore(tmp_path / "other.db", _test_protector=store._protector)
    other.initialize()
    with sqlite3.connect(store.path) as connection:
        cipher = connection.execute("SELECT payload FROM credentials WHERE id=?", (identifier,)).fetchone()[0]
    with sqlite3.connect(other.path) as connection:
        connection.execute("INSERT INTO credentials VALUES (?, 'active', ?)", (identifier, cipher))
    with pytest.raises(ProtectionError):
        other.load_research(identifier)
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE metadata SET namespace=?", (secrets.token_bytes(32),))
    with pytest.raises(ProtectionError):
        store.list_credentials()


def test_schema_and_checksum_check_even_if_protector_returns_corrupted_output(store):
    identifier = store.enroll("subject", *material())["credential_id"]
    with sqlite3.connect(store.path) as connection:
        cipher = connection.execute("SELECT payload FROM credentials WHERE id=?", (identifier,)).fetchone()[0]
    clear, entropy = store._protector.records[cipher]
    envelope = json.loads(clear)
    envelope["body"] = envelope["body"].replace('"RESEARCH_ONLY"', '"OTHER_PURPOSE"')
    store._protector.records[cipher] = (json.dumps(envelope).encode(), entropy)
    with pytest.raises(ProtectionError, match="integrity"):
        store.load_research(identifier)


def test_duplicate_serialized_fields_are_rejected(store):
    identifier = store.enroll("subject", *material())["credential_id"]
    with sqlite3.connect(store.path) as connection:
        cipher = connection.execute("SELECT payload FROM credentials WHERE id=?", (identifier,)).fetchone()[0]
    clear, entropy = store._protector.records[cipher]
    store._protector.records[cipher] = (b'{"body":"ignored",' + clear[1:], entropy)
    with pytest.raises(ProtectionError, match="schema"):
        store.load_research(identifier)


@pytest.mark.parametrize("kind", ["nan", "complex", "shape", "zero", "mask", "key", "oversize", "raw_scale"])
def test_invalid_material_never_enrolls(store, kind):
    template, context = material()
    if kind == "nan":
        template[0, 0] = np.nan
    elif kind == "complex":
        template = template.astype(complex) + 1j
    elif kind == "shape":
        template = template.ravel()
    elif kind == "zero":
        template[0] = 0
    elif kind == "mask":
        context = ProtectionContext(context.key, np.ones(template.shape, dtype=bool))
    elif kind == "key":
        context = ProtectionContext(context.key * 2, context.keep_mask)
    elif kind == "oversize":
        template = np.zeros((MAX_GROUPS + 1, 16))
    else:
        template *= 2
    with pytest.raises(ValueError):
        store.enroll("subject", template, context)
    assert store.list_credentials() == []


@pytest.mark.parametrize("subject", ["", " ", "a" * 257, "line\nbreak", "\x00", "\ud800"])
def test_subject_validation_and_no_implicit_store_creation(store, tmp_path, subject):
    with pytest.raises(ValueError):
        store.enroll(subject, *material())
    missing = TemplateStore(tmp_path / "missing.db", _test_protector=MemoryProtector())
    with pytest.raises(Exception, match="does not exist"):
        missing.list_credentials()
    assert not missing.path.exists()


def test_existing_file_is_preserved_and_oversize_ciphertext_rejected(store, tmp_path):
    existing = tmp_path / "existing.txt"
    existing.write_bytes(b"caller-owned-content")
    with pytest.raises(FileExistsError):
        TemplateStore(existing, _test_protector=MemoryProtector()).initialize()
    assert existing.read_bytes() == b"caller-owned-content"
    identifier = store.enroll("subject", *material())["credential_id"]
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE credentials SET payload=? WHERE id=?", (b"x" * (MAX_CIPHERTEXT_BYTES + 1), identifier))
    with pytest.raises(ProtectionError, match="size"):
        store.load_research(identifier)


def test_nonwindows_has_no_default_protector_or_cli_fallback(tmp_path, capsys):
    with patch("modules.template_store.os.name", "posix"):
        with pytest.raises(ProtectionUnavailable):
            DPAPIProtector()
        assert manage_templates.main(["--store", str(tmp_path / "unavailable.db"), "init"]) == 2
    assert not (tmp_path / "unavailable.db").exists()
    report = json.loads(capsys.readouterr().err)
    assert report["authenticated"] is False


def test_demo_gate_happens_before_feature_extraction():
    source = Path(__file__).resolve().parents[1] / "samples/spoof_screen_replay.jpg"
    with patch("manage_templates.extract_face_features") as extractor:
        with pytest.raises(ValueError, match="heuristic"):
            manage_templates._demo_material(source)
        extractor.assert_not_called()


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI integration requires a Windows user profile")
def test_real_dpapi_synthetic_lifecycle_wrong_namespace_and_tamper(tmp_path):
    protector = DPAPIProtector()
    entropy = secrets.token_bytes(64)
    ciphertext = protector.protect(b"ephemeral synthetic DPAPI check", entropy)
    assert ciphertext != b"ephemeral synthetic DPAPI check"
    assert protector.unprotect(ciphertext, entropy) == b"ephemeral synthetic DPAPI check"
    with pytest.raises(ProtectionError):
        protector.unprotect(ciphertext, secrets.token_bytes(64))
    mutated = ciphertext[:-1] + bytes([ciphertext[-1] ^ 1])
    with pytest.raises(ProtectionError):
        protector.unprotect(mutated, entropy)
    store = TemplateStore(tmp_path / "dpapi.db")
    store.initialize()
    identifier = store.enroll("ephemeral-dpapi-test-subject", *material())["credential_id"]
    assert store.load_research(identifier).authenticated is False
    assert b"ephemeral-dpapi-test-subject" not in store.path.read_bytes()
    result = store.rotate(identifier, *material(15))
    with pytest.raises(CredentialNotFound):
        store.load_research(identifier)
    store.revoke(result["credential_id"])
    assert all(row["status"] == "revoked" for row in store.list_credentials())


@pytest.mark.skipif(os.name != "nt", reason="Production CLI uses Windows DPAPI only")
def test_real_cli_synthetic_enroll_rotate_revoke_has_no_sensitive_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    database = tmp_path / "cli.db"

    def command(*arguments):
        completed = subprocess.run([sys.executable, "-B", str(root / "manage_templates.py"),
                                    "--store", str(database), *arguments], cwd=root,
                                   capture_output=True, text=True, check=False, timeout=60)
        assert completed.returncode == 0, completed.stderr
        assert "private-cli-subject" not in completed.stdout + completed.stderr
        result = json.loads(completed.stdout)
        assert result["authenticated"] is False
        assert result["assurance"] == "RESEARCH_ONLY_UNTRAINED"
        assert all(word not in completed.stdout for word in ('"template"', '"key"', '"keep_mask"'))
        return result

    command("init")
    first = command("enroll-demo", "--image", str(root / "samples/genuine_personA_1.jpg"),
                    "--subject", "private-cli-subject")["credential_id"]
    second = command("rotate", first, "--image", str(root / "samples/genuine_personA_2.jpg"))["credential_id"]
    command("revoke", second)
    assert all(value["status"] == "revoked" for value in command("list")["credentials"])
    assert b"private-cli-subject" not in database.read_bytes()
