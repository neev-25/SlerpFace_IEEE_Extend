"""Encrypted, local-owner research templates; never an authentication service.

The production protector is Windows user-scope DPAPI. Protector injection is an
explicit testing seam, not a plaintext fallback or a production configuration.
"""

from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3
import uuid

import numpy as np

from verify_with_antispoof import (
    GROUP_SIZE, ProtectionContext, validate_protection_parameters,
)

MAX_GROUPS = 1024
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024
MAX_CIPHERTEXT_BYTES = MAX_PAYLOAD_BYTES + 65536
SCHEMA_VERSION = 1
_DOMAIN = b"GuardFace local research store v1\x00"


class TemplateStoreError(RuntimeError):
    """A store operation failed without an authentication decision."""


class ProtectionUnavailable(TemplateStoreError):
    """The operating-system protection service is unavailable."""


class ProtectionError(TemplateStoreError):
    """Protection, unprotection, or payload integrity validation failed."""


class CredentialNotFound(TemplateStoreError):
    """The credential is absent or has been revoked."""


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint32),
                ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


class DPAPIProtector:
    """Windows user-scope protection with noninteractive API calls only."""

    def __init__(self):
        if os.name != "nt":
            raise ProtectionUnavailable("This research store requires Windows user-scope DPAPI.")
        try:
            self._crypt = ctypes.WinDLL("crypt32", use_last_error=True)
            self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            args = [ctypes.POINTER(_DataBlob), ctypes.c_void_p,
                    ctypes.POINTER(_DataBlob), ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(_DataBlob)]
            for name in ("CryptProtectData", "CryptUnprotectData"):
                function = getattr(self._crypt, name)
                function.argtypes = args
                function.restype = ctypes.c_int
            self._kernel.LocalFree.argtypes = [ctypes.c_void_p]
            self._kernel.LocalFree.restype = ctypes.c_void_p
        except (AttributeError, OSError) as exc:
            raise ProtectionUnavailable("Windows DPAPI could not be initialized.") from exc

    def _call(self, data, entropy, *, decrypt):
        if not isinstance(data, bytes) or not data or len(data) > MAX_CIPHERTEXT_BYTES:
            raise ProtectionError("Protection input is empty or exceeds its size limit.")
        if not isinstance(entropy, bytes) or not 32 <= len(entropy) <= 256:
            raise ProtectionError("Protection namespace is invalid.")
        source = ctypes.create_string_buffer(data, len(data))
        extra = ctypes.create_string_buffer(entropy, len(entropy))
        source_blob = _DataBlob(len(data), ctypes.cast(source, ctypes.POINTER(ctypes.c_ubyte)))
        entropy_blob = _DataBlob(len(entropy), ctypes.cast(extra, ctypes.POINTER(ctypes.c_ubyte)))
        output = _DataBlob()
        function = self._crypt.CryptUnprotectData if decrypt else self._crypt.CryptProtectData
        try:
            # No description, prompt, reserved value, or LOCAL_MACHINE scope.
            if not function(ctypes.byref(source_blob), None, ctypes.byref(entropy_blob),
                            None, None, 0x01, ctypes.byref(output)):
                code = ctypes.get_last_error()
                raise ProtectionError(f"Windows DPAPI operation failed (error {code}).")
            if not output.pbData or not 0 < output.cbData <= MAX_CIPHERTEXT_BYTES:
                raise ProtectionError("Windows DPAPI returned invalid output.")
            return ctypes.string_at(output.pbData, output.cbData)
        finally:
            ctypes.memset(source, 0, len(data))
            ctypes.memset(extra, 0, len(entropy))
            if output.pbData:
                if decrypt:
                    ctypes.memset(output.pbData, 0, output.cbData)
                self._kernel.LocalFree(ctypes.cast(output.pbData, ctypes.c_void_p))

    def protect(self, data, entropy):
        return self._call(data, entropy, decrypt=False)

    def unprotect(self, data, entropy):
        return self._call(data, entropy, decrypt=True)


def _identifier(value):
    if not isinstance(value, str) or len(value) != 32:
        raise ValueError("Credential identifier must be a 32-character random hexadecimal ID.")
    try:
        if uuid.UUID(hex=value).hex != value:
            raise ValueError
    except ValueError as exc:
        raise ValueError("Credential identifier is invalid.") from exc
    return value


def _subject(value):
    try:
        valid_length = isinstance(value, str) and 0 < len(value.encode("utf-8")) <= 256
    except UnicodeError:
        valid_length = False
    if not valid_length or not value.strip():
        raise ValueError("Research subject label must contain 1 to 256 UTF-8 bytes.")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Research subject label must not contain control characters.")
    return value


def _validated_material(template, context, alpha, drop_rate):
    if any(not np.isscalar(value) or np.asarray(value).dtype.kind not in "uif"
           for value in (alpha, drop_rate)):
        raise ValueError("Protection parameters must be real numeric scalars.")
    validate_protection_parameters(alpha, drop_rate)
    if not isinstance(context, ProtectionContext):
        raise ValueError("An explicit ProtectionContext is required.")
    for array in (template, context.key):
        if np.iscomplexobj(array):
            raise ValueError("Research template material must be real numeric arrays.")
    try:
        protected = np.asarray(template)
        key = np.asarray(context.key)
        keep = np.asarray(context.keep_mask)
        if protected.dtype.kind not in "uif" or key.dtype.kind not in "uif":
            raise ValueError
        if (protected.ndim != 2 or not 1 <= protected.shape[0] <= MAX_GROUPS or
                protected.shape[1] != GROUP_SIZE or key.shape != protected.shape or
                keep.shape != protected.shape or keep.dtype != np.bool_):
            raise ValueError
        protected = protected.astype(np.float64, copy=True)
        key = key.astype(np.float64, copy=True)
        keep = keep.copy()
        if not np.isfinite(protected).all() or not np.isfinite(key).all():
            raise ValueError
        if not np.all(keep.sum(axis=1) == GROUP_SIZE - int(GROUP_SIZE * drop_rate)):
            raise ValueError
        if not np.allclose(np.linalg.norm(key, axis=1), 1, rtol=1e-6, atol=1e-7):
            raise ValueError
        if (not np.allclose(np.linalg.norm(protected, axis=1), 1, rtol=1e-6, atol=1e-7) or
                np.any(protected[~keep] != 0)):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Research material has invalid shape, normalization, mask, or numeric values.") from exc
    for array in (protected, key, keep):
        array.setflags(write=False)
    return protected, ProtectionContext(key, keep)


def _strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate payload field.")
        value[key] = item
    return value


@dataclass(frozen=True)
class ResearchTemplate:
    credential_id: str
    subject: str = field(repr=False)
    protected_template: np.ndarray = field(repr=False)
    context: ProtectionContext = field(repr=False)
    alpha: float
    drop_rate: float
    authenticated: bool = field(default=False, init=False)


class TemplateStore:
    """Explicit SQLite location, encrypted payloads, and atomic local lifecycle.

    Public results contain only opaque identifiers/status and authenticated=False.
    ``load_research`` exposes sensitive material in memory to the local researcher;
    no matcher or authentication operation is implemented.
    """

    def __init__(self, path, *, _test_protector=None):
        self.path = Path(path).absolute()
        self._protector = _test_protector if _test_protector is not None else DPAPIProtector()

    def _protect(self, plaintext, entropy):
        result = self._protector.protect(plaintext, entropy)
        if not isinstance(result, bytes) or not result or len(result) > MAX_CIPHERTEXT_BYTES:
            raise ProtectionError("Protector returned invalid ciphertext.")
        return result

    def _unprotect(self, ciphertext, entropy):
        if not isinstance(ciphertext, bytes) or not 0 < len(ciphertext) <= MAX_CIPHERTEXT_BYTES:
            raise ProtectionError("Stored ciphertext is invalid or exceeds its size limit.")
        result = self._protector.unprotect(ciphertext, entropy)
        if not isinstance(result, bytes) or not 0 < len(result) <= MAX_PAYLOAD_BYTES:
            raise ProtectionError("Unprotected payload is invalid or exceeds its size limit.")
        return result

    @contextmanager
    def _connection(self):
        if self.path.is_symlink() or not self.path.is_file():
            raise TemplateStoreError("Research store does not exist or is not a regular file.")
        connection = sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True, timeout=10)
        try:
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA secure_delete=ON")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self):
        if not self.path.parent.is_dir():
            raise TemplateStoreError("Choose an existing private directory for the research store.")
        namespace = secrets.token_bytes(32)
        seal = self._protect(_DOMAIN + namespace, _DOMAIN + namespace)
        # Exclusive creation preserves all existing caller files.
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("CREATE TABLE metadata (singleton INTEGER PRIMARY KEY CHECK(singleton=1), version INTEGER NOT NULL, namespace BLOB NOT NULL, seal BLOB NOT NULL)")
                connection.execute("CREATE TABLE credentials (id TEXT PRIMARY KEY, status TEXT NOT NULL CHECK(status IN ('active','revoked')), payload BLOB, CHECK((status='active' AND payload IS NOT NULL) OR (status='revoked' AND payload IS NULL)))")
                connection.execute("INSERT INTO metadata VALUES (1, ?, ?, ?)",
                                   (SCHEMA_VERSION, namespace, seal))
                connection.commit()
        except BaseException:
            # Only the file exclusively created above belongs to this operation.
            self.path.unlink(missing_ok=True)
            raise
        return {"status": "INITIALIZED_RESEARCH_ONLY", "authenticated": False}

    def _namespace(self, connection):
        try:
            rows = connection.execute("SELECT version, namespace, seal FROM metadata").fetchall()
            if len(rows) != 1:
                raise ValueError
            version, namespace, seal = rows[0]
            if version != SCHEMA_VERSION or not isinstance(namespace, bytes) or len(namespace) != 32:
                raise ValueError
            clear = self._unprotect(seal, _DOMAIN + namespace)
            if not hmac.compare_digest(clear, _DOMAIN + namespace):
                raise ValueError
            return namespace
        except (sqlite3.Error, ValueError) as exc:
            raise ProtectionError("Research store header is invalid or incompatible.") from exc

    def _encode(self, credential_id, namespace, subject, template, context, alpha, drop_rate):
        subject = _subject(subject)
        template, context = _validated_material(template, context, alpha, drop_rate)
        material = {"version": SCHEMA_VERSION, "credential_id": credential_id,
                    "namespace": namespace.hex(), "purpose": "RESEARCH_ONLY",
                    "subject": subject, "template": template.tolist(),
                    "key": context.key.tolist(), "keep_mask": context.keep_mask.tolist(),
                    "alpha": float(alpha), "drop_rate": float(drop_rate)}
        body = json.dumps(material, allow_nan=False, separators=(",", ":"),
                          sort_keys=True).encode("utf-8")
        # Extra structure/checksum rejects corrupted output even if a DPAPI call
        # succeeds. This digest is not a separate secret or anti-rollback scheme.
        envelope = json.dumps({"body": body.decode("utf-8"),
                               "sha256": hashlib.sha256(body).hexdigest()},
                              separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        if len(envelope) > MAX_PAYLOAD_BYTES:
            raise ValueError("Research payload exceeds its size limit.")
        return self._protect(envelope, _DOMAIN + namespace + credential_id.encode("ascii"))

    def _decode(self, credential_id, namespace, ciphertext):
        try:
            clear = self._unprotect(ciphertext, _DOMAIN + namespace + credential_id.encode("ascii"))
            envelope = json.loads(clear, object_pairs_hook=_strict_object)
            if not isinstance(envelope, dict) or set(envelope) != {"body", "sha256"}:
                raise ValueError
            body, digest = envelope["body"], envelope["sha256"]
            if not isinstance(body, str) or not isinstance(digest, str):
                raise ValueError
            if not hmac.compare_digest(hashlib.sha256(body.encode("utf-8")).hexdigest(), digest):
                raise ValueError
            value = json.loads(body, object_pairs_hook=_strict_object)
            expected = {"version", "credential_id", "namespace", "purpose", "subject",
                        "template", "key", "keep_mask", "alpha", "drop_rate"}
            if (not isinstance(value, dict) or set(value) != expected or
                    value["version"] != SCHEMA_VERSION or value["credential_id"] != credential_id or
                    value["namespace"] != namespace.hex() or value["purpose"] != "RESEARCH_ONLY"):
                raise ValueError
            subject = _subject(value["subject"])
            template, context = _validated_material(
                value["template"], ProtectionContext(np.asarray(value["key"]),
                                                     np.asarray(value["keep_mask"])),
                value["alpha"], value["drop_rate"])
            return ResearchTemplate(credential_id, subject, template, context,
                                    float(value["alpha"]), float(value["drop_rate"]))
        except (ValueError, TypeError, OverflowError, KeyError, RecursionError, UnicodeError) as exc:
            raise ProtectionError("Research payload failed integrity or schema validation.") from exc

    @staticmethod
    def _public(credential_id, status):
        return {"credential_id": credential_id, "status": status, "authenticated": False}

    def enroll(self, subject, template, context, *, alpha=0.9, drop_rate=0.5):
        credential_id = uuid.uuid4().hex
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            namespace = self._namespace(connection)
            ciphertext = self._encode(credential_id, namespace, subject, template, context, alpha, drop_rate)
            connection.execute("INSERT INTO credentials VALUES (?, 'active', ?)", (credential_id, ciphertext))
            connection.commit()
        return self._public(credential_id, "active")

    def list_credentials(self):
        with self._connection() as connection:
            self._namespace(connection)
            rows = connection.execute("SELECT id, status FROM credentials ORDER BY id").fetchall()
            for credential_id, status in rows:
                _identifier(credential_id)
                if status not in {"active", "revoked"}:
                    raise ProtectionError("Credential status is invalid.")
            return [self._public(*row) for row in rows]

    def _active(self, connection, credential_id):
        row = connection.execute("SELECT payload FROM credentials WHERE id=? AND status='active'",
                                 (_identifier(credential_id),)).fetchone()
        if row is None:
            raise CredentialNotFound("Credential is absent or revoked.")
        return row[0]

    def load_research(self, credential_id):
        with self._connection() as connection:
            namespace = self._namespace(connection)
            return self._decode(_identifier(credential_id), namespace,
                                self._active(connection, credential_id))

    def revoke(self, credential_id):
        credential_id = _identifier(credential_id)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            namespace = self._namespace(connection)
            ciphertext = self._active(connection, credential_id)
            self._decode(credential_id, namespace, ciphertext)
            connection.execute("UPDATE credentials SET status='revoked', payload=NULL WHERE id=?", (credential_id,))
            connection.commit()
        return self._public(credential_id, "revoked")

    def rotate(self, credential_id, template, context, *, alpha=0.9, drop_rate=0.5):
        credential_id = _identifier(credential_id)
        replacement = uuid.uuid4().hex
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            namespace = self._namespace(connection)
            previous = self._decode(credential_id, namespace, self._active(connection, credential_id))
            if isinstance(context, ProtectionContext) and np.array_equal(previous.context.key, context.key):
                raise ValueError("Rotation requires a fresh independently generated protection context.")
            ciphertext = self._encode(replacement, namespace, previous.subject,
                                      template, context, alpha, drop_rate)
            connection.execute("INSERT INTO credentials VALUES (?, 'active', ?)", (replacement, ciphertext))
            connection.execute("UPDATE credentials SET status='revoked', payload=NULL WHERE id=?", (credential_id,))
            connection.commit()
        return {"revoked_credential_id": credential_id, **self._public(replacement, "active")}
