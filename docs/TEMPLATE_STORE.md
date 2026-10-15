# Local encrypted research template lifecycle

`manage_templates.py` provisions a local Windows desktop research registry. It supports initialization, explicit synthetic enrollment, listing opaque IDs, revocation, and atomic rotation. **Every result remains unauthenticated.** It is separate from the dashboard and verification pipeline, and it contains no matcher, authentication endpoint, user-account service, or production enrollment claim.

## Explicit use

Choose an existing private directory and a new database filename. Initialization refuses to replace an existing file. Nothing is created at import time. Example PowerShell commands from the project root:

```powershell
python -B .\manage_templates.py --store C:\private-research\templates.db init
python -B .\manage_templates.py --store C:\private-research\templates.db enroll-demo --image .\samples\genuine_personA_1.jpg --subject synthetic-example
python -B .\manage_templates.py --store C:\private-research\templates.db list
python -B .\manage_templates.py --store C:\private-research\templates.db rotate CREDENTIAL_ID --image .\samples\genuine_personA_2.jpg
python -B .\manage_templates.py --store C:\private-research\templates.db revoke NEW_CREDENTIAL_ID
```

Replace the placeholder IDs with the random IDs returned by enrollment/rotation. Use generated illustrations for these commands. `enroll-demo` explicitly uses untrained statistics; it does not acquire biometric enrollment evidence. The static-image heuristic runs before extraction and can reject an illustration. The input is decoded once and remains in memory; the tool does not save input images or original vectors. Rotation creates a fresh random context and ID, preserves the encrypted research label, and revokes the old ID in the same transaction. Reusing the old context is rejected.

CLI success exits 0 and includes `authenticated: false` and `assurance: RESEARCH_ONLY_UNTRAINED`; an invalid, unavailable, or failed operation exits 2. Lists contain only opaque IDs/status and false authentication flags. The CLI has no plaintext-export, reveal-subject, protector-selection, or matching option. Subject labels supplied on the command line can still be present in the invoking shell's own history/process arguments; use nonidentifying synthetic labels.

## Storage and operating-system protection

Production protection uses Windows user-scope [CryptProtectData](https://learn.microsoft.com/en-us/windows/win32/api/dpapi/nf-dpapi-cryptprotectdata), [CryptUnprotectData](https://learn.microsoft.com/en-us/windows/win32/api/dpapi/nf-dpapi-cryptunprotectdata), and [LocalFree](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-localfree) through ctypes. Calls prohibit interactive prompts and do not set machine scope. Data is bound to the Windows profile's protection context; roaming/recovery arrangements may affect portability. There is no insecure fallback on another operating system.

Each payload encrypts the research label, normalized protected groups, context key/mask, transform parameters, schema, and its credential/store bindings. Raw face images, original feature vectors, and pretransform group features are not stored. Public database metadata consists of a schema version, random namespace, opaque credential IDs, active/revoked status, and ciphertext. Record counts/status and database size therefore remain observable. The namespace is additional DPAPI entropy and a context binding, not a secret encryption key. A protected header verifies that the namespace and user context are usable before store operations.

Ciphertexts are bound to both their store namespace and credential ID. Decrypted envelopes have a checksum plus strict shape, purpose, ID, namespace, numeric, mask, and normalization checks. The checksum detects corrupted decoded content; it is not an independent keyed authentication or anti-rollback mechanism. Complex/nonfinite material, inconsistent masks, zero/unnormalized groups, malformed subjects, and oversized payloads are rejected. Limits are 1,024 groups of 16 dimensions and 4 MiB of clear serialized payload.

SQLite uses immediate write transactions, full synchronization, DELETE journals, and secure deletion. Revocation clears the current payload and keeps an ID/status tombstone; loading the revoked ID fails. Rotation inserts the replacement and clears the old payload atomically. Encryption and SQL failures roll back the transaction. These operations do not erase old backups, snapshots, SSD history, or copies already held in another process. A researcher who controls the Windows account can decrypt material and can restore an older database copy; anti-rollback and compromise recovery are not implemented.

The DPAPI output allocation is freed, and native decrypted buffers are cleared before release. Python/NumPy/JSON can retain immutable copies or memory pages until reuse; reliable process-wide memory erasure is not claimed. Keep the directory under appropriate operating-system permissions. DPAPI does not isolate the store from malware or other code already running as the same Windows user.

## Research API and validation

`modules.template_store.TemplateStore` accepts an explicit path. `initialize`, `enroll`, `list_credentials`, `revoke`, and `rotate` return public status only. `load_research` is the sensitive, local in-memory API for controlled experiments; its record hides arrays/context/subject in its representation and has `authenticated=False`. It is deliberately not exposed as a CLI export. The `_test_protector` constructor argument is an explicit test seam; production CLI construction hardwires DPAPI.

The focused tests use opaque in-memory test tokens for failure injection and transaction checks; these test tokens are not encryption. Windows integration tests use real DPAPI and ephemeral synthetic data. Tests cover round trips, wrong namespaces, tampering, row/store substitution, validation boundaries, public-output privacy, revocation, atomic rotation, SQL/protector rollback, concurrent rotation, closed behavior off Windows, and the heuristic-before-extraction gate. A separate subprocess test exercises the real CLI lifecycle. Run:

```powershell
python -B -m pytest -q tests\test_template_store.py
```

The final focused Windows run passed all **30 store tests**, including real DPAPI and the production CLI lifecycle. An earlier combined run passed 56 tests across the then-current store and existing pipeline boundary suite. Targeted Ruff checks pass. Final aggregate totals are recorded with the overall validation report. None of these checks establishes identity accuracy, physical attack resistance, template unlinkability, inversion resistance, or production credential security. Independent model/data evaluation and an explicit product/security policy remain necessary before any authentication integration.
