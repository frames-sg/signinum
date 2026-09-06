# Security Policy

## Supported versions

| Version | Status |
| --- | --- |
| `0.11.x` | Candidate line; publication requires the release gates |
| `0.10.0` | Latest published and security-supported release |
| `0.9.0` | Previous published release line; security-supported |
| `0.8.1` | Previous published release line; security-supported |
| `0.8.0` | Previous published release line; security-supported |
| `0.7.5` | Previous published release line; security-supported, except for the documented `j2k-ml` CUDA and Metal packaging defect |
| `0.7.3` | Previous published release line; security-supported |
| `0.7.2` | Previous published release line; security-supported |
| `0.7.1` | Previous published release line; security-supported |
| `0.7.0` | Previous published release line; security-supported |
| `0.6.x` | Supported for security fixes during the pre-1.0 transition |
| Earlier than `0.6` | Unsupported |

Security fixes are developed on the current workspace line and backported to
older supported lines when applicable. See
[`docs/release.md`](docs/release.md) for the publication state.

## Reporting vulnerabilities

When the repository **Security** tab shows **Report a vulnerability**, report
suspected vulnerabilities through the corresponding
[GitHub private reporting form](https://github.com/frames-sg/j2k/security/advisories/new).
If that button is unavailable, do not put vulnerability details in a public
issue. Open a [minimal issue](https://github.com/frames-sg/j2k/issues/new)
asking the maintainers to provide a private contact, without naming the
affected code or including proof-of-concept details. A
verified direct private channel must be published before any future release is
approved.

The tag-publish preflight authenticates to GitHub and reads the repository's
private-vulnerability-reporting setting. Publication fails closed unless that
setting reports `enabled: true`; API authorization failures and malformed
responses also block publication. Before creating the release tag, a repository
administrator must enable **Security > Private vulnerability reporting** and
confirm that **Report a vulnerability** is visible. Ordinary offline repository
lint does not make this network request.

Response expectations:

- Acknowledgment of a private report within **3 business days**.
- Triage decision (accepted / declined / needs more information) within
  **14 days** of acknowledgment.
- Coordinated disclosure: we will agree on a publication date with the
  reporter before any advisory or fix details are made public.

## Baseline expectations

- Unsupported input must fail explicitly.
- Error responses must avoid sensitive internal details.
- Device backends must not silently substitute a different explicit backend.
- Unsafe Rust inventory is tracked in `docs/unsafe-audit.md`.
- Fuzzing and malformed-input tests are part of release hardening.
