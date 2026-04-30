# Security Policy

## Reporting a Vulnerability

JPEG decoders ingest adversarial byte streams from the wild. If you find a
crash, memory-safety violation, or undefined behavior in `ashlar`, please
report it privately rather than opening a public issue.

Use GitHub's private vulnerability reporting for the repository, or contact the
maintainer through the repository owner profile if private reporting is not yet
enabled.

Please include:
- A minimal reproducer (input bytes + API call).
- Rust version, target triple, and cargo features used.
- Expected vs. observed behavior.

Reports are acknowledged within 7 days. Patches are issued as soon as possible,
generally within 30 days for high-severity issues.

## Supported versions

Until 1.0, only the most recent minor version receives security patches.
