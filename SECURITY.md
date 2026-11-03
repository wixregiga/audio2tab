# Security policy

## Supported versions

Security fixes go into the latest release and the `ghcr.io/wixregiga/audio2tab:latest` image.

## Reporting a vulnerability

Please **don't open a public issue**. Report it privately through GitHub instead: on the **Security** tab, choose **Report a vulnerability**. You should get a reply within 7 days.

audio2tab processes untrusted media files with ffmpeg, Demucs, and basic-pitch. Reports about crafted input files, the model-weight download (a pinned URL checked by SHA-256), or the Docker image are all in scope.
