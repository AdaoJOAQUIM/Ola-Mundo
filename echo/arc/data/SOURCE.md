# ARC-AGI-1 task data (source record)

- Author: François Chollet
- Upstream repo: https://github.com/fchollet/ARC (branch `master`)
- Pinned commit: `399030444e0ab0cc8b4e199870fb20b863846f34`
- Files fetched from: `https://raw.githubusercontent.com/fchollet/ARC/399030444e0ab0cc8b4e199870fb20b863846f34/data/<split>/<id>.json`
- Task-id list: taken from a shallow clone of the same commit (the GitHub API and archive endpoints are blocked by the proxy, and raw cannot list directories).
- License: Apache License 2.0 (`LICENSE` in the upstream repo, fetched from raw and matched).

## Contents

| Split      | Files | Size   |
|------------|-------|--------|
| training/  | 400   | 2.3 MB |
| evaluation/| 400   | 3.1 MB |

Total task JSON: 3,782,011 bytes.

## Verification

- Every file parses as JSON and has `train` and `test` lists of `{"input", "output"}` objects.
- Every grid is a non-empty rectangular list of integers in 0-9.
- Files are byte-identical to the pinned clone.
