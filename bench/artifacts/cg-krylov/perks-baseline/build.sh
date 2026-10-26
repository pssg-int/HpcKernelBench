#!/usr/bin/env bash
# perks-cg-baseline runs ../perks's cg_perks.exe with --baseline; nothing of
# its own to compile. This just runs that build. Idempotent. Exit 0 = built.
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/../perks/build.sh"
