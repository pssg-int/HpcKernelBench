#!/usr/bin/env bash
# Pre-fetch every git-ignored artifact checkout (source/, source_*/) from the
# recorded <name>.provenance files, applying each recorded source.patch.
# build.sh does the same lazily for its own directory (via toolchain.sh); run
# this once on a new machine to do it up front and see what cannot be fetched.
#
#   bench/artifacts/fetch_sources.sh            # all artifact directories
#   bench/artifacts/fetch_sources.sh spmm/rode  # one or more <track>/<short>
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KB_NO_FETCH=1 source "$HERE/toolchain.sh"   # defines kb_ensure_source; no fetch for this directory
if [ "$#" -gt 0 ]; then dirs=("${@/#/$HERE/}"); else dirs=("$HERE"/*/*/); fi
ok=0; failed=(); skipped=0
for d in "${dirs[@]}"; do
  d="${d%/}"
  ls "$d"/*.provenance >/dev/null 2>&1 || { skipped=$((skipped+1)); continue; }
  if kb_ensure_source "$d"; then ok=$((ok+1)); else failed+=("${d#$HERE/}"); fi
done
echo "fetch_sources: $ok directories complete, ${#failed[@]} failed, $skipped without provenance"
for f in "${failed[@]}"; do echo "  FAILED: $f"; done
[ "${#failed[@]}" -eq 0 ]
