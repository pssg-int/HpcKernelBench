#!/usr/bin/env bash
# Run every implemented kernel's smoke test and summarize.
#
# Smoke runs use synthetic inputs and a reduced protocol, so every result is
# marked non-conforming by construction — this checks that the harness, the
# domain modules and the spec parsing all work end to end, nothing more.
# It is safe on a login node: CPU only, seconds per kernel, no GPU execution.

set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"

mapfile -t KERNELS < <("$PY" - <<'EOF'
import sys; sys.path.insert(0, '.')
from kernelbench import domains
st = domains.load_all()
for k, m in sorted(domains.OWNER.items()):
    if st.get(m) == "ok":
        d = domains.load(k)
        if k in getattr(d, "KERNELS", []):
            print(k)
EOF
)

pass=0; fail=0; failed=()
for k in "${KERNELS[@]}"; do
  # first variant of the kernel's spec, unless a domain-preferred one exists
  v=$("$PY" - "$k" <<'EOF'
import sys; sys.path.insert(0, '.')
from kernelbench import spec, domains
k = sys.argv[1]
d = domains.load(k)
pref = getattr(d, "SMOKE_VARIANT", {}).get(k)
print(pref or spec.load(k).variant_ids[0])
EOF
)
  out=$("$PY" -m kernelbench.runner --kernel "$k" --variant "$v" --smoke 2>&1)
  line=$(echo "$out" | grep -E "runs valid ->" | head -1)
  if echo "$line" | grep -qE "^([1-9][0-9]*)/\1 runs valid"; then
    printf "  %-22s %-34s %s\n" "$k" "$v" "${line%% ->*}"
    pass=$((pass+1))
  else
    printf "  %-22s %-34s FAILED: %s\n" "$k" "$v" "${line:-$(echo "$out" | tail -2 | head -1)}"
    fail=$((fail+1)); failed+=("$k")
  fi
done

echo
echo "kernels green: $pass ; failing: $fail"
[ "$fail" -gt 0 ] && echo "failing kernels: ${failed[*]}"
exit "$fail"
