#!/usr/bin/env bash
# Insum / IndirectEinsum has no native build step: it is pure Python
# (torch.compile + torch._inductor -> Triton codegen), no CUDA extension to
# compile. "Building" here means: confirm the interpreter/CUDA stack can
# actually import and run the artifact's own SpMM-COO example end to end
# (which is what proves the kernel is real, not just importable).
#
# Idempotent: safe to re-run; exit 0 = usable.
set -euo pipefail
# Toolchain pin (2026-09-04): the login default became cudatoolkit/13.2 while
# torch is cu128; this re-points CUDA_HOME/PATH/CPATH at the matching 12.9
# toolkit + math_libs headers. See bench/artifacts/toolchain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/../../toolchain.sh"
cd "$(dirname "$0")"

PY="${PY:-/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python}"

echo "== Insum / IndirectEinsum build check =="
echo "commit: $(git -C source rev-parse HEAD 2>/dev/null || echo unknown)"
echo "python: $PY"
"$PY" - <<'EOF'
import torch, triton
print("torch  :", torch.__version__)
print("triton :", triton.__version__)
print("cuda   :", torch.version.cuda, "available:", torch.cuda.is_available())
EOF

# README's own claim: peak perf needs torch._inductor.config.triton.native_matmul
# (shipped in torch 2.10). This machine's torch is 2.8.0+cu128, where that
# config attribute does not exist -- confirmed and recorded, not worked around
# by patching the artifact's kernel code (ARTIFACT_GUIDE.md rule 3).
"$PY" - <<'EOF'
import torch
try:
    torch._inductor.config.triton.native_matmul = True
    print("native_matmul: available (unexpected on torch<2.10)")
except AttributeError as e:
    print("native_matmul: NOT available on this torch build ->", e)
EOF

# Functional check: run the repo's own SpMM-COO example unmodified, proving
# torch.compile can still lower the indirect-einsum expression to a working
# Triton kernel without native_matmul.
"$PY" source/example.py
echo "== Insum build check: OK (example.py SpMM-COO ran + its own correctness check passed) =="
