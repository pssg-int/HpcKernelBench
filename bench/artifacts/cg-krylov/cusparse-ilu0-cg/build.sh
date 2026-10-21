#!/usr/bin/env bash
# cusparse-ilu0-cg has no code of its own to compile: it uses the bridge.so
# built by ../cusparse-cg/build.sh (one bridge.cu holds both the plain and
# the ILU(0)-preconditioned CG). This script just runs that build.
# Idempotent. Exit 0 = built.
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/../cusparse-cg/build.sh"
