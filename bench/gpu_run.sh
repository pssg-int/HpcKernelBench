#!/usr/bin/env bash
# bench/gpu_run.sh [-g <gres type>] [-t MINUTES] [-m MEM] -- '<shell command>'
#
# Run one shell command on a GPU compute node through SLURM, with bench/env.sh
# sourced and cwd = bench/. Written for machines whose login nodes have no GPU
# (zaratan): every gate / GPU-needing build goes through this. Nothing in it is
# machine-specific beyond the defaults, all overridable:
#   KB_SLURM_PARTITION   partition name                       (default: gpu)
#   KB_SLURM_GRES        default gres type for -g             (default: a100_1g.5gb,
#                        an A100 MIG slice: sm_80, 14 SMs, ~5 GB, allocates in
#                        seconds; full cards queue for hours -- pass -g a100 /
#                        -g h100 when a gate needs more memory or a newer arch)
#   KB_SLURM_EXTRA       extra srun flags (account, qos, ...)  (default: none)
#
#   bench/gpu_run.sh -- '$PY -m kernelbench.runner --kernel spmm --variant spmm-gpu-kernel-f32 --smoke --impl rode-spmm-f32'
#   bench/gpu_run.sh -g h100 -t 40 -- 'bench/artifacts/spmm/voltrix/build.sh'
set -u
gpu="${KB_SLURM_GRES:-a100_1g.5gb}"; mins=20; mem=""
while [ $# -gt 0 ]; do
  case "$1" in
    -g) gpu="$2"; shift 2;;
    -t) mins="$2"; shift 2;;
    -m) mem="$2"; shift 2;;
    --) shift; break;;
    *) break;;
  esac
done
[ -n "$mem" ] || { case "$gpu" in *1g.*) mem=16G;; *) mem=32G;; esac; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cmd="$*"
# shellcheck disable=SC2086
exec srun -N 1 -n 1 -c 4 -t "${mins}:00" -p "${KB_SLURM_PARTITION:-gpu}" --mem="$mem" \
  --gres="gpu:${gpu}:1" ${KB_SLURM_EXTRA:-} \
  bash -c "cd '$HERE' && source env.sh >/dev/null && echo \"[gpu_run] host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1) job=\${SLURM_JOB_ID:-?}\" && $cmd"
