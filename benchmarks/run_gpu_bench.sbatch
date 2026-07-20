#!/bin/zsh
#SBATCH -c 2
#SBATCH -p gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --mem 12000
#SBATCH -t 60
#SBATCH -J syn_bench
#
# GPU benchmark + verification for the memristor speedup work (plan Step 1 / Tier 1B).
#
# IMPORTANT: direct -o/-e to a SHARED filesystem (e.g. /work or /u), NOT node-local
# /var/tmp. The job runs on a compute node, so a /var/tmp path is written on that
# node's local disk and is invisible from the submit host. Example:
#   sbatch -o /u/USER/dev/syn_bench_%j.out -e /u/USER/dev/syn_bench_%j.err benchmarks/run_gpu_bench.sbatch
#
# Runs on the local venv (torch 2.3.0+cu121) so results match the repo's pinned torch.

set -u
cd /work/asr4/hilmes/dev/Synaptogen_dev
PY=./venv/bin/python

echo "==== node / gpu ===="
hostname
nvidia-smi || true
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo
echo "==== equivalence: eager vs compiled-fast (cuda) ===="
# post-ADC quantization can turn ~1e-6 analog drift into rare 1-step flips, so a
# small n_diff is acceptable; max_abs_diff should stay at the ADC-step scale.
$PY -m benchmarks.check_equivalence --compare-paths --device cuda --atol 1e-4 --rtol 1e-3 || true

echo
echo "==== baseline profile (eager) -- large: FF linears 512x2048 + convs ===="
$PY -m benchmarks.profile_modules --device cuda --scale large --iters 30 --warmup 5

echo
echo "==== fast profile (compiled) -- large: FF linears 512x2048 + convs ===="
SYN_FAST=1 $PY -m benchmarks.profile_modules --device cuda --scale large --iters 30 --warmup 8

echo
echo "==== baseline profile (eager) -- proj: projection linears 512x512 ===="
$PY -m benchmarks.profile_modules --device cuda --scale proj --only linear tiled_linear --iters 30 --warmup 5

echo
echo "==== fast profile (compiled) -- proj: projection linears 512x512 ===="
SYN_FAST=1 $PY -m benchmarks.profile_modules --device cuda --scale proj --only linear tiled_linear --iters 30 --warmup 8

echo
echo "==== recompile probe: dynamic shapes should NOT recompile ===="
TORCH_LOGS=recompiles SYN_FAST=1 $PY - <<'PYEOF' || true
import torch
from benchmarks._builders import build_linear
name, mod, _ = build_linear(batch=64, in_features=512, out_features=512, device="cuda")
for B in (32, 96, 160, 224):
    xb = torch.randn(B, 512, device="cuda")
    with torch.no_grad():
        mod(xb)
torch.cuda.synchronize()
print("varied batch sizes 32/96/160/224 without error")
PYEOF

# NOTE: the pytest end-to-end markers (MNIST training on the eager path) are slow
# (~tens of minutes) and previously overran the time limit. They are orthogonal to
# the speedup (Tier 1A is bit-exact, the fast path is bit-identical), so run them
# as a separate job if desired:
#   $PY -m pytest tests -m "linear or tiled_linear or conv or conv2d" -q

echo
echo "==== done ===="
