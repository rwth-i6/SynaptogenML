#!/bin/zsh
#SBATCH -c 2
#SBATCH -p gpu_11gb
#SBATCH --gres=gpu:gtx_1080:1
#SBATCH --exclude=cn-286
#SBATCH --mem 12000
#SBATCH -t 40
#SBATCH -J syn_pascal_jit
#
# TorchScript(NNC)-fusion prototype bench, pinned to a gtx_1080 (sm_61) node —
# the GPUs where torch.compile/triton is unavailable. See pascal_jit_bench.py.
#
# Submit with output on a shared filesystem:
#   sbatch -o /u/USER/dev/syn_pascal_jit_%j.out -e /u/USER/dev/syn_pascal_jit_%j.err \
#       benchmarks/run_pascal_jit_bench.sbatch

set -u
cd /work/asr4/hilmes/dev/Synaptogen_dev
PY=./venv/bin/python

echo "==== node / gpu ===="
hostname
nvidia-smi || true
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"

echo
echo "==== eager vs jit(NNC) vs fast-nocompile on gtx_1080 ===="
$PY -m benchmarks.pascal_jit_bench --device cuda --iters 30 --warmup 8
