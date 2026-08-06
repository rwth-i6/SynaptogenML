#!/bin/zsh
#SBATCH -c 16
#SBATCH -p gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --mem 16000
#SBATCH -t 60
#SBATCH -J syn_prog_torch
#
# End-to-end serial-vs-parallel programming comparison on a GPU node.
# Direct -o to a SHARED filesystem, e.g.:
#   sbatch -o /work/asr4/hilmes/dev/Synaptogen_dev/prog_torch_%j.out \
#          benchmarks/run_programming_torch.sbatch

set -u
cd /work/asr4/hilmes/dev/Synaptogen_dev
PY=./venv/bin/python

echo "==== node / gpu ===="
hostname
nvidia-smi || true
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo
echo "==== serial vs parallel programming (torch level) ===="
$PY -m benchmarks.check_programming_torch

echo
echo "==== done ===="
