#!/bin/zsh
#SBATCH -c 2
#SBATCH -p gpu_24gb
#SBATCH --gres=gpu:1
#SBATCH --mem 12000
#SBATCH -t 30
#SBATCH -J syn_conv2d_test
#
# Runs the conv2d-marked tests on a GPU: the end-to-end MNIST conv2d test plus the
# structural reference tests (tests/test_conv_reference.py) that guard the
# transposed-kernel fix. Submit with logs on a SHARED path, e.g.:
#   sbatch -o /u/USER/dev/syn_conv2d_%j.out -e /u/USER/dev/syn_conv2d_%j.err benchmarks/run_conv2d_tests.sbatch

set -u
cd /work/asr4/hilmes/dev/Synaptogen_dev
PY=./venv/bin/python

echo "== node / torch =="
hostname
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo
echo "== pytest -m conv2d =="
$PY -m pytest tests -m conv2d -v

echo
echo "== done =="
