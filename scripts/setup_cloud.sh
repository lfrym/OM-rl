#!/bin/bash
# Setup script for cloud GPU instances (RunPod, Lambda, etc.)
# Run this after cloning the repo onto the instance.
set -euo pipefail

echo "=== OM-rl Cloud Setup ==="

# 1. Install Python dependencies
echo "Installing Python dependencies..."
pip install torch transformers trl peft bitsandbytes datasets pyyaml networkx accelerate

# 2. Build omsim shared library (Linux)
echo "Building omsim..."
bash build_omsim.sh

# 3. Quick sanity check
echo "Running sanity check..."
python3 -c "
import sys; sys.path.insert(0, '.')
from vendor.opus_magnum.verifier import _get_lib
_get_lib()
print('omsim: OK')

from om_rl.puzzle_gen.generator import generate_puzzle
p = generate_puzzle(complexity_level=1, seed=0)
print(f'puzzle gen: OK ({p.name})')

import torch
print(f'torch: OK (cuda={torch.cuda.is_available()}, device_count={torch.cuda.device_count()})')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Quick test (10 steps, ~2 min):"
echo "  python scripts/train.py --max-steps 10 --batch-size 1 --num-completions 2"
echo ""
echo "First real run (1000 steps, ~8 hrs):"
echo "  python scripts/train.py --max-steps 1000"
