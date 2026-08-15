#!/bin/bash
# setup.sh — One-shot environment setup for OrthoTrack
#
# Installs pinned deps from requirements.txt (including Open3D for
# create_movingdrone.py --render). Headless/EGL nodes may also need:
#   sudo apt-get install -y libegl1 libgles2 libgl1

set -e

echo "=== OrthoTrack Setup ==="

# 1. Clone RoMaV2 if thirdparty/RoMaV2 is missing
if [ ! -d "thirdparty/RoMaV2/src" ]; then
    echo "[1/3] Cloning RoMaV2..."
    mkdir -p thirdparty
    git clone https://github.com/Parskatt/RoMaV2.git thirdparty/RoMaV2
else
    echo "[1/3] RoMaV2 already present."
fi

# 2. Install Python dependencies
echo "[2/3] Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# Install RoMaV2 in editable mode (weights download on first use via torch.hub)
pip install -e thirdparty/RoMaV2 --no-deps

# 3. Remind about RoMaV2 weights
echo "[3/3] RoMaV2 pretrained weights download automatically on first tracker run."

echo ""
echo "=== Setup complete ==="
echo "Download a scene (example):"
echo "  PYTHONPATH=. python -c \"from dataset.MovingDrone import MovingDrone; MovingDrone(dataset_dir='data/MovingDrone', sequences=['airport10'], predownload=True, load_dop=True, load_dsm=True)\""
echo "Run tracking (turbo ≈ 8 GB VRAM):"
echo "  PYTHONPATH=. python scripts/run_tracking.py --sequence_dir data/MovingDrone/scenes/airport10 --fine_matcher turbo --output outputs/tracking/airport10 --end_frame 30"
echo "Create a MovingDrone scene (mesh + trajectory; see dataset/movingdrone.md):"
echo "  PYTHONPATH=. python dataset/create_movingdrone.py --help"
