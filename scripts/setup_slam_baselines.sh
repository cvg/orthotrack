#!/bin/bash
set -e

# Setup directory for external SLAM repositories
THIRDPARTY_DIR="$(pwd)/thirdparty"
mkdir -p "$THIRDPARTY_DIR"

echo "====================================================="
echo " Setting up External SLAM Baselines for Benchmarking "
echo "====================================================="
echo ""

# 1. VGGT-SLAM v1
echo "[1/3] Cloning VGGT-SLAM v1 (NeurIPS 2025)..."
if [ ! -d "$THIRDPARTY_DIR/VGGT-SLAM-v1" ]; then
    git clone -b version1.0 https://github.com/MIT-SPARK/VGGT-SLAM.git "$THIRDPARTY_DIR/VGGT-SLAM-v1"
else
    echo "  -> VGGT-SLAM-v1 already exists."
fi

# 2. VGGT-SLAM v2 (main branch)
echo "[2/3] Cloning VGGT-SLAM v2 (main branch)..."
if [ ! -d "$THIRDPARTY_DIR/VGGT-SLAM" ]; then
    git clone https://github.com/MIT-SPARK/VGGT-SLAM.git "$THIRDPARTY_DIR/VGGT-SLAM"
else
    echo "  -> VGGT-SLAM v2 already exists."
fi

# 3. VGGT-Long
echo "[3/3] Cloning VGGT-Long..."
if [ ! -d "$THIRDPARTY_DIR/VGGT-Long" ]; then
    git clone https://github.com/DengKaiCQ/VGGT-Long.git "$THIRDPARTY_DIR/VGGT-Long"
else
    echo "  -> VGGT-Long already exists."
fi

echo ""
echo "====================================================="
echo " Repositories cloned to $THIRDPARTY_DIR/"
echo "====================================================="
echo ""
echo "Please set the following environment variables before running benchmarks:"
echo "  export VGGT_SLAM_V1_DIR=$THIRDPARTY_DIR/VGGT-SLAM-v1"
echo "  export VGGT_SLAM_V2_DIR=$THIRDPARTY_DIR/VGGT-SLAM"
echo "  export VGGT_LONG_DIR=$THIRDPARTY_DIR/VGGT-Long"
echo ""
echo "Next steps: Conda Environment Setup"
echo "-----------------------------------"
echo "You must manually create the isolated Conda environments for each method."
echo ""
echo "For VGGT-SLAM (v1 & v2 use the same env base):"
echo "  cd $THIRDPARTY_DIR/VGGT-SLAM"
echo "  conda create -n vggt-slam python=3.9 -y"
echo "  conda activate vggt-slam"
echo "  pip install -r requirements.txt"
echo "  # Follow additional PyTorch/CUDA instructions in their README!"
echo ""
echo "For VGGT-Long:"
echo "  cd $THIRDPARTY_DIR/VGGT-Long"
echo "  conda env create -f environment.yml"
echo "  # See their repo for submodule build instructions."
echo ""
echo "Setup script completed."
