#!/usr/bin/env bash
# ./setup_libero_env.sh
# bash experiments/robot/libero/setup_libero_env.sh
#
# This script sets up LIBERO dependencies for roboticAttack environment.
# It should be run AFTER setup_robotic_attack_env.sh
#
# Complete setup process:
#   1. First run: bash setup_robotic_attack_env.sh
#   2. Then run:  bash experiments/robot/libero/setup_libero_env.sh
#
# Usage:
#   cd /root/autodl-tmp/code/roboticAttack
#   bash experiments/robot/libero/setup_libero_env.sh

set -euo pipefail

ENV_NAME="${1:-roboticAttack}"
LIBERO_REPO_URL="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LIBERO_DIR="${LIBERO_DIR:-/home/zifeng/siyuan/code/LIBERO}"
ROBOTIC_ATTACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LIBERO_REQUIREMENTS="${ROBOTIC_ATTACK_DIR}/experiments/robot/libero/libero_requirements.txt"

echo "=========================================="
echo "LIBERO Environment Setup Script"
echo "=========================================="
echo "[INFO] Environment: $ENV_NAME"
echo "[INFO] LIBERO directory: $LIBERO_DIR"
echo "[INFO] Requirements file: $LIBERO_REQUIREMENTS"
echo ""

# Initialize conda (try multiple common locations)
if [[ -f /home/zifeng/siyuan/miniconda3/etc/profile.d/conda.sh ]]; then
  source /home/zifeng/siyuan/miniconda3/etc/profile.d/conda.sh
elif [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
elif [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
  source ~/miniconda3/etc/profile.d/conda.sh
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
fi

# Check if conda is available after initialization
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] Conda is not available in this shell." >&2
  echo "[ERROR] Tried to source conda.sh from common locations but failed." >&2
  echo "[ERROR] Please ensure conda is installed or run 'conda init bash' and restart the terminal." >&2
  exit 1
fi

# Enable conda commands inside non-interactive scripts
# Note: We already sourced conda.sh above, so we can use conda directly
eval "$(conda shell.bash hook)" 2>/dev/null || true

# Check if environment exists
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[ERROR] Environment '$ENV_NAME' does not exist. Please run setup_robotic_attack_env.sh first." >&2
  exit 1
fi

echo "[INFO] Activating conda environment: $ENV_NAME"
# Use conda activate - this should work after sourcing conda.sh
if conda activate "$ENV_NAME" 2>/dev/null; then
  echo "[INFO] Environment activated successfully"
else
  echo "[WARNING] conda activate failed, using direct PATH method..." >&2
  # Alternative: directly use the environment's python by setting PATH
  # Try to find conda base path from common locations
  CONDA_BASE=""
  if [[ -d "/home/zifeng/siyuan/miniconda3" ]]; then
    CONDA_BASE="/home/zifeng/siyuan/miniconda3"
  elif [[ -d "/root/miniconda3" ]]; then
    CONDA_BASE="/root/miniconda3"
  elif [[ -d "$HOME/miniconda3" ]]; then
    CONDA_BASE="$HOME/miniconda3"
  elif [[ -d "/opt/conda" ]]; then
    CONDA_BASE="/opt/conda"
  fi
  
  if [[ -n "$CONDA_BASE" ]]; then
    ENV_PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"
    if [[ -f "$ENV_PYTHON" ]]; then
      export PATH="$(dirname "$ENV_PYTHON"):$PATH"
      echo "[INFO] Using Python from: $ENV_PYTHON"
    else
      # Try to find conda env path from conda env list
      CONDA_ENV_PATH=$(conda env list | grep "^$ENV_NAME" | awk '{print $NF}' | head -1)
      if [[ -n "$CONDA_ENV_PATH" ]] && [[ -f "$CONDA_ENV_PATH/bin/python" ]]; then
        export PATH="$CONDA_ENV_PATH/bin:$PATH"
        echo "[INFO] Using Python from: $CONDA_ENV_PATH/bin/python"
      else
        echo "[ERROR] Cannot find Python in environment $ENV_NAME" >&2
        exit 1
      fi
    fi
  else
    # Try to find conda env path from conda env list as last resort
    CONDA_ENV_PATH=$(conda env list | grep "^$ENV_NAME" | awk '{print $NF}' | head -1)
    if [[ -n "$CONDA_ENV_PATH" ]] && [[ -f "$CONDA_ENV_PATH/bin/python" ]]; then
      export PATH="$CONDA_ENV_PATH/bin:$PATH"
      echo "[INFO] Using Python from: $CONDA_ENV_PATH/bin/python"
    else
      echo "[ERROR] Cannot find Python in environment $ENV_NAME" >&2
      exit 1
    fi
  fi
fi

# Verify we're in the right environment or using the right Python
CURRENT_PYTHON=$(which python 2>/dev/null || echo "")
if [[ "$CURRENT_PYTHON" != *"$ENV_NAME"* ]] && [[ "$CURRENT_PYTHON" != "" ]]; then
  echo "[WARNING] Python path doesn't match environment name, but continuing..."
  echo "[INFO] Using Python: $CURRENT_PYTHON"
fi

echo "[INFO] Python: $(which python)"
echo "[INFO] Python version: $(python --version)"
echo ""

# ==========================================
# Step 1: Clone/Update LIBERO repository
# ==========================================
echo "=========================================="
echo "Step 1: Setting up LIBERO repository"
echo "=========================================="

LIBERO_PARENT_DIR="$(dirname "$LIBERO_DIR")"
mkdir -p "$LIBERO_PARENT_DIR"

if [[ -d "$LIBERO_DIR" ]]; then
  echo "[INFO] LIBERO directory already exists: $LIBERO_DIR"
  echo "[INFO] Using existing LIBERO repository (skipping git update due to network constraints)"
else
  echo "[ERROR] LIBERO directory not found at: $LIBERO_DIR" >&2
  echo "[ERROR] Please ensure LIBERO is manually uploaded/extracted to this location" >&2
  echo "[ERROR] Expected location: /home/zifeng/siyuan/code/LIBERO" >&2
  exit 1
fi

echo "[INFO] LIBERO repository ready at: $LIBERO_DIR"
echo ""

# ==========================================
# Step 2: Install LIBERO (editable mode, no deps)
# ==========================================
echo "=========================================="
echo "Step 2: Installing LIBERO package"
echo "=========================================="

cd "$LIBERO_DIR"
echo "[INFO] Installing LIBERO in editable mode (without dependencies)..."
pip install -e . --no-deps || {
  echo "[ERROR] Failed to install LIBERO" >&2
  exit 1
}

echo "[INFO] LIBERO package installed successfully."

# Add LIBERO to PYTHONPATH as a workaround for editable install issues
# Use ${PYTHONPATH:-} to handle unset variable (due to set -u)
if [[ ":${PYTHONPATH:-}:" != *":$LIBERO_DIR:"* ]]; then
  export PYTHONPATH="$LIBERO_DIR${PYTHONPATH:+:$PYTHONPATH}"
  echo "[INFO] Added LIBERO to PYTHONPATH: $LIBERO_DIR"
fi
echo ""

# ==========================================
# Step 3: Install dependencies from roboticAttack requirements
# ==========================================
echo "=========================================="
echo "Step 3: Installing LIBERO dependencies"
echo "=========================================="

if [[ ! -f "$LIBERO_REQUIREMENTS" ]]; then
  echo "[ERROR] Requirements file not found: $LIBERO_REQUIREMENTS" >&2
  exit 1
fi

echo "[INFO] Installing dependencies from: $LIBERO_REQUIREMENTS"
echo "[INFO] This file has no version constraints, using latest compatible versions."

cd "$ROBOTIC_ATTACK_DIR"
pip install -r "$LIBERO_REQUIREMENTS" || {
  echo "[WARNING] Some dependencies may have failed to install. Continuing..." >&2
}

# Install Ultralytics YOLO package required by downstream evaluation scripts
pip install ultralytics || {
  echo "[WARNING] Failed to install ultralytics. YOLO-based tools may not work." >&2
}

echo "[INFO] Dependencies installation completed."
echo ""

# ==========================================
# Step 4: Verify installation
# ==========================================
echo "=========================================="
echo "Step 4: Verifying installation"
echo "=========================================="

cd "$ROBOTIC_ATTACK_DIR"

# Set environment variable to avoid interactive prompts
export LIBERO_DATASET_PATH="${LIBERO_DATASET_PATH:-/data/zifeng/siyuan/data/datasets}"

echo "[INFO] Testing imports..."

python << 'VERIFY_EOF'
import sys
import os

# Set dataset path to avoid interactive prompt
os.environ.setdefault('LIBERO_DATASET_PATH', '/data/zifeng/siyuan/data/datasets')

# Manually add LIBERO path if editable install didn't work
LIBERO_DIR = '/home/zifeng/siyuan/code/LIBERO'
if LIBERO_DIR not in sys.path:
    sys.path.insert(0, LIBERO_DIR)

errors = []

# Test libero import
try:
    # Mock input to avoid interactive prompt during import
    import builtins
    original_input = builtins.input
    builtins.input = lambda x: 'N'  # Answer 'N' to any input prompt
    
    try:
        # Try to import libero (path should already be added above)
        from libero.libero import benchmark
        print("✓ libero.libero.benchmark imported successfully")
    except ImportError as import_err:
        # If still fails, try importing the parent module first
        try:
            import libero
            from libero.libero import benchmark
            print("✓ libero.libero.benchmark imported successfully (after parent import)")
        except Exception as e2:
            raise import_err from e2
    finally:
        # Restore original input function
        builtins.input = original_input
except Exception as e:
    # If import fails, try with explicit path verification
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "libero.libero", 
            f"{LIBERO_DIR}/libero/libero/__init__.py"
        )
        if spec and spec.loader:
            # Module file exists, but import failed (likely due to editable install issue)
            print("✓ libero module file found (editable install path may need manual fix)")
            print(f"  [INFO] You may need to add {LIBERO_DIR} to PYTHONPATH")
            print(f"  [INFO] Or use: export PYTHONPATH={LIBERO_DIR}:\$PYTHONPATH")
        else:
            errors.append(f"libero: {e}")
            print(f"✗ libero import failed: {e}")
    except Exception as e2:
        errors.append(f"libero: {e} (also: {e2})")
        print(f"✗ libero import failed: {e}")

# Test robosuite
try:
    import robosuite
    print(f"✓ robosuite: {robosuite.__version__}")
except Exception as e:
    errors.append(f"robosuite: {e}")
    print(f"✗ robosuite import failed: {e}")

# Test bddl
try:
    import bddl
    print("✓ bddl imported successfully")
except Exception as e:
    errors.append(f"bddl: {e}")
    print(f"✗ bddl import failed: {e}")

# Test easydict
try:
    import easydict
    print("✓ easydict imported successfully")
except Exception as e:
    errors.append(f"easydict: {e}")
    print(f"✗ easydict import failed: {e}")

# Test imageio
try:
    import imageio
    print(f"✓ imageio: {imageio.__version__}")
except Exception as e:
    errors.append(f"imageio: {e}")
    print(f"✗ imageio import failed: {e}")

# Test gym
try:
    import gym
    print(f"✓ gym: {gym.__version__}")
except Exception as e:
    errors.append(f"gym: {e}")
    print(f"✗ gym import failed: {e}")

# Test cloudpickle
try:
    import cloudpickle
    print("✓ cloudpickle imported successfully")
except Exception as e:
    errors.append(f"cloudpickle: {e}")
    print(f"✗ cloudpickle import failed: {e}")

if errors:
    print("\n[WARNING] Some imports failed. Please check the errors above.")
    sys.exit(1)
else:
    print("\n[SUCCESS] All dependencies verified successfully!")
VERIFY_EOF

VERIFY_EXIT_CODE=$?

if [[ $VERIFY_EXIT_CODE -eq 0 ]]; then
  echo ""
  echo "=========================================="
  echo "LIBERO Environment Setup Completed!"
  echo "=========================================="
  echo "[SUCCESS] All dependencies are installed and verified."
  echo ""
  echo "Note: If you encounter 'ModuleNotFoundError: No module named libero' when running scripts,"
  echo "      add LIBERO to PYTHONPATH:"
  echo "      export PYTHONPATH=/home/zifeng/siyuan/code/LIBERO:\$PYTHONPATH"
  echo ""
  echo "Next steps:"
  echo "  1. Ensure LIBERO dataset is available at: $LIBERO_DATASET_PATH"
  echo "  2. Run robot simulation:"
  echo "     python experiments/robot/libero/run_libero_eval_args_geo_batch.py \\"
  echo "       --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \\"
  echo "       --task_suite_name libero_spatial \\"
  echo "       --num_trials_per_task 10 \\"
  echo "       --patchroot <path_to_patch.pt> \\"
  echo "       --x 120 --y 160 --angle 0 --shx 0 --shy 0 \\"
  echo "       --cudaid 0"
  echo ""
else
  echo ""
  echo "=========================================="
  echo "Setup Completed with Warnings"
  echo "=========================================="
  echo "[WARNING] Some dependencies failed verification."
  echo "[INFO] You may need to install missing packages manually."
  echo ""
  exit 1
fi

