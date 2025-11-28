#!/usr/bin/env bash
#./setup_robotic_attack_env.sh
#bash setup_robotic_attack_env.sh
#hf auth login
# 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

#source /etc/network_turbo

set -eo pipefail
# Note: removed 'u' (unset variable check) to avoid false positives

ENV_NAME="${1:-roboticAttack}"
PYTHON_VERSION="${2:-3.10}"

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
eval "$(conda shell.bash hook)" 2>/dev/null || true

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[INFO] Environment '$ENV_NAME' already exists; reusing it."
else
  echo "[INFO] Creating environment '$ENV_NAME' with python=$PYTHON_VERSION."
  # Try to create environment, but continue if it fails (environment might already exist)
  if ! conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y 2>/dev/null; then
    echo "[WARNING] Failed to create conda environment, checking if it exists..."
    if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
      echo "[INFO] Environment '$ENV_NAME' exists, continuing..."
    else
      echo "[ERROR] Cannot create or find environment '$ENV_NAME'" >&2
      exit 1
    fi
  fi
fi

# Activate environment (try multiple methods)
if conda activate "$ENV_NAME" 2>/dev/null; then
  echo "[INFO] Environment activated successfully"
else
  # Fallback: directly use the environment's python
  # Try to find conda base path from environment list or common locations
  CONDA_BASE=""
  if [[ -d "/home/zifeng/siyuan/miniconda3" ]]; then
    CONDA_BASE="/home/zifeng/siyuan/miniconda3"
  elif [[ -d "/root/miniconda3" ]]; then
    CONDA_BASE="/root/miniconda3"
  elif [[ -d "$HOME/miniconda3" ]]; then
    CONDA_BASE="$HOME/miniconda3"
  elif [[ -d "/opt/conda" ]]; then
    CONDA_BASE="/opt/conda"
  else
    # Try to get from conda env list
    CONDA_ENV_PATH=$(conda env list | grep "^$ENV_NAME" | awk '{print $NF}' | head -1)
    if [[ -n "$CONDA_ENV_PATH" ]]; then
      CONDA_BASE="$(dirname "$(dirname "$CONDA_ENV_PATH")")"
    fi
  fi
  
  if [[ -n "$CONDA_BASE" ]]; then
    ENV_PYTHON="$CONDA_BASE/envs/$ENV_NAME/bin/python"
    if [[ -f "$ENV_PYTHON" ]]; then
      export PATH="$(dirname "$ENV_PYTHON"):$PATH"
      echo "[INFO] Using Python from: $ENV_PYTHON"
    else
      echo "[ERROR] Cannot activate environment or find Python at $ENV_PYTHON" >&2
      exit 1
    fi
  else
    echo "[ERROR] Cannot activate environment or find conda base directory" >&2
    exit 1
  fi
fi

python -m pip install --upgrade pip setuptools wheel

CONSTRAINT_FILE="$(mktemp /tmp/robotic_attack_constraints.XXXXXX.txt)"
cat >"$CONSTRAINT_FILE" <<'EOF'
numpy==1.26.4
typing-extensions==4.12.2
protobuf<4.25
torch==2.2.*
torchvision==0.17.*
torchaudio==2.2.*
tensorflow==2.15.0
keras==2.15.0
tensorboard==2.15.0
tensorflow-estimator==2.15.0
EOF

echo "[INFO] Pinning numeric core (numpy/typing-extensions/protobuf)."
pip install -c "$CONSTRAINT_FILE" --force-reinstall \
  numpy==1.26.4 typing-extensions==4.12.2 "protobuf<4.25"

echo "[INFO] Installing PyTorch 2.2.x (CUDA 12.1 wheels)."
pip install \
  "torch==2.2.*" "torchvision==0.17.*" "torchaudio==2.2.*" \
  -f https://mirrors.tuna.tsinghua.edu.cn/pytorch/wheels/cu121/

echo "[INFO] Installing core project dependencies."
pip install -c "$CONSTRAINT_FILE" --no-deps \
  timm==0.9.10 tokenizers==0.19.1 transformers==4.40.1 || {
  echo "[WARNING] Some core dependencies failed to install, continuing..."
}

pip install -c "$CONSTRAINT_FILE" \
  "huggingface-hub>=0.19.3,<1.0" "pyyaml>=6.0" "regex!=2019.12.17" \
  "safetensors>=0.4.1" "pillow>=10.0"

pip install -c "$CONSTRAINT_FILE" \
  accelerate==0.27.2 draccus==0.8.0 einops==0.8.1 json-numpy==2.0.0 \
  jsonlines==4.0.0 matplotlib==3.8.4 peft==0.11.1 rich==13.9.3 wandb==0.17.0

echo "[INFO] Installing robosuite and simulation dependencies."
pip install -c "$CONSTRAINT_FILE" \
  seaborn==0.13.2 imageio==2.31.5 gym==0.25.2 robosuite==1.4.0 requests==2.32.3 || {
  echo "[WARNING] Some simulation dependencies failed to install, continuing..."
}

echo "[INFO] Installing TensorFlow and RLDS stack."
pip install -U -c "$CONSTRAINT_FILE" \
  "tensorflow==2.15.0" "tensorflow-datasets==4.9.*" "tensorflow-metadata==1.15.0" \
  "rlds>=0.1.8" "dm-reverb==0.14.0" "tensorflow-graphics==2021.12.3"

echo "[INFO] Installing dlimp fork without dependency constraints."
# Check if dlimp file exists in multiple locations
DLIMP_FILE=""
for possible_path in \
  "/data/zifeng/siyuan/download/dlimp_openvla-main.zip" \
  "/data/zifeng/download/dlimp_openvla-main.zip" \
  "/root/autodl-tmp/Download/dlimp_openvla-main.zip"; do
  if [[ -f "$possible_path" ]]; then
    DLIMP_FILE="$possible_path"
    break
  fi
done

if [[ -n "$DLIMP_FILE" ]]; then
  echo "[INFO] Found dlimp file at: $DLIMP_FILE"
  pip install --no-deps "$DLIMP_FILE"
else
  echo "[WARNING] dlimp_openvla-main.zip not found in common locations, skipping..."
  echo "[WARNING] You may need to install dlimp manually later if required."
fi

echo "[INFO] Installing project in editable mode."
pip install -e .

echo "[INFO] Running pip check."
pip check || true

rm -f "$CONSTRAINT_FILE"

cat <<MSG
[INFO] Environment setup completed.
[INFO] Activate the environment with:  conda activate $ENV_NAME
[INFO] Run repository scripts from:     ~/code/roboticAttack
MSG

