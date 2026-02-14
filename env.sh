#!/bin/bash
# WARNING: You MUST run this script using 'source env.sh' (not './env.sh')

# Directory Path
ENV_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Setting
# Python Release: https://www.python.org/downloads/
ENV_NAME="flash_attn"
PYTHON_VERSION="3.12.10"

# Environment Variable
## PYENV
if ! command -v pyenv >/dev/null; then
  echo "pyenv not found, please install it first."
  return 1
fi
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
## CUDA
if [ ! -d "/usr/local/cuda" ]; then
  echo "CUDA not found at /usr/local/cuda"
  return 1
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

echo "Install Python $PYTHON_VERSION"
pyenv install -s $PYTHON_VERSION

echo "Create Python Virtual Env: $ENV_NAME"
pyenv virtualenv $PYTHON_VERSION $ENV_NAME

echo "Activate Python Virtual Env"
pyenv activate $ENV_NAME

echo "Upgrade pip and install requirements"
pip install --upgrade pip
pip install -r $ENV_DIRECTORY/torch_requirements.txt
pip install -r $ENV_DIRECTORY/requirements.txt
