export CUDNN_PATH="/home/jeromeku/nemo/.venv/lib/python3.12/site-packages/nvidia/cudnn"
export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH}:${CUDNN_PATH}/include"
export LIBRARY_PATH="${LIBRARY_PATH}:${CUDNN_PATH}/lib"

uv pip install --no-build-isolation transformer_engine[pytorch]