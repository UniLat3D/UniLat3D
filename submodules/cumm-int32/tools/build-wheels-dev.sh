#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    outpath="$2"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w "$outpath"
    fi
}

export CUMM_DISABLE_JIT="1"
export CUMM_CUDA_ARCH_LIST="all"
export CUDA_VERSION="126"
if [ -d /io/third_party/cccl ]; then
  echo "CCCL already cloned."
else
    # clone nvidia cuda cccl to cumm/third_party
    if [ $CUDA_VERSION -lt "120" ]; then
        CCCL_BRANCH_NAME="2.7.0"
    elif [ $CUDA_VERSION -lt "122" ]; then
        CCCL_BRANCH_NAME="2.2.0"
    else
        CCCL_MINOR_VERSION=${CUDA_VERSION:2:1}
        CCCL_BRANCH_NAME="2.$CCCL_MINOR_VERSION.0"
    fi
    git clone https://github.com/NVIDIA/cccl.git /io/third_party/cccl -b v$CCCL_BRANCH_NAME
fi
"/opt/python/cp39-cp39/bin/pip" wheel /io/ --no-deps -w /io/wheelhouse_tmp -v

# "/opt/python/cp311-cp311/bin/pip" wheel /io/ --no-deps -w /io/wheelhouse_tmp -v

# "/opt/python/cp313-cp313/bin/pip" wheel /io/ --no-deps -w /io/wheelhouse_tmp

# Bundle external shared libraries into the wheels
for whl in /io/wheelhouse_tmp/*.whl; do
    repair_wheel "$whl" /io/dist
done

rm -rf /io/wheelhouse_tmp