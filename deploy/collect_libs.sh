#!/usr/bin/env bash
# Copy the two shared libraries the runtime image needs into deploy/lib/.
#   deploy/collect_libs.sh <NCCL root with sm_70 kernels> <CUDA 12.8 root>
#   e.g. deploy/collect_libs.sh /tmp/nccl/nvidia/nccl /usr/local/cuda-12.8
set -euo pipefail
nccl=${1:?NCCL root (the directory holding lib/libnccl.so.2)}
cuda=${2:-/usr/local/cuda-12.8}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/lib"
cp -L "$nccl/lib/libnccl.so.2" "$here/lib/libnccl.so.2"
cp -L "$cuda/lib64/libcudart.so.12" "$here/lib/libcudart.so.12"
ls -la "$here/lib"
