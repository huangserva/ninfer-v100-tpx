# Runtime image for ninfer-v100-tpx: the single-GPU ninfer-serve, the TP2 proxy and the two CUDA
# libraries the binaries need. It packages binaries you built on the host (README, 编译); it does
# not compile anything. Steps:
#
#   deploy/collect_libs.sh /tmp/nccl/nvidia/nccl /usr/local/cuda-12.8    # -> deploy/lib/
#   docker build -f deploy/Dockerfile -t ninfer-v100-tpx:latest .
#
# The NVIDIA driver libraries come from the host at run time (--gpus, nvidia-container-toolkit).
FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3 libavcodec60 libavformat60 libavutil58 libcurl4t64 libswscale7 \
    && rm -rf /var/lib/apt/lists/*
COPY build-v100/apps/ninfer /usr/local/bin/ninfer
COPY build-v100/apps/ninfer-serve /usr/local/bin/ninfer-serve
COPY tools/tp2/tp2_proxy.py /usr/local/bin/tp2_proxy.py
COPY deploy/lib/libnccl.so.2 /usr/local/lib/ninfer-cuda/libnccl.so.2
COPY deploy/lib/libcudart.so.12 /usr/local/lib/ninfer-cuda/libcudart.so.12
ENV LD_LIBRARY_PATH=/usr/local/lib/ninfer-cuda
WORKDIR /workspace
EXPOSE 8080
STOPSIGNAL SIGTERM
CMD ["python3", "/usr/local/bin/tp2_proxy.py", "--help"]
