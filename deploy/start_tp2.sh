#!/usr/bin/env bash
# Start the two-GPU service as one container: tp2_proxy.py launches one ninfer-serve per GPU and
# serves the OpenAI/Anthropic-compatible API on $PORT with the same contract as the single-GPU build.
#
#   ROOT    directory holding models-tp2/ (the two rank artifacts), api-key, logs/
#   NAME    container name           (default ninfer-tpx)
#   PORT    host port                (default 18881, bound to 127.0.0.1)
#   IMAGE   image built from deploy/Dockerfile
#   MAXCTX  --max-context; 262144 fits on 2x V100 32G with --vision
#   P2P     off (default, NCCL_P2P_DISABLE=1) or auto (NVLink / working PCIe P2P)
set -euo pipefail
ROOT=${ROOT:?set ROOT to the deployment directory}
NAME=${NAME:-ninfer-tpx}
PORT=${PORT:-18881}
IMAGE=${IMAGE:-ninfer-v100-tpx:latest}
P2P=${P2P:-off}
MAXCTX=${MAXCTX:-262144}
if docker container inspect "$NAME" >/dev/null 2>&1; then
  docker start "$NAME"; exit 0
fi
docker run -d --name "$NAME" --restart unless-stopped --gpus '"device=0,1"' --ipc=host \
  -p 127.0.0.1:${PORT}:8080 \
  -v "$ROOT/models-tp2:/models:ro" \
  -v "$ROOT/api-key:/run/secrets/ninfer_api_key:ro" \
  -v "$ROOT/logs:/var/log/ninfer" \
  "$IMAGE" \
  python3 -u /usr/local/bin/tp2_proxy.py \
    --listen 0.0.0.0:8080 --binary /usr/local/bin/ninfer-serve \
    --model-prefix /models/qwen3_8_27b_nvfp4 \
    --api-key-file /run/secrets/ninfer_api_key --log-dir /var/log/ninfer \
    --gpus 0,1 --stall-seconds 180 --lockstep-timeout 300 \
    --lockstep-file /dev/shm/ninfer_tpx_lockstep --id-file /dev/shm/ninfer_tpx.id \
    --max-waiting 4 --pending-timeout 600 --nccl-p2p "$P2P" -- \
    --model-id qwen38-ninfer \
    --max-context "$MAXCTX" --kv-capacity auto \
    --max-concurrency 1 --max-pending-requests 4 \
    --pending-timeout-ms 3600000 \
    --prefill-chunk 2048 --kv-dtype int8 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --vision --preserve-thinking --seed 42
