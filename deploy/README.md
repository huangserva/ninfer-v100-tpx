# deploy/

Production runs the binaries built on the host inside a small runtime container; nothing is
compiled in Docker.

- `collect_libs.sh <nccl-root> <cuda-root>`: copies `libnccl.so.2` (an NCCL with sm_70 kernels,
  e.g. unpacked from the `nvidia-nccl-cu12==2.21.5` wheel) and `libcudart.so.12` into `deploy/lib/`
  (git-ignored).
- `Dockerfile`: `ubuntu:24.04` + python3 + FFmpeg/curl runtime libraries + `ninfer`, `ninfer-serve`,
  `tp2_proxy.py` and the two libraries above. Build from the repository root after `cmake --build`:
  `docker build -f deploy/Dockerfile -t ninfer-v100-tpx:latest .`
- `start_tp2.sh`: starts the two-GPU service as one container (`--ipc=host` so the lockstep and
  mailbox files in `/dev/shm` are the host's). `P2P=auto` passes `--nccl-p2p auto` for NVLink
  machines. Rollback is `docker stop <name>` and start whatever you ran before; the two services
  share the port and GPU 0, so never run both at once.

Operational notes from the first deployment (2× V100 32G PCIe, 2026-09-26):

- switching from the single-GPU container interrupted the service for about 24 s; each restart of the pair takes about 25 s;
- with `--max-context 262144 --vision` each GPU uses about 20.4 GB;
- the lockstep check fired once during a 300-request evaluation run; the proxy restarted both ranks in 23 s and the request was retried by the client, no wrong output was served.
