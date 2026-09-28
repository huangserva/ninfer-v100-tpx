# 从零装到跑完整套测试

目标：一台干净的机器，两张 Tesla V100 32G，照这份清单走完，得到能和 README 成绩表逐格对照的数字。每一步都写了预期输出，对不上就停在那一步，不要带着问题往下走。带 NVLink 的机器多做第 10 节。

这份清单是照我们那台机器（Ubuntu 24.04，2× V100 32G PCIe，直连坏掉，无 NVLink）整理的。2026-09-28 在第二台机器（2× V100-SXM2-32GB，NVLink 六条）上从零走了一遍，碰到的偏差已经改进下面各节。再碰到新的偏差，请开 issue 或直接提 PR 改这份文件。

## 0. 机器

- 两张 Tesla V100 32G。16G 的放不下：双卡每张卡要占约 20GB。
- 看两卡之间怎么连：`nvidia-smi topo -m`。`NV1`/`NV2` 是 NVLink，`PHB`/`PIX`/`SYS` 是 PCIe。记下来，第 10 节要用。
- 驱动版本 570 以上（CUDA 12.8 的要求）。我们用的是 580.173.02 和 580.178.04。
- **先看显存频率**：`nvidia-smi -q -d CLOCK | grep -A3 "Max Clocks"`。同样叫 V100 32G，显存频率不一样：我们第一台是 Tesla PG500-216，显存 1107 MHz；第二台是 V100-SXM2-32GB，显存 877 MHz。生成速度主要受显存带宽限制，877 MHz 的卡单卡和双卡生成都要慢两成左右，这是卡的上限，不是装错了。下面对照表的数字都来自 1107 MHz 那台。
- Ubuntu 24.04。别的发行版没有试过。
- 磁盘留 80GB 以上：模型文件约 21.5GB，切成两份各 12.4GB，编译产物几个 GB。
- `df -h /dev/shm` 至少有 1GB 空闲。两个进程的核对文件和锁页信箱放在这里。

## 1. 依赖

```bash
sudo apt-get install -y cmake ninja-build g++ pkg-config git python3 python3-pip \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libcurl4-openssl-dev
```

CUDA 12.8 从 NVIDIA 官网装 toolkit（runfile 或 apt 都行），装到 `/usr/local/cuda-12.8`。**CUDA 13 不行**，它去掉了给 V100（sm_70）编译的能力。

没有 sudo 装不了 ffmpeg 的四个 `-dev` 包时，可以把 ffmpeg 6.1 编到自己的目录里，几分钟就好，不需要 nasm：

```bash
curl -LO https://ffmpeg.org/releases/ffmpeg-6.1.1.tar.xz && tar xf ffmpeg-6.1.1.tar.xz && cd ffmpeg-6.1.1
./configure --prefix=$HOME/ffmpeg --enable-shared --disable-static --disable-programs --disable-doc \
  --disable-x86asm --disable-avdevice --disable-postproc --disable-avfilter
make -j"$(nproc)" && make install
export PKG_CONFIG_PATH=$HOME/ffmpeg/lib/pkgconfig      # 第 3 步配置前设置
export LD_LIBRARY_PATH=$HOME/ffmpeg/lib                 # 运行 ninfer-serve 前设置，libswresample 不在 rpath 里
```

预期：

```bash
/usr/local/cuda-12.8/bin/nvcc --version | tail -1    # 含 cuda_12.8
cmake --version | head -1                            # 3.28 以上
g++ --version | head -1                              # 13 以上
```

## 2. 带 sm_70 内核的 NCCL

Ubuntu 24.04 自带的 NCCL 2.31 没有 Volta 内核，双卡第一次 all-reduce 就报 `named symbol not found`。直接解 pip 包。**别解到 `/tmp`**：编出来的 ninfer-serve 运行时按这个路径加载 libnccl，`/tmp` 重启就清空，服务就起不来了。

```bash
pip download --no-deps nvidia-nccl-cu12==2.21.5 -d $HOME/nccl && (cd $HOME/nccl && unzip -q *.whl)
ls $HOME/nccl/nvidia/nccl/include/nccl.h $HOME/nccl/nvidia/nccl/lib/libnccl.so.2   # 两个都要在
```

## 3. 编译

```bash
git clone https://github.com/huangserva/ninfer-v100-tpx.git && cd ninfer-v100-tpx
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DBUILD_TESTING=OFF \
  -DNINFER_NCCL_ROOT=$HOME/nccl/nvidia/nccl
cmake --build build-v100 -j"$(nproc)"
```

`-DBUILD_TESTING=OFF` 不能省。CUTLASS 被拉进来时会把这个开关打开，单元测试 `ninfer_cli_options_test` 链接失败（`undefined reference to ninfer::product::parse_log_level`），整个编译就停了。

配置阶段 CMake 会从 GitHub 拉 CUTLASS，机器要能访问 GitHub；不能直连就先 `export https_proxy=...`，或者先用镜像把它克隆下来，再告诉 CMake 用本地目录：

```bash
git clone --depth 1 -b v4.4.2 https://ghfast.top/https://github.com/nvidia/cutlass.git $HOME/cutlass
# 配置行加上  -DFETCHCONTENT_SOURCE_DIR_CUTLASS=$HOME/cutlass
```

44 线程的 E5-2696 v4 上完整编译约 5 分钟。

预期：`build-v100/apps/ninfer` 和 `build-v100/apps/ninfer-serve` 存在，`./build-v100/apps/ninfer-serve --help` 能打印帮助。配置阶段如果报 `NINFER_NCCL_ROOT must point to ...`，是第 2 步的路径不对。

## 4. 模型和切权重

```bash
pip install -U huggingface_hub
hf download neroued/Qwen3.8-27B-nvfp4-NInfer qwen3_8_27b_nvfp4.ninfer --local-dir models
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu    # 切权重用，CPU 版够
python3 -m tools.tp2.shard_qwen38_27b models/qwen3_8_27b_nvfp4.ninfer models-tp2/qwen3_8_27b_nvfp4
python3 -m tools.tp2.shard_qwen38_27b models/qwen3_8_27b_nvfp4.ninfer models-tp2/qwen3_8_27b_nvfp4 --verify
```

两条要分开跑：带 `--verify` 时脚本只核对、不切，第一次就带上会报找不到 `rank0.ninfer`。切一次约 2 分钟，内存峰值约 21GB；核对约 3 分钟。

国内机器访问不了 huggingface.co 时，下载前 `export HF_ENDPOINT=https://hf-mirror.com`。同一局域网里已经有一台装好的机器，直接 `rsync` 那个 21.5GB 的原始文件更省事，切分在本机做。

预期：我们用的模型文件是 21,492,695,040 字节（v2 artifact）；不一样说明上游更新了文件，照常切，数字可能略有出入。切完 `models-tp2/` 下两份各 12,380,094,464 字节，最后一行打印 `verify ok (452 split tensors)`。两份的 sha256：

```text
298f25c07aa1be463ed3f9ba0ebfe42de505c5cbc3c2bb7a104cd18e998c2d3b  qwen3_8_27b_nvfp4.rank0.ninfer
1ced2ce2d8960d1627ee648385e85597051b580095ea276a9dc12ae053f2980b  qwen3_8_27b_nvfp4.rank1.ninfer
```

## 5. 单卡冒烟

先确认单卡能跑，再上双卡，问题好定位。

```bash
echo test-key > api-key
CUDA_VISIBLE_DEVICES=0 ./build-v100/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --model-id qwen38-ninfer --api-key test-key --host 127.0.0.1 --port 18950 \
  --max-context 241664 --kv-capacity auto --max-concurrency 1 --prefill-chunk 2048 \
  --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft --seed 42 > single.log 2>&1 &
# 等 /v1/models 返回 200
python3 tools/bench/v100/decode_bench.py --url http://127.0.0.1:18950 --key test-key \
  --kind code --blocks 29 --seeds 1,2 --max-tokens 256
```

预期：`prompt_n` 7353；8K 代码单卡我们在显存 1107 MHz 的卡上测到 97 tok/s，落在 90 到 100 之间算正常；`accept` 0.7 上下。显存 877 MHz 的 SXM2 卡测到 80.6 tok/s（`ms_per_round` 36.7），也算正常，见第 0 节。单卡 nvfp4 最大上下文是 241664（236K），262144 放不下。测完 `kill %1`。

## 6. 双卡启动

```bash
python3 tools/tp2/tp2_proxy.py \
  --listen 127.0.0.1:18881 --binary build-v100/apps/ninfer-serve \
  --model-prefix models-tp2/qwen3_8_27b_nvfp4 \
  --api-key-file api-key --log-dir ./logs --gpus 0,1 --nccl-p2p off -- \
  --model-id qwen38-ninfer --max-context 262144 --kv-capacity auto \
  --max-concurrency 1 --prefill-chunk 2048 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --vision --seed 42
```

`--nccl-p2p off` 是默认值，给两张卡直连坏掉的机器用；有 NVLink 或直连正常的机器换成 `auto`（第 10 节）。

预期：代理输出依次有 `listening on 127.0.0.1:18881`、`started ranks gen=1`、`both ranks ready gen=1`。`nvidia-smi` 两张卡各占约 20GB。`logs/tp2_rank0.log` 和 `tp2_rank1.log` 里没有 `FATAL`、`TP2 LOCKSTEP FAILURE`。

```bash
curl -s -H "Authorization: Bearer test-key" http://127.0.0.1:18881/v1/models    # 200，列出 qwen38-ninfer
```

## 7. 功能测试

```bash
python3 tools/bench/v100/tp2_scenarios.py --port 18881 --model qwen38-ninfer --key test-key \
  --lockstep-file /dev/shm/ninfer_tp2_lockstep
```

12 个场景：普通请求、流式、不带种子采样、流式中途断开、读 prompt 中途断开、客户端超时、写满截断、提前停止、带图、带图流式中途断开、Anthropic 接口、Responses 接口。每个场景之后发一条小请求并读两张卡的核对计数。

预期：12 行都是 `[PASS]`，每行 `units=(a, b)` 两个数相等，最后一行 `ALL PASS`。

## 8. 故障注入

```bash
python3 tools/bench/v100/tp2_fault.py kill --port 18881 --model qwen38-ninfer --key test-key   # 杀掉 1 号进程
python3 tools/bench/v100/tp2_fault.py stop --port 18881 --model qwen38-ninfer --key test-key   # 冻住 1 号进程
```

预期：进行中的请求得到 503 或连接被断开；代理日志出现 `RESTART both ranks`，约 25 秒后 `both ranks ready gen=2`；脚本最后一行 `[PASS] kill_recovery`（`stop` 那次要等看门狗的 180 秒）。

## 9. 速度矩阵

README 成绩表就是这个脚本跑出来的格式：8K、32K、128K、256K，代码和中文各一份，每格先冷启动一次（完整读 prompt），再换随机种子命中缓存一次，每次生成 1024 个 token。

```bash
URL=http://127.0.0.1:18881 MODEL=qwen38-ninfer KEYFILE=api-key tools/bench/v100/matrix.sh results/tp2
cat results/tp2/table.md
```

再跑一遍 README 的头条数字（186K 代码、193K 中文，4 个种子平均）：

```bash
MODE=robust URL=http://127.0.0.1:18881 MODEL=qwen38-ninfer KEYFILE=api-key tools/bench/v100/matrix.sh results/tp2-robust
```

单卡的成绩表用第 5 节的启动方式（`--max-context 241664`，256K 那档跑不了）对着同一个脚本跑，`URL` 指到 18950。

对照数字（2× V100 32G PCIe，无 NVLink，`--nccl-p2p off`，双卡）：

| 上下文 | 代码 冷 / 缓存 tok/s | 中文 冷 / 缓存 tok/s | 首字等待（冷） |
|---|---:|---:|---:|
| 8K | 140.2 | 105.8 | 6.2 s |
| 32K | 141.2 / 134.7 | 94.5 / 95.0 | 29.4 s |
| 128K | 113.9 / 109.1 | 81.1 / 83.7 | 152 s |
| 256K | 94.7 / 91.0 | 71.0 / 67.7 | 393 s |
| 186K 代码 / 193K 中文，4 种子平均 | 101.9 | 73.4 | 250 s |

单卡（本仓库内核）：186K 代码 56，193K 中文 43，8K 代码 97 / 中文 73，186K 首字等待 379 秒。

第二台机器的数字（2× V100-SXM2-32GB，显存 877 MHz，NVLink 六条，`--nccl-p2p auto`，信箱默认开，2026-09-28）：

| 上下文 | 代码 冷 / 缓存 tok/s | 中文 冷 / 缓存 tok/s | 首字等待（冷） |
|---|---:|---:|---:|
| 8K | 108.0 / 130.6 | 99.4 / 96.3 | 4.3 s |
| 32K | 130.5 / 130.6 | 91.8 / 91.5 | 21.7 s |
| 128K | 104.5 / 110.0 | 80.5 / 74.7 | 118 s |
| 256K（260,173 / 260,854 token） | 80.6 / 86.0 | 65.3 / 63.9 | 346 s |
| 186K 代码 / 193K 中文，4 种子平均 | 96.6 | 69.0 | 219 s |

同一台机器单卡 8K 代码 80.6 tok/s。双卡和第一台差得比单卡少：第一台双卡生成比这台快 0 到 10%，单卡快 20%。

怎么看差异：生成速度看 `ms_per_round`（每轮毫秒）比看 tok/s 稳，tok/s 里混着猜中率的随机波动；同一档差 5% 以内当作一样。首字等待是 `prefill_s`，两次跑差 2% 以内。

## 10. NVLink 机器专项

前面 1 到 9 节先照 `--nccl-p2p off` 跑完一遍，拿到和 PCIe 机器同口径的数字，再做下面三组对比。每组只改一个变量。

**直连开关。** 重启代理，把 `--nccl-p2p off` 换成 `--nccl-p2p auto`，再跑第 9 节的矩阵。想确认 NCCL 真的走了 NVLink，启动代理前设置下面三个变量，再发一条读长提示词的请求（小消息走信箱，不经过 NCCL）：

```bash
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT NCCL_DEBUG_FILE=$PWD/logs/nccl.%p.log
grep -h " via " logs/nccl.*.log | head
```

走 NVLink 或直连是 `via P2P/CUMEM`，走主机内存中转是 `via SHM/...`。不设 `NCCL_DEBUG_FILE` 时 NCCL 的输出混在程序标准输出里被缓冲，`tp2_rank*.log` 里只剩开头几行，1 号进程一行都没有，看不出走了哪条路。

先单独确认两张卡的直连真能用，别只看 `nvidia-smi topo -m` 报 `NV#`：我们第一台机器也报直连可用，实际 4KB 拷贝要 49 毫秒。用 `cudaMemcpyPeerAsync` 测一下，NVLink 六条的机器上 4KB 约 3 微秒、256MB 约 145GB/s。V100 这版驱动下 `nvidia-smi nvlink -gt d` 的流量计数全是 `N/A`，不能拿它判断。预期：首字等待明显缩短，生成速度变化不到一成（我们按 PCIe 机器的时间线估算过：每轮 33 毫秒里卡间通信只占 2.2 毫秒）。

**小消息走哪条路。** 默认 128KB 以下的 all-reduce 走 `/dev/shm` 上的锁页内存信箱，不走 NCCL。启动代理前 `export NINFER_TP_MAILBOX=0` 让它全走 NCCL，再跑 `MODE=robust`。PCIe 机器上信箱快（一次 40KB 从 26 微秒降到 16），NVLink 机器上 NCCL 直连可能反过来更快，两组都记。

**长段开关。** `NINFER_SM70_LONG_SPLIT_KEYS` 和卡间怎么连无关，不用重测。

第二台机器（NVLink 六条）的结果，头条两档 4 种子平均：

| 配置 | 186K 代码 tok/s | 193K 中文 tok/s | 每轮 ms | 186K 首字等待 |
|---|---:|---:|---:|---:|
| `--nccl-p2p off` | 98.5 | 70.2 | 33.8 | 255 s |
| `--nccl-p2p auto` | 96.6 | 69.0 | 34.5 | 219 s |
| `auto` + `NINFER_TP_MAILBOX=0` | 95.3 | 67.2 | 35.0 | 206 s |

打开直连后，首字等待在 8K 从 6.6 秒降到 4.3 秒，32K 从 32.4 降到 21.7，128K 从 152 降到 118，256K 从 394 降到 346。生成每轮耗时的差别在 2% 以内，落在噪声范围里。关掉信箱后生成反而慢一点，所以 NVLink 机器用 `auto`，信箱保持默认开。

把三组的 `table.md` 和 `nvidia-smi topo -m` 的输出一起提交到 issue，我们汇总进 README。

## 11. 容器化（可选）

生产用的是把宿主机编好的程序打包成运行镜像，镜像里不编译。

```bash
deploy/collect_libs.sh /tmp/nccl/nvidia/nccl /usr/local/cuda-12.8
docker build -f deploy/Dockerfile -t ninfer-v100-tpx:latest .
mkdir -p /srv/ninfer && cp -r models-tp2 /srv/ninfer/ && cp api-key /srv/ninfer/ && mkdir -p /srv/ninfer/logs
ROOT=/srv/ninfer PORT=18881 P2P=off deploy/start_tp2.sh      # NVLink 机器 P2P=auto
docker logs -f ninfer-tpx                                     # 等 both ranks ready
```

之后第 7 到 9 节的测试原样对着 18881 跑，`tp2_scenarios.py` 加 `--container ninfer-tpx --lockstep-file /dev/shm/ninfer_tpx_lockstep`。

没有 docker、也没有 sudo 时，可以直接在宿主机跑代理，用用户自己的 crontab 开机启动，外面套一层循环在代理退出后重启。第二台机器就是这么跑的：

```bash
# serve_forever.sh：先清掉可能残留的 rank 进程（代理崩了它们还占着每张卡 20GB），再起代理，退出后 10 秒重来
while true; do
  pkill -9 -f "^$PWD/build-v100/apps/ninfer-serve"; sleep 2
  P2P=auto ./run_tp2.sh >> logs/proxy.log 2>&1; rc=$?
  echo "$(date '+%F %T') run_tp2.sh exited with $rc" >> logs/proxy.log; sleep 10
done
# crontab -e 加一行
@reboot /usr/bin/flock -n /dev/shm/ninfer-tp2.lock /path/to/serve_forever.sh
```

`run_tp2.sh` 就是第 6 节那条代理命令，另外 `export LD_LIBRARY_PATH` 指向 ffmpeg 和 NCCL 的库目录。实测把代理强杀后，约 35 秒恢复服务。

## 12. 常见问题

- 第一次 all-reduce 报 `named symbol not found`：用了没有 sm_70 内核的 NCCL，回第 2 步。
- `cudaHostRegister` 报错：`--lockstep-file` 不在 `/dev/shm` 这类内存文件系统上。
- 只有一张卡起来、代理一直 `startup timed out`：看 `logs/tp2_rank1.log` 的最后 20 行；多数是那张卡上还有别的进程占着显存。
- 刚停掉代理马上重启，报 `OSError: [Errno 98] Address already in use`：旧代理还没退完。等 `ss -ltn | grep 18881` 没有输出再起。
- 矩阵里 256K 那档报 `context_length_exceeded`：用的是旧版 `ctx_prompt.py`，256K 档在 30 万编号偏移下块数太多。新版已改成 939 / 1516 块，约 26 万 token。
- 打印 `TP2 LOCKSTEP FAILURE` 然后两个进程一起重启：两卡算出的 token 对不上。我们线上 300 题的评测里出现过一次，根因没有查到；出现频率高于每几百个请求一次，请带着 `logs/` 开 issue。
- 数字比对照表低两成以上：先查 `nvidia-smi -q -d CLOCK` 里显卡有没有降频，再查 `--kv-dtype int8`、`--draft-tokens 3`、`--prefill-chunk 2048` 是不是和第 6 节一样。

## 13. 记录什么

一次完整复现请留下：机器（卡、驱动、`nvidia-smi topo -m`）、第 3 步的 CMake 配置行、第 4 步两份文件的大小和 `verify ok` 那一行、第 7 到 9 节的完整输出、第 10 节三组 `table.md`。这些东西够我们把你的机器加进成绩表。
