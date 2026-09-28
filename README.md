# ninfer-v100-tpx

Qwen3.8-27B on Tesla V100s: rewritten sm70 attention kernels + N-GPU tensor parallel for [NInfer](https://github.com/Neroued/ninfer), built on the [ninfer-v100](https://github.com/geoffwatts/ninfer-v100) port. Two GPUs today (`x` is reserved for more), ~100 tok/s at 186K context on 2× V100 32G over plain PCIe, no NVLink. Full API contract of the single-GPU build (OpenAI / Anthropic-compatible HTTP, MTP, INT8 KV, prefix cache, vision).

中文说明在下面。

## English summary

| Scenario (Qwen3.8-27B nvfp4, MTP draft 3) | 1× V100, upstream kernels | 1× V100, this repo | 2× V100, this repo (TP2) |
|---|---:|---:|---:|
| 186K-token code prompt, decode | 36 tok/s | 56 | **101.9** |
| 193K-token Chinese prompt, decode | 26 | 43 | **73.4** |
| 8K code / Chinese, decode | 79 / 58 | 97 / 73 | 140 / 106 |
| 186K first-token latency | 455 s | 379 s | 250 s |

Head-to-head with one RTX 4090 48G (same prompts, same settings; 4090 runs stock NInfer with groupwise-int weights): decode is on par or slightly faster on the two V100s at every context (256K code 94.7 vs 80.9 tok/s, Chinese 71.0 vs 67.4), while the 4090 reads prompts about 1.85× faster (256K first token 210 s vs 393 s). Full table in the Chinese section. Decode numbers at 186K/193K are the mean of 4 seeds × 1024 generated tokens on one fixed prompt; 8K numbers are single 256-token runs. Quality: 237/300 on our public [model-evaluation](https://github.com/huangserva/model-evaluation) core-300 set versus 245 (llama.cpp Q4_K_M) and 238 (Q8_0) on a 4090, differences within noise (paired McNemar p=0.20 / p=1.0).

A from-scratch install and test checklist is in [docs/reproduce.md](docs/reproduce.md); the prompt generator and benchmark scripts behind every number here are in [tools/bench/v100/](tools/bench/v100/). The Volta substitutes for the hardware V100 lacks (llama.cpp's Volta flash kernel for prefill, an m8n8k4-based decode attention kernel, software NVFP4 dequant into FP16 tensor cores) come from upstream ninfer-v100. This repo rewrites the slowest of them, the INT8 decode attention kernel, retunes the prefill kernel and adds tensor parallel; the model's attention math is unchanged. What changed versus upstream is listed in [CHANGES.md](CHANGES.md); the design notes are in [docs/tp2-design-notes.md](docs/tp2-design-notes.md); the upstream README is kept as [docs/UPSTREAM-README.md](docs/UPSTREAM-README.md).

## 这是什么

NInfer 官方只支持 RTX 5090。社区的 [ninfer-v100](https://github.com/geoffwatts/ninfer-v100) 把它移植到了 V100（sm_70）。这个仓库从那个移植版的提交 `b37d0dd` 开始，做了两件事：

1. **单卡提速**。重写了长上下文下最慢的那个 attention 内核，读 prompt 用的内核重新调参。186K 上下文的生成速度从 36 tok/s 提到 56，首字等待从 455 秒降到 379 秒。这部分和卡的数量无关，单卡也能直接用。
2. **双卡张量并行（TP2）**。每张卡一个 `ninfer-serve` 进程，各放一半权重，前面一个代理把同一个请求发给两边，每层算完用 NCCL 或主机锁页内存把两边的一半加起来。186K 上下文的生成速度到 101.9 tok/s，首字等待 250 秒。

两天的排查过程、每一步为什么这么做，写在了一篇长文里（链接后补）。

## 成绩

单位 tok/s。186K / 193K 的生成速度是同一个 prompt 换 4 个随机种子、每次生成 1024 个 token 的平均值；8K 是单次 256 个 token。

| 场景 | 单卡，上游内核 | 单卡，本仓库 | 双卡，本仓库 |
|---|---:|---:|---:|
| 186K 代码，生成 | 36 | 56 | **101.9** |
| 193K 中文，生成 | 26 | 43 | **73.4** |
| 8K 代码 / 中文，生成 | 79 / 58 | 97 / 73 | 140 / 106 |
| 186K 首字等待 | 455 秒 | 379 秒 | 250 秒 |
| 26 万 token（喂满 262,144），生成 | 放不下 | 放不下 | 代码 91–95，中文 68–71，首字约 393 秒 |

代码和中文每轮耗时一样（双卡 32.4 毫秒），中文慢只是因为这份中文测试文档的猜词命中率低（0.46 对 0.77）。

和一张 4090 48G 正面比（同一份 prompt、同样参数，冷启动 / 命中缓存，tok/s；4090 跑 NInfer 官方程序、groupwise-int 权重、单卡）：

| 上下文 | V100 双卡 代码 | 4090 单卡 代码 | V100 双卡 中文 | 4090 单卡 中文 | 首字等待 V100 / 4090 |
|---|---:|---:|---:|---:|---:|
| 8K | 140.2 | 108.6 | 105.8 | 112.2 | 6.2 s / 3.5 s |
| 32K | 141.2 / 134.7 | 112.5 / 119.3 | 94.5 / 95.0 | 94.3 / 89.9 | 29.4 s / 16.5 s |
| 128K | 113.9 / 109.1 | 99.8 / 105.2 | 81.1 / 83.7 | 77.6 / 78.1 | 152 s / 83 s |
| 256K | 94.7 / 91.0 | 80.9 / 82.2 | 71.0 / 67.7 | 67.4 / 65.3 | 393 s / 210 s |

生成速度双卡 V100 各档差不多或略快（每轮耗时 256K 时 35 ms 对 37.7 ms），读 prompt 4090 快约 1.85 倍。

能力没有变化：在我们的公开题库 [model-evaluation](https://github.com/huangserva/model-evaluation) 上，双卡 nvfp4 版 300 题答对 237，以前 4090 上 llama.cpp 的 4 位版 245、8 位版 238，逐题配对比较差距在随机误差内。

## 改了什么

先说清楚起点。V100 缺的那些指令（ldmatrix、m16n8k16、cp.async、bf16 张量核）的替代方案，包括用 llama.cpp 的 Volta flash 内核读 prompt、按 m8n8k4 重建的生成 attention 内核、软件 NVFP4 解压进 FP16 张量核，都是上游 ninfer-v100 做的。本仓库改的是这套替代方案里最慢的那段生成 attention 内核，调了读 prompt 内核的参数，再加上双卡。模型的 attention 算法没有改。

本仓库的改动分三块，都在 [CHANGES.md](CHANGES.md) 里按文件列出。

**sm70 内核（单卡也生效）**

- `src/ops/softmax_attention/dense/causal_cache/small_t_i8_volta_v2.cuh`：新的 INT8 KV 生成 attention 内核。原版每批取 16 个 key，4 个 warp 各自把这 16 个 key 的 QK 全算一遍，每批同步两次，186K 下只跑出约 140GB/s。新版 8 个 warp、每批 64 个 key，每个 warp 只算自己那 8 个 key 的 QK，通过共享内存交换行最大值和概率，再各自做 D/8 列的 PV，每 64 个 key 同步 3 次；下一批 K/V 提前读进寄存器；int8 转 fp16 用 PRMT 字节重排代替 Volta 上很慢的转换指令。186K、T=4 时每次调用从 2.79 毫秒降到 1.29。输入输出契约和原版一致，`NINFER_SM70_ATTN_V2=0` 切回原版。
- `src/ops/launcher/gqa_attention_volta_flash.cu`：读 prompt 用的 flash 内核配置重调（8 warp、每批 64 key、K/V 每次读入减半），131K 下 1024 token 的处理从 119 毫秒降到 88.4 毫秒。
- 残差投影的激活先转成 fp16 再进 QPN 内核（原来在内层循环里现场转，只有约 350GB/s）；NVFP4 内核一次先发 8 组读取；残差相加并进输出阶段；attention 分段合并的小内核从 29 微秒降到 13。

**双卡目标 `src/targets/qwen3_6_27b_tp2/`**

- 模型变体：每卡 12 个 q 头、2 个 kv 头，GDN 的 k/v 头减半，MLP 中间维度 8704。
- `tp2_comm.cpp`：每层三类输出投影（attention 输出、GDN 输出、MLP down）之后的 all-reduce，放在 CUDA Graph 里。rank 0 把自己的一半加到残差上，rank 1 直接覆盖残差，再对残差做一次 all-reduce。
- `tp2_mailbox.cu`：128KB 以下的消息不走 NCCL，改走 `/dev/shm` 上的锁页内存，每个线程块一个标志位，两卡按固定顺序相加，结果逐位相同。40KB 一次从 26 微秒降到 16。`NINFER_TP_MAILBOX=0` 退回全走 NCCL。两卡直连能用的机器（NVLink）上，信箱改放在显存里：每个进程通过 CUDA IPC 映射对方的收件箱，把数据和标志直接写进对方显存，只在自己显存上等；启动时自检并和主机信箱比一次速度，赢了才启用。NVLink 六条的机器上 40KB 一次 7.8 微秒（LL 版 5.6，启动时在 Graph 里比一次取快的），词表拼接从 335 微秒降到 27，生成快 9% 到 10%。
- LM head 和草稿 head 按词表切到两张卡，通过同一条路拼回完整 logits。`NINFER_TP_SHARD_HEADS=0` 关闭。
- 散在 `src/ops/` 各处的十几处半宽形状登记（fp8 / nvfp4 / gdn / attn_input），以及 `d256-h12-kv2` 的 attention 几何路由。

**运行与安全**

- `src/runtime/engine/tp_lockstep.h`：每个 prefill 块、每轮解码之后，两个进程通过一个共享内存文件交换本轮的 token 数、token 哈希、位置和取消标志。对不上就打印 `TP2 LOCKSTEP FAILURE` 退出；取消取两边的「或」，两张卡在同一轮停下。这条路故意不走 NCCL。
- `tools/tp2/tp2_proxy.py`：前置代理兼看门狗。启动两个 rank，把生成请求排队一次一个，同一个请求发给两边，只把 rank 0 的回复转给客户端；客户端断开就取消；任一进程退出，或请求进行中核对计数 180 秒不动，两个进程一起重启并给客户端返回 503。
- `tools/tp2/shard_qwen38_27b.py`：把一个 Qwen3.8-27B nvfp4 artifact 切成两份 rank artifact。q/k/v/gate/up 这类按输出行切，三类输出投影和 down 按输入列切，词嵌入、LM head、草稿 head、归一化、视觉部分整份复制，`--verify` 逐位核对能拼回原文件。

## 前提

- 两张 Tesla V100 32G（PCIe 即可；这套代码就是在直连坏掉的机器上开发的，代理会设 `NCCL_P2P_DISABLE=1`）。16G 的 V100 没有测过，按每卡约 20GB 的占用算放不下。
- CUDA 12.8（CUDA 13 去掉了 Volta 的离线编译），CMake 3.28+，Ninja，C++20 编译器，FFmpeg 开发库，libcurl。和上游一样。
- 一份带 sm_70 内核的 NCCL。Ubuntu 24.04 自带的 NCCL 2.31 没有 Volta 内核，第一次 all-reduce 就报 `named symbol not found`。可以直接解开 pip 包：

```bash
pip download --no-deps nvidia-nccl-cu12==2.21.5 -d /tmp/nccl && cd /tmp/nccl && unzip -q *.whl
# 之后 /tmp/nccl/nvidia/nccl 下有 include/nccl.h 和 lib/libnccl.so.2
```

- 切权重的工具要 Python 3.12 和 torch（CPU 版就够）。
- 模型：Hugging Face 上 neroued 发布的 `Qwen3.8-27B-nvfp4-NInfer`（v2 artifact）。

## 编译

```bash
git clone https://github.com/huangserva/ninfer-v100-tpx.git && cd ninfer-v100-tpx
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DNINFER_NCCL_ROOT=/tmp/nccl/nvidia/nccl
cmake --build build-v100 -j
```

产物是 `build-v100/apps/ninfer` 和 `build-v100/apps/ninfer-serve`。单卡用法和上游完全一样，新内核默认开启。

## 切权重

```bash
python -m tools.tp2.shard_qwen38_27b \
  /path/to/qwen3_8_27b_nvfp4.ninfer /path/to/models-tp2/qwen3_8_27b_nvfp4 --verify
# 得到 qwen3_8_27b_nvfp4.rank0.ninfer 和 qwen3_8_27b_nvfp4.rank1.ninfer，各约 12.4GB
```

## 运行双卡

```bash
python3 tools/tp2/tp2_proxy.py \
  --listen 127.0.0.1:18881 --binary build-v100/apps/ninfer-serve \
  --model-prefix /path/to/models-tp2/qwen3_8_27b_nvfp4 \
  --api-key-file /path/to/api-key --log-dir ./logs --gpus 0,1 -- \
  --model-id qwen38-ninfer --max-context 262144 --kv-capacity auto \
  --max-concurrency 1 --prefill-chunk 4096 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft --vision --seed 42
```

`--` 后面的参数原样传给两个 `ninfer-serve`。日志里出现 `both ranks ready` 就可以用了，接口和单卡版一样。几点说明：

- 两个进程必须用同一个随机种子（`--seed`），否则同一个概率表会抽出不同的字，核对会失败。客户端请求里带了 seed 就按客户端的来。
- 一次只处理一个请求，最多排队 4 个（`--max-waiting`），排队超过 `--pending-timeout` 秒返回 503。
- 两张卡各占约 20GB 显存，`--max-context 262144` 加 `--vision` 放得下。
- 有 NVLink 或者两卡直连正常的机器加 `--nccl-p2p auto`，让 NCCL 自己选路。默认 `off`，即给两个进程设 `NCCL_P2P_DISABLE=1`，这是开发机（直连坏掉）需要的。
- 容器化部署见 [deploy/README.md](deploy/README.md)：先在宿主机编译，镜像只做打包。

## 从零复现与测试

从装依赖到跑完整套测试的清单在 [docs/reproduce.md](docs/reproduce.md)，每一步写了预期输出。成绩表里每个数字背后的 prompt 生成器和测速脚本在 [tools/bench/v100/](tools/bench/v100/)：`matrix.sh` 跑出和成绩表同样格式的表，`tp2_scenarios.py` 和 `tp2_fault.py` 是上线前过的功能测试和故障注入。

## 运行时开关

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `NINFER_SM70_ATTN_V2` | 开 | 0 切回上游的生成 attention 内核 |
| `NINFER_SM70_LONG_SPLIT_KEYS` | 双卡 1920，单卡沿用上游的 480 | 长上下文 attention 每段的 key 数，段太碎时合并步骤会成为大头。单卡设成 1920 时这个内核从 1402 微秒降到 1130（186K，T=4），还没有改成默认 |
| `NINFER_TP_MAILBOX` | 开 | 0 让小消息也走 NCCL |
| `NINFER_TP_NVLINK` | 自动 | 直连能用且比主机信箱快时把信箱放进显存（CUDA IPC）；0 强制关 |
| 代理 `--max-inflight N` | 1 | 同时生成的请求数，两个 rank 要用 `--max-concurrency >= N` 启动。两卡在核对机制里协商准入：每个单元交换各自队列长度，按少的放行，谁都不按自己的到达时间做决定 |
| `NINFER_TP_SHARD_HEADS` | 开 | 0 让两张卡各算完整词表 |
| `NINFER_MTP_ATTN_WINDOW` | 0（关） | 猜词时只看最近 N 个 token。193K 中文能从 73 到 79，但代码的命中率从 0.72 掉到 0.59，所以默认关。NVLink 机器上复测 4096：186K 中文快 3% 到 6%，128K 中文反而慢 3%，仍然默认关 |
| `NINFER_TP_LOCKSTEP_TIMEOUT_S` | 300 | 等对方核对的超时 |

`NINFER_TP_RANK`、`NINFER_TP_ID_FILE`、`NINFER_TP_LOCKSTEP_FILE`、`NINFER_TP_MAILBOX_FILE` 由代理设置，不用手动管。

## 已知问题

- 两卡偶发结果不一致。上线后的一次 300 题评测里出现过一次，被每轮核对拦下，两个进程重启 23 秒恢复，没有输出错的内容。根因还没有查到。
- 只实现了 2 卡。仓库名里的 x 是给后面留的，4 卡的切分和 all-reduce 还没有做。
- 中文长上下文稳定在 73 左右，没有到 80。剩余的耗时里 attention 10.5 毫秒是大头，Volta 每个 SM 只能驻留一个线程块，几版重写都只快了 3% 到 14%。
- 主机锁页内存那块必须放在 `/dev/shm` 这类内存文件系统上，放磁盘上的文件 `cudaHostRegister` 会拒绝。
- 代理是 Python 标准库写的单文件脚本，够用，但不是一个精致的服务框架。
- 一个请求在读长提示词的时候不放新请求进来（引擎的调度规则），128K 的提示词会让后面的人等 100 秒；读完之后两条一起生成。

## 路线图

- ~~NVLink 机器上的实测。~~ 已做（2× V100-SXM2，NVLink 六条，见 `docs/reproduce.md` 第 10 节）：NCCL 走直连只缩短首字等待；把信箱搬进显存后生成快 9%，186K 代码 105.7 tok/s、193K 中文 73.3，每轮 31.5 毫秒。这台卡的显存是 877 MHz（开发机 1107 MHz），单卡生成慢两成，双卡靠直连信箱反超。
- 并发 2 已做（2× V100-SXM2 上：两个人各约 100 tok/s，四个突发请求 20 轮无一失步；见 `docs/reproduce.md` 第 7 节）。并发 3 到 4 没有测。
- 4 卡。
- 中文长上下文到 80。

## 致谢与许可

- [Neroued/ninfer](https://github.com/Neroued/ninfer)：NInfer 本体，Apache-2.0。
- [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100)：V100 移植，本仓库的起点，Apache-2.0。
- [plus1998/NInfer-V100-Duo](https://github.com/plus1998/NInfer-V100-Duo)：双卡参考实现，我们测了它的基线并用 nsys 拆过它的时间线。
- [ValerioDolci/ninfer-tp2](https://github.com/ValerioDolci/ninfer-tp2)：锁页内存信箱的思路来自这里。
- [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)：对照测试用的另一个 V100 引擎。
- 模型权重来自 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 和 neroued 发布的 NInfer artifact。

本仓库沿用 [Apache License 2.0](LICENSE)，修改说明见 [NOTICE](NOTICE) 和 [CHANGES.md](CHANGES.md)。
