#include "targets/qwen3_6_27b_tp2/impl/tp2_comm.h"
#include "runtime/engine/tp_lockstep.h"
#include "targets/qwen3_6_27b_tp2/impl/tp2_mailbox.h"

#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_launch.h"
#include "ninfer/ops/residual_add.h"

#include <cuda_bf16.h>

#include <nccl.h>

#include <atomic>
#include <cstddef>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2 {
namespace {

ncclComm_t g_comm = nullptr;
int g_rank        = -1;
MailboxShared* g_mailbox_dev = nullptr;  // device alias of the shared host block (nullptr: NCCL only)
MailboxShared* g_mailbox_host = nullptr; // host alias of the same block (bootstrap channel)
std::uint64_t* g_mailbox_steps = nullptr;
std::uint64_t* g_gather_steps  = nullptr;
Inbox* g_inbox_mine            = nullptr;  // NVLink path (nullptr: host mailbox / NCCL)
Inbox* g_inbox_peer            = nullptr;  // the peer's inbox, IPC-mapped
std::uint64_t* g_nvlink_steps  = nullptr;
std::uint64_t* g_nvlink_gather_steps = nullptr;
__nv_bfloat16* g_head_local    = nullptr;  // [N/2, T] scratch for the sharded heads
void* g_head_act               = nullptr;  // fp16 staging of the head activation (K x T <= 5120 x 32)

// Row slice of a head weight. Q4G64 row-split and FP8 row-scaled (plain or QPN-prepacked, whose
// 32-row CTA tiles are laid out in row order) are both addressable by a row offset.
bool head_row_view(const Weight& w, std::int32_t row_begin, std::int32_t rows, Weight& out) {
    out = w;
    if (w.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        if (row_begin % 32 != 0) { return false; }
        out.qdata  = static_cast<const std::byte*>(w.qdata) + static_cast<std::uint64_t>(row_begin) * w.k;
        out.scales = static_cast<const std::byte*>(w.scales) + static_cast<std::uint64_t>(row_begin) * 2;
        out.scale_ne[0] = rows;
        for (int d = 1; d < 4; ++d) { out.scale_nb[d] = static_cast<std::int64_t>(rows) * 2; }
    } else if (w.qtype == QType::Q4G64_F16S && w.layout == QuantLayout::RowSplit) {
        const std::uint64_t groups = static_cast<std::uint64_t>(w.padded_shape[1] / w.group);
        out.qdata  = static_cast<const std::byte*>(w.qdata) + static_cast<std::uint64_t>(row_begin) * groups * 32;
        out.scales = static_cast<const std::byte*>(w.scales) + static_cast<std::uint64_t>(row_begin) * groups * 2;
    } else {
        return false;
    }
    out.n               = rows;
    out.shape[0]        = rows;
    out.padded_shape[0] = rows;
    return true;
}

std::uint64_t id_nonce(const ncclUniqueId& id) {
    std::uint64_t h = 1469598103934665603ULL;
    for (char c : id.internal) { h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ULL; }
    return h | 1;  // never zero (the unset value)
}

// Rank 0 creates and zeroes the block, then publishes this run's nonce; rank 1 attaches only when it
// sees the same nonce, so a stale file from an earlier run can never be mistaken for this one.
void init_mailbox(int rank, const std::string& path, std::uint64_t nonce) {
    const std::size_t bytes = (sizeof(MailboxShared) + 4095) / 4096 * 4096;
    void* host = nullptr;
    if (rank == 0) {
        // Build the block under a private name and publish it with an atomic rename: the peer may
        // still have a previous run's file mapped, and truncating that inode in place would SIGBUS it.
        const std::string tmp = path + ".tmp";
        ::unlink(tmp.c_str());
        const int fd = ::open(tmp.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd < 0 || ::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
            throw std::runtime_error("TP2: cannot create mailbox " + tmp);
        }
        host = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ::close(fd);
        if (host == MAP_FAILED) { throw std::runtime_error("TP2: mailbox mmap failed"); }
        std::memset(host, 0, bytes);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *reinterpret_cast<volatile std::uint64_t*>(&static_cast<MailboxShared*>(host)->magic) = nonce;
        if (std::rename(tmp.c_str(), path.c_str()) != 0) {
            throw std::runtime_error("TP2: cannot publish mailbox " + path);
        }
    } else {
        // Re-open until the file carries this run's nonce (an older file is simply skipped).
        for (int attempt = 0;; ++attempt) {
            if (attempt > 6000) { throw std::runtime_error("TP2: timed out waiting for mailbox"); }
            const int fd = ::open(path.c_str(), O_RDWR);
            struct stat st {};
            if (fd >= 0 && ::fstat(fd, &st) == 0 && static_cast<std::size_t>(st.st_size) == bytes) {
                void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                ::close(fd);
                if (p != MAP_FAILED) {
                    if (*reinterpret_cast<volatile std::uint64_t*>(&static_cast<MailboxShared*>(p)->magic) == nonce) {
                        host = p;
                        break;
                    }
                    ::munmap(p, bytes);
                }
            } else if (fd >= 0) {
                ::close(fd);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    std::fprintf(stderr, "[ninfer] TP2 rank %d: mailbox %s attached\n", rank, path.c_str());
    CUDA_CHECK(cudaHostRegister(host, bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
    std::fprintf(stderr, "[ninfer] TP2 rank %d: mailbox registered\n", rank);
    void* dev = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&dev, host, 0));
    CUDA_CHECK(cudaMalloc(&g_mailbox_steps, sizeof(std::uint64_t) * kMailboxMaxBlocks));
    CUDA_CHECK(cudaMemset(g_mailbox_steps, 0, sizeof(std::uint64_t) * kMailboxMaxBlocks));
    CUDA_CHECK(cudaMalloc(&g_gather_steps, sizeof(std::uint64_t) * kGatherMaxBlocks));
    CUDA_CHECK(cudaMemset(g_gather_steps, 0, sizeof(std::uint64_t) * kGatherMaxBlocks));
    CUDA_CHECK(cudaMalloc(&g_head_local, sizeof(__nv_bfloat16) * kGatherMaxElements));
    CUDA_CHECK(cudaMalloc(&g_head_act, sizeof(std::uint16_t) * 5120 * 32));
    g_mailbox_dev  = static_cast<MailboxShared*>(dev);
    g_mailbox_host = static_cast<MailboxShared*>(host);
}

// Times `iters` launches of `launch` on `stream`, in microseconds per launch.
template <class F>
double time_us(F&& launch, int iters, cudaStream_t stream) {
    cudaEvent_t e0 = nullptr;
    cudaEvent_t e1 = nullptr;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    for (int i = 0; i < 50; ++i) { launch(); }
    CUDA_CHECK(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; ++i) { launch(); }
    CUDA_CHECK(cudaEventRecord(e1, stream));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.F;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    return static_cast<double>(ms) * 1000.0 / iters;
}

// NVLink inbox: allocate this rank's inbox, publish its IPC handle next to the mailbox file, map the
// peer's, self-check, then race it against the host mailbox on a decode-sized message. Rank 0 makes
// the on/off decision and publishes it through the host mailbox so both ranks always agree. Any
// failure leaves the host mailbox in charge. `buffer` is a >= 128 KB device scratch, `stream` idle.
void init_nvlink_inbox(int rank, const std::string& path, std::uint64_t nonce, void* buffer,
                       cudaStream_t stream) {
    struct Handle { std::uint64_t nonce; cudaIpcMemHandle_t handle; };
    const int peer = 1 - rank;
    Inbox* mine    = nullptr;
    Inbox* theirs  = nullptr;
    auto give_up = [&](const std::string& why) {
        std::fprintf(stderr, "[ninfer] TP2 rank %d: NVLink inbox off (%s)\n", rank, why.c_str());
        if (theirs != nullptr) { (void)cudaIpcCloseMemHandle(theirs); }
        if (mine != nullptr) { (void)cudaFree(mine); }
        (void)cudaGetLastError();
    };
    if (cudaMalloc(&mine, sizeof(Inbox)) != cudaSuccess) { give_up("cudaMalloc failed"); return; }
    CUDA_CHECK(cudaMemset(mine, 0, sizeof(Inbox)));
    Handle h{nonce, {}};
    if (cudaIpcGetMemHandle(&h.handle, mine) != cudaSuccess) { give_up("cudaIpcGetMemHandle failed"); return; }
    const std::string mine_path = path + ".ipc" + std::to_string(rank);
    const std::string peer_path = path + ".ipc" + std::to_string(peer);
    {
        const std::string tmp = mine_path + ".tmp";
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&h), sizeof(h));
        if (!out || std::rename(tmp.c_str(), mine_path.c_str()) != 0) { give_up("cannot publish " + mine_path); return; }
    }
    Handle ph{};
    bool got = false;
    for (int attempt = 0; attempt < 1200 && !got; ++attempt) {
        std::ifstream in(peer_path, std::ios::binary);
        if (in && in.read(reinterpret_cast<char*>(&ph), sizeof(ph)) && ph.nonce == nonce) { got = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!got) { give_up("timed out waiting for the peer's IPC handle"); return; }
    const cudaError_t oe = cudaIpcOpenMemHandle(reinterpret_cast<void**>(&theirs), ph.handle,
                                                cudaIpcMemLazyEnablePeerAccess);
    if (oe != cudaSuccess) { theirs = nullptr; give_up(std::string("cudaIpcOpenMemHandle: ") + cudaGetErrorString(oe)); return; }
    std::uint64_t* steps = nullptr;
    std::uint64_t* gsteps = nullptr;
    CUDA_CHECK(cudaMalloc(&steps, sizeof(std::uint64_t) * kMailboxMaxBlocks));
    CUDA_CHECK(cudaMemset(steps, 0, sizeof(std::uint64_t) * kMailboxMaxBlocks));
    CUDA_CHECK(cudaMalloc(&gsteps, sizeof(std::uint64_t) * kGatherMaxBlocks));
    CUDA_CHECK(cudaMemset(gsteps, 0, sizeof(std::uint64_t) * kGatherMaxBlocks));

    // Self-check: rank r contributes r + 1 everywhere, the sum must be exactly 3 on both ranks.
    auto* x = static_cast<__nv_bfloat16*>(buffer);
    const int n = kMailboxMaxElements;
    std::vector<__nv_bfloat16> host(n, __float2bfloat16(static_cast<float>(rank + 1)));
    bool ok = true;
    for (int round = 0; round < 3 && ok; ++round) {
        CUDA_CHECK(cudaMemcpyAsync(x, host.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));
        launch_nvlink_allreduce(x, n, rank, mine, theirs, steps, stream);
        std::vector<__nv_bfloat16> out(n);
        CUDA_CHECK(cudaMemcpyAsync(out.data(), x, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < n; ++i) {
            if (__bfloat162float(out[i]) != 3.0f) { ok = false; break; }
        }
    }
    // Race on a decode-sized residual (4 tokens x 5120). Both ranks run the identical sequence.
    const int probe = 20480;
    const double nvl_us = time_us([&] { launch_nvlink_allreduce(x, probe, rank, mine, theirs, steps, stream); }, 500, stream);
    const double mb_us  = g_mailbox_dev != nullptr
        ? time_us([&] { launch_mailbox_allreduce(x, probe, rank, g_mailbox_dev, g_mailbox_steps, stream); }, 500, stream)
        : 1.0e9;
    // Rank 0 decides; rank 1 waits for the verdict in the shared host block (pad0[1]: 1 = on, 2 = off).
    volatile std::uint64_t* verdict = &g_mailbox_host->pad0[1];
    std::uint64_t decision = 0;
    if (rank == 0) {
        decision = (ok && nvl_us < mb_us && nvl_us < 100.0) ? 1 : 2;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *verdict = decision;
    } else {
        for (int attempt = 0; attempt < 6000 && *verdict == 0; ++attempt) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        decision = *verdict;
    }
    std::fprintf(stderr, "[ninfer] TP2 rank %d: NVLink inbox probe: nvlink %.1f us, host mailbox %.1f us per 40 KB all-reduce, self-check %s\n",
                 rank, nvl_us, mb_us, ok ? "ok" : "FAILED");
    if (decision != 1) {
        CUDA_CHECK(cudaFree(steps));
        CUDA_CHECK(cudaFree(gsteps));
        give_up(rank == 0 ? "not faster than the host mailbox, or self-check failed" : "rank 0 decided");
        return;
    }
    g_inbox_mine          = mine;
    g_inbox_peer          = theirs;
    g_nvlink_steps        = steps;
    g_nvlink_gather_steps = gsteps;
    std::fprintf(stderr, "[ninfer] TP2 rank %d: NVLink inbox on\n", rank);
}

void nccl_check(ncclResult_t result, const char* what) {
    if (result != ncclSuccess) {
        throw std::runtime_error(std::string("TP2 ") + what + ": " + ncclGetErrorString(result));
    }
}

ncclUniqueId exchange_id(int rank, const std::string& path) {
    ncclUniqueId id{};
    if (rank == 0) {
        nccl_check(ncclGetUniqueId(&id), "ncclGetUniqueId");
        const std::string tmp = path + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(&id), sizeof(id));
            if (!out) { throw std::runtime_error("TP2: cannot write " + tmp); }
        }
        if (std::rename(tmp.c_str(), path.c_str()) != 0) {
            throw std::runtime_error("TP2: cannot publish " + path);
        }
        return id;
    }
    for (int attempt = 0; attempt < 6000; ++attempt) {
        std::ifstream in(path, std::ios::binary);
        if (in && in.read(reinterpret_cast<char*>(&id), sizeof(id))) { return id; }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("TP2: timed out waiting for rank 0 id at " + path);
}

} // namespace

void init() {
    if (g_comm != nullptr) { return; }
    const char* rank_env = std::getenv("NINFER_TP_RANK");
    const char* file_env = std::getenv("NINFER_TP_ID_FILE");
    if (rank_env == nullptr || file_env == nullptr) {
        throw std::runtime_error("TP2 target requires NINFER_TP_RANK and NINFER_TP_ID_FILE");
    }
    const int rank = std::atoi(rank_env);
    if (rank != 0 && rank != 1) { throw std::runtime_error("TP2: NINFER_TP_RANK must be 0 or 1"); }
    const ncclUniqueId id = exchange_id(rank, file_env);
    ncclComm_t comm       = nullptr;
    nccl_check(ncclCommInitRank(&comm, 2, id, rank), "ncclCommInitRank");
    std::fprintf(stderr, "[ninfer] TP2 rank %d: NCCL communicator up\n", rank);

    // NCCL connects lazily on the first collective; do that (for every decode-sized message and a
    // prefill-sized one) before any graph capture.
    void* buffer        = nullptr;
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaMalloc(&buffer, std::size_t{32} << 20));
    CUDA_CHECK(cudaMemset(buffer, 0, std::size_t{32} << 20));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (std::size_t bytes : {std::size_t{10240}, std::size_t{40960}, std::size_t{163840},
                              std::size_t{1} << 20, std::size_t{21} << 20}) {
        nccl_check(ncclAllReduce(buffer, buffer, bytes / 2, ncclBfloat16, ncclSum, comm, stream),
                   "warm-up all-reduce");
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const char* mailbox_env = std::getenv("NINFER_TP_MAILBOX");
    if (mailbox_env == nullptr || mailbox_env[0] != '0') {
        // Must live on tmpfs: cudaHostRegister rejects file-backed mappings of a regular filesystem.
        const char* mailbox_path = std::getenv("NINFER_TP_MAILBOX_FILE");
        init_mailbox(rank, mailbox_path != nullptr ? mailbox_path : "/dev/shm/ninfer_tp2_mailbox",
                     id_nonce(id));
        // Self-check: rank r contributes r + 1 everywhere, the sum must be exactly 3 on both ranks.
        const int n = kMailboxMaxElements;
        std::vector<__nv_bfloat16> host(n, __float2bfloat16(static_cast<float>(rank + 1)));
        auto* x = static_cast<__nv_bfloat16*>(buffer);
        for (int round = 0; round < 3; ++round) {
            CUDA_CHECK(cudaMemcpyAsync(x, host.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));
            launch_mailbox_allreduce(x, n, rank, g_mailbox_dev, g_mailbox_steps, stream);
            std::vector<__nv_bfloat16> out(n);
            CUDA_CHECK(cudaMemcpyAsync(out.data(), x, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (int i = 0; i < n; ++i) {
                if (__bfloat162float(out[i]) != 3.0f) {
                    throw std::runtime_error("TP2: mailbox all-reduce self-check failed");
                }
            }
        }
        // NVLink inbox on top of the mailbox (auto: probe and keep it only when it wins).
        const char* nvlink_env = std::getenv("NINFER_TP_NVLINK");
        if (nvlink_env == nullptr || nvlink_env[0] != '0') {
            init_nvlink_inbox(rank, mailbox_path != nullptr ? mailbox_path : "/dev/shm/ninfer_tp2_mailbox",
                              id_nonce(id), buffer, stream);
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(buffer));
    g_comm = comm;
    g_rank = rank;
    std::fprintf(stderr, "[ninfer] TP2 rank %d/2 ready (NCCL %d, mailbox %s, nvlink %s)\n", rank, NCCL_VERSION_CODE,
                 g_mailbox_dev != nullptr ? "on" : "off", g_inbox_peer != nullptr ? "on" : "off");
    // Create/attach the per-unit lockstep block now so both ranks agree on its lifetime.
    (void)::ninfer::runtime::tp::Lockstep::instance();
}

int rank() { return g_rank; }

void allreduce(Tensor& residual, cudaStream_t stream) {
    if (g_comm == nullptr) { throw std::logic_error("TP2 collectives used before init()"); }
    if (residual.dtype != DType::BF16 || !residual.is_contiguous()) {
        throw std::invalid_argument("TP2 all-reduce requires a contiguous BF16 tensor");
    }
    if (g_inbox_peer != nullptr && residual.numel() <= kMailboxMaxElements) {
        launch_nvlink_allreduce(static_cast<__nv_bfloat16*>(residual.data),
                                static_cast<int>(residual.numel()), g_rank, g_inbox_mine,
                                g_inbox_peer, g_nvlink_steps, stream);
        return;
    }
    if (g_mailbox_dev != nullptr && residual.numel() <= kMailboxMaxElements) {
        launch_mailbox_allreduce(static_cast<__nv_bfloat16*>(residual.data),
                                 static_cast<int>(residual.numel()), g_rank, g_mailbox_dev,
                                 g_mailbox_steps, stream);
        return;
    }
    nccl_check(ncclAllReduce(residual.data, residual.data, static_cast<std::size_t>(residual.numel()),
                             ncclBfloat16, ncclSum, g_comm, stream),
               "ncclAllReduce");
}

bool head_linear(const Tensor& hidden, const Weight& head, Tensor& out, cudaStream_t stream) {
    static const bool enabled = [] {
        const char* v = std::getenv("NINFER_TP_SHARD_HEADS");
        return v == nullptr || v[0] != '0';
    }();
    if (!enabled || g_mailbox_dev == nullptr || head.n % 64 != 0) { return false; }
    const std::int32_t nh = head.n / 2;
    const std::int32_t t  = static_cast<std::int32_t>(hidden.ne[1]);
    if (static_cast<std::int64_t>(nh) * t > kGatherMaxElements || (static_cast<std::int64_t>(nh) * t) % 8 != 0 ||
        out.ne[0] != head.n || out.ne[1] != t || out.dtype != DType::BF16 || !out.is_contiguous()) {
        return false;
    }
    Weight half;
    if (!head_row_view(head, g_rank * nh, nh, half)) { return false; }
    Tensor local(g_head_local, DType::BF16, {nh, t});
    if (head.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        // The public FP8 entry validates the whole-tensor plane geometry, which a row slice cannot
        // satisfy; call the QPN launcher directly (same kernel the vocabulary route uses at T<=32).
        if (head.layout != QuantLayout::VoltaQpnPrepacked || t > ops::detail::kFp8VoltaQpnMaxTokens ||
            hidden.ne[0] != 5120 || !ops::detail::fp8_volta_qpn_supported(nh, head.k, t)) {
            return false;
        }
        ops::detail::fp8_stage_bf16_activation_sm70(hidden, g_head_act, stream);
        ops::detail::launch_fp8_volta_qpn_fp16(hidden, half, g_head_act, local, stream);
    } else {
        ops::linear(hidden, half, local, stream);
    }
    if (g_inbox_peer != nullptr) {
        launch_nvlink_gather_rows(g_head_local, nh, t, static_cast<__nv_bfloat16*>(out.data), g_rank,
                                  g_inbox_mine, g_inbox_peer, g_nvlink_gather_steps, stream);
        return true;
    }
    launch_mailbox_gather_rows(g_head_local, nh, t, static_cast<__nv_bfloat16*>(out.data), g_rank,
                               g_mailbox_dev, g_gather_steps, stream);
    return true;
}

void combine_partial(const Tensor& partial, Tensor& residual, cudaStream_t stream) {
    if (g_rank == 0) {
        ops::residual_add(partial, residual, stream);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(residual.data, partial.data, residual.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    allreduce(residual, stream);
}

} // namespace ninfer::targets::qwen3_6_27b_tp2::detail::tp2
