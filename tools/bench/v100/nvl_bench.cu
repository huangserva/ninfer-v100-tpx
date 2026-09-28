// Two-process micro-benchmark: TP2 small-message all-reduce / row all-gather
//   (a) production path: mapped pinned host mailbox in /dev/shm (mailbox_allreduce_kernel copied verbatim)
//   (b) proposed path: each rank pushes into the PEER's device-memory inbox over NVLink (CUDA IPC handle)
// Run as two processes, one GPU visible each, e.g.
//   CUDA_VISIBLE_DEVICES=0 ./nvl_bench 0 &  CUDA_VISIBLE_DEVICES=1 ./nvl_bench 1
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"rank? ERR %s: %s (line %d)\n",#x,cudaGetErrorString(e),__LINE__); exit(1);}}while(0)

constexpr int kSlice = 1024, kMaxBlocks = 64, kMaxElems = kSlice * kMaxBlocks;
constexpr int kGChunk = 8192, kGMaxBlocks = 128, kGMaxElems = kGChunk * kGMaxBlocks;

struct alignas(128) MailboxShared {           // production layout (host memory, shared by both ranks)
    uint64_t magic; uint64_t pad0[15];
    uint64_t flag[2][kMaxBlocks][16];
    __nv_bfloat16 data[2][2][kMaxElems];
    uint64_t gflag[2][kGMaxBlocks][16];
    __nv_bfloat16 gdata[2][2][kGMaxElems];
};
struct alignas(128) Inbox {                   // proposed layout: one per rank, in that rank's device memory
    uint64_t flag[kMaxBlocks][16];
    __nv_bfloat16 data[2][kMaxElems];
    uint64_t gflag[kGMaxBlocks][16];
    __nv_bfloat16 gdata[2][kGMaxElems];
};

__device__ __forceinline__ void add_rank_order(uint4& mine, const uint4& other, int rank) {
    auto* a = reinterpret_cast<__nv_bfloat162*>(&mine);
    const auto* b = reinterpret_cast<const __nv_bfloat162*>(&other);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 fa = __bfloat1622float2(a[j]); const float2 fb = __bfloat1622float2(b[j]);
        const float2 s = rank == 0 ? make_float2(fa.x + fb.x, fa.y + fb.y) : make_float2(fb.x + fa.x, fb.y + fa.y);
        a[j] = __float22bfloat162_rn(s);
    }
}

// ---- (a) production host mailbox kernel, verbatim ----
__global__ void mailbox_allreduce_kernel(__nv_bfloat16* x, int n, int rank, MailboxShared* mb, uint64_t* steps) {
    const int blk = blockIdx.x; const uint64_t step = steps[blk] + 1; const int parity = step & 1; const int peer = 1 - rank;
    const int base = blk * kSlice; const int end = min(n, base + kSlice);
    for (int i = base + threadIdx.x * 8; i < end; i += blockDim.x * 8)
        *reinterpret_cast<uint4*>(&mb->data[rank][parity][i]) = *reinterpret_cast<const uint4*>(&x[i]);
    __threadfence_system(); __syncthreads();
    if (threadIdx.x == 0) {
        *reinterpret_cast<volatile uint64_t*>(&mb->flag[rank][blk][0]) = step;
        const long long start = clock64();
        while (*reinterpret_cast<volatile uint64_t*>(&mb->flag[peer][blk][0]) < step) { if (clock64() - start > 83000000000LL) __trap(); }
    }
    __syncthreads(); __threadfence_system();
    for (int i = base + threadIdx.x * 8; i < end; i += blockDim.x * 8) {
        uint4 mine = *reinterpret_cast<const uint4*>(&x[i]);
        const uint4 other = __ldcv(reinterpret_cast<const uint4*>(&mb->data[peer][parity][i]));
        add_rank_order(mine, other, rank); *reinterpret_cast<uint4*>(&x[i]) = mine;
    }
    if (threadIdx.x == 0) steps[blk] = step;
}
__global__ void mailbox_gather_kernel(const __nv_bfloat16* local, int nh, int t, __nv_bfloat16* out, int rank, MailboxShared* mb, uint64_t* steps) {
    const int blk = blockIdx.x; const uint64_t step = steps[blk] + 1; const int parity = step & 1; const int peer = 1 - rank;
    const int total = nh * t; const int base = blk * kGChunk; const int end = min(total, base + kGChunk); const int n = 2 * nh;
    for (int e = base + threadIdx.x * 8; e < end; e += blockDim.x * 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(&local[e]);
        *reinterpret_cast<uint4*>(&mb->gdata[rank][parity][e]) = v;
        const __nv_bfloat16* pv = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j) { const int idx = e + j; const int col = idx / nh; const int row = idx - col * nh; out[(int64_t)col * n + rank * nh + row] = pv[j]; }
    }
    __threadfence_system(); __syncthreads();
    if (threadIdx.x == 0) {
        *reinterpret_cast<volatile uint64_t*>(&mb->gflag[rank][blk][0]) = step;
        const long long start = clock64();
        while (*reinterpret_cast<volatile uint64_t*>(&mb->gflag[peer][blk][0]) < step) { if (clock64() - start > 83000000000LL) __trap(); }
    }
    __syncthreads(); __threadfence_system();
    for (int e = base + threadIdx.x * 8; e < end; e += blockDim.x * 8) {
        const uint4 v = __ldcv(reinterpret_cast<const uint4*>(&mb->gdata[peer][parity][e]));
        const __nv_bfloat16* pv = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j) { const int idx = e + j; const int col = idx / nh; const int row = idx - col * nh; out[(int64_t)col * n + peer * nh + row] = pv[j]; }
    }
    if (threadIdx.x == 0) steps[blk] = step;
}

// ---- (b) NVLink push kernels: write my slice + flag into the PEER's inbox, wait on MY inbox ----
__global__ void nvl_allreduce_kernel(__nv_bfloat16* x, int n, int rank, Inbox* mine, Inbox* peer, uint64_t* steps) {
    const int blk = blockIdx.x; const uint64_t step = steps[blk] + 1; const int parity = step & 1;
    const int base = blk * kSlice; const int end = min(n, base + kSlice);
    for (int i = base + threadIdx.x * 8; i < end; i += blockDim.x * 8)
        *reinterpret_cast<uint4*>(&peer->data[parity][i]) = *reinterpret_cast<const uint4*>(&x[i]);
    __threadfence_system(); __syncthreads();
    if (threadIdx.x == 0) {
        *reinterpret_cast<volatile uint64_t*>(&peer->flag[blk][0]) = step;
        const long long start = clock64();
        while (*reinterpret_cast<volatile uint64_t*>(&mine->flag[blk][0]) < step) { if (clock64() - start > 83000000000LL) __trap(); }
    }
    __syncthreads(); __threadfence_system();
    for (int i = base + threadIdx.x * 8; i < end; i += blockDim.x * 8) {
        uint4 v = *reinterpret_cast<const uint4*>(&x[i]);
        const uint4 other = __ldcv(reinterpret_cast<const uint4*>(&mine->data[parity][i]));
        add_rank_order(v, other, rank); *reinterpret_cast<uint4*>(&x[i]) = v;
    }
    if (threadIdx.x == 0) steps[blk] = step;
}
__global__ void nvl_gather_kernel(const __nv_bfloat16* local, int nh, int t, __nv_bfloat16* out, int rank, Inbox* mine, Inbox* peer, uint64_t* steps) {
    const int blk = blockIdx.x; const uint64_t step = steps[blk] + 1; const int parity = step & 1; const int prank = 1 - rank;
    const int total = nh * t; const int base = blk * kGChunk; const int end = min(total, base + kGChunk); const int n = 2 * nh;
    for (int e = base + threadIdx.x * 8; e < end; e += blockDim.x * 8) {
        const uint4 v = *reinterpret_cast<const uint4*>(&local[e]);
        *reinterpret_cast<uint4*>(&peer->gdata[parity][e]) = v;
        const __nv_bfloat16* pv = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j) { const int idx = e + j; const int col = idx / nh; const int row = idx - col * nh; out[(int64_t)col * n + rank * nh + row] = pv[j]; }
    }
    __threadfence_system(); __syncthreads();
    if (threadIdx.x == 0) {
        *reinterpret_cast<volatile uint64_t*>(&peer->gflag[blk][0]) = step;
        const long long start = clock64();
        while (*reinterpret_cast<volatile uint64_t*>(&mine->gflag[blk][0]) < step) { if (clock64() - start > 83000000000LL) __trap(); }
    }
    __syncthreads(); __threadfence_system();
    for (int e = base + threadIdx.x * 8; e < end; e += blockDim.x * 8) {
        const uint4 v = __ldcv(reinterpret_cast<const uint4*>(&mine->gdata[parity][e]));
        const __nv_bfloat16* pv = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
        for (int j = 0; j < 8; ++j) { const int idx = e + j; const int col = idx / nh; const int row = idx - col * nh; out[(int64_t)col * n + prank * nh + row] = pv[j]; }
    }
    if (threadIdx.x == 0) steps[blk] = step;
}

static void wait_file(const std::string& p, size_t sz) { for (int i = 0; i < 3000; ++i) { struct stat st{}; if (stat(p.c_str(), &st) == 0 && (size_t)st.st_size == sz) return; std::this_thread::sleep_for(std::chrono::milliseconds(20)); } fprintf(stderr, "timeout waiting %s\n", p.c_str()); exit(1); }

int main(int argc, char** argv) {
    const int rank = atoi(argv[1]); const int peer = 1 - rank; const std::string dir = argc > 2 ? argv[2] : "/dev/shm/nvl_bench";
    mkdir(dir.c_str(), 0700);
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0)); printf("[rank %d] %s\n", rank, prop.name);
    // -- host mailbox (production style)
    const size_t mb_bytes = (sizeof(MailboxShared) + 4095) / 4096 * 4096; const std::string mbp = dir + "/mailbox";
    void* host = nullptr;
    if (rank == 0) { unlink(mbp.c_str()); int fd = open(mbp.c_str(), O_RDWR | O_CREAT, 0600); if (ftruncate(fd, mb_bytes)) exit(2); host = mmap(nullptr, mb_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); close(fd); memset(host, 0, mb_bytes); ((MailboxShared*)host)->magic = 7; }
    else { for (;;) { struct stat st{}; int fd = open(mbp.c_str(), O_RDWR); if (fd >= 0 && fstat(fd, &st) == 0 && (size_t)st.st_size == mb_bytes) { host = mmap(nullptr, mb_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); close(fd); if (((volatile MailboxShared*)host)->magic == 7) break; munmap(host, mb_bytes); } else if (fd >= 0) close(fd); std::this_thread::sleep_for(std::chrono::milliseconds(20)); } }
    CK(cudaHostRegister(host, mb_bytes, cudaHostRegisterMapped | cudaHostRegisterPortable));
    MailboxShared* mb_dev = nullptr; CK(cudaHostGetDevicePointer((void**)&mb_dev, host, 0));
    // -- device inbox + IPC exchange (proposed)
    Inbox* mine = nullptr; CK(cudaMalloc(&mine, sizeof(Inbox))); CK(cudaMemset(mine, 0, sizeof(Inbox)));
    cudaIpcMemHandle_t h{}; CK(cudaIpcGetMemHandle(&h, mine));
    { const std::string tmp = dir + "/h" + std::to_string(rank) + ".tmp"; FILE* f = fopen(tmp.c_str(), "wb"); fwrite(&h, sizeof(h), 1, f); fclose(f); rename(tmp.c_str(), (dir + "/h" + std::to_string(rank)).c_str()); }
    wait_file(dir + "/h" + std::to_string(peer), sizeof(h));
    cudaIpcMemHandle_t ph{}; { FILE* f = fopen((dir + "/h" + std::to_string(peer)).c_str(), "rb"); if (fread(&ph, sizeof(ph), 1, f) != 1) exit(3); fclose(f); }
    Inbox* peer_box = nullptr;
    cudaError_t oe = cudaIpcOpenMemHandle((void**)&peer_box, ph, cudaIpcMemLazyEnablePeerAccess);
    if (oe != cudaSuccess) { printf("[rank %d] cudaIpcOpenMemHandle FAILED: %s\n", rank, cudaGetErrorString(oe)); return 4; }
    printf("[rank %d] peer inbox mapped at %p\n", rank, (void*)peer_box);
    uint64_t *st_a, *st_b, *st_c, *st_d; for (uint64_t** p : {&st_a, &st_b, &st_c, &st_d}) { CK(cudaMalloc(p, 8 * kGMaxBlocks)); CK(cudaMemset(*p, 0, 8 * kGMaxBlocks)); }
    __nv_bfloat16 *x, *local, *out; CK(cudaMalloc(&x, kMaxElems * 2)); CK(cudaMalloc(&local, kGMaxElems * 2)); CK(cudaMalloc(&out, 2 * kGMaxElems * 2));
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    // correctness check for both all-reduce paths: x = rank+1 everywhere -> 3
    {
        std::vector<__nv_bfloat16> hx(kMaxElems, __float2bfloat16(rank + 1.0f)), ho(kMaxElems);
        for (int path = 0; path < 2; ++path) {
            CK(cudaMemcpyAsync(x, hx.data(), kMaxElems * 2, cudaMemcpyHostToDevice, s));
            if (path == 0) mailbox_allreduce_kernel<<<kMaxBlocks, 64, 0, s>>>(x, kMaxElems, rank, mb_dev, st_a);
            else nvl_allreduce_kernel<<<kMaxBlocks, 64, 0, s>>>(x, kMaxElems, rank, mine, peer_box, st_b);
            CK(cudaMemcpyAsync(ho.data(), x, kMaxElems * 2, cudaMemcpyDeviceToHost, s)); CK(cudaStreamSynchronize(s));
            int bad = 0; for (int i = 0; i < kMaxElems; ++i) if (__bfloat162float(ho[i]) != 3.0f) ++bad;
            printf("[rank %d] %s all-reduce self-check: %s\n", rank, path ? "nvlink" : "host-mailbox", bad ? "FAILED" : "ok");
        }
    }
    const int iters = 2000;
    auto bench = [&](const char* name, auto launch) {
        for (int i = 0; i < 100; ++i) launch(); CK(cudaStreamSynchronize(s));
        cudaEventRecord(e0, s); for (int i = 0; i < iters; ++i) launch(); cudaEventRecord(e1, s); CK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1); const float us_stream = ms * 1000 / iters;
        // CUDA-graph version: launch overhead removed, like production decode
        cudaGraph_t g; cudaGraphExec_t ge; CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < 200; ++i) launch();
        CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
        cudaEventRecord(e0, s); for (int i = 0; i < 10; ++i) CK(cudaGraphLaunch(ge, s)); cudaEventRecord(e1, s); CK(cudaEventSynchronize(e1));
        cudaEventElapsedTime(&ms, e0, e1); const float us_graph = ms * 1000 / 2000;
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
        if (rank == 0) printf("%-34s stream %7.2f us/op   graph %7.2f us/op\n", name, us_stream, us_graph);
    };
    for (int n : {5120, 20480, 30720, 65536}) {
        char nm[64]; const int blocks = (n + kSlice - 1) / kSlice;
        snprintf(nm, 64, "allreduce host-mailbox n=%d", n); bench(nm, [&] { mailbox_allreduce_kernel<<<blocks, 64, 0, s>>>(x, n, rank, mb_dev, st_a); });
        snprintf(nm, 64, "allreduce nvlink-push n=%d", n); bench(nm, [&] { nvl_allreduce_kernel<<<blocks, 64, 0, s>>>(x, n, rank, mine, peer_box, st_b); });
    }
    const int nh = 124160;
    for (int t : {1, 4, 6}) {
        char nm[64]; const int blocks = (nh * t + kGChunk - 1) / kGChunk;
        snprintf(nm, 64, "gather host-mailbox nh=%d t=%d", nh, t); bench(nm, [&] { mailbox_gather_kernel<<<blocks, 256, 0, s>>>(local, nh, t, out, rank, mb_dev, st_c); });
        snprintf(nm, 64, "gather nvlink-push nh=%d t=%d", nh, t); bench(nm, [&] { nvl_gather_kernel<<<blocks, 256, 0, s>>>(local, nh, t, out, rank, mine, peer_box, st_d); });
    }
    CK(cudaIpcCloseMemHandle(peer_box));
    printf("[rank %d] done\n", rank);
    return 0;
}
