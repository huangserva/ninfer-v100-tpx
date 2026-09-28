# Changes relative to geoffwatts/ninfer-v100 @ b37d0dd

Apache License 2.0, section 4(b): the files listed under "Modified" carry changes made in this
repository; the files under "Added" are new. Commit history is preserved (`git log b37d0dd..main`).

## Commits

- b45e718e perf(sm70): key-split 8-warp Bc=64 INT8 small-T decode attention
- cd13c919 perf(sm70): PRMT int8->fp16 dequant in small-T v2 attention
- 11339e9a perf(sm70): retune D256 flash prompt attention for Volta
- 2b0b9555 feat(sm70/tp2): register d256-h12-kv2 causal attention geometry
- ba219436 feat(sm70/tp2): Qwen3.8-27B two-GPU tensor parallel (NCCL in graph)
- 48d1bca9 perf(sm70/tp2): longer KV splits for 12/2 small-T attention
- 67e7b3f9 feat(tp2): per-unit lockstep between ranks + front proxy/supervisor
- 53f65504 perf(sm70): stage bf16 activations to fp16 for QPN residual/down projections
- 2b3f16cd perf(sm70/tp2): faster decode round (attention, mailbox all-reduce, sharded heads)
- 2c9e20c9 perf(sm70/tp2): fused NVFP4 residual epilogue, faster split reduce, MTP key window knob
- 42ed9995 tp2_proxy: per-deployment mailbox file next to the lockstep file

## Added files

- TP2_PLAN.md
- src/ops/softmax_attention/dense/causal_cache/small_t_i8_volta_v2.cuh
- src/runtime/engine/tp_lockstep.h
- src/targets/qwen3_6/impl/debug_dump.h
- src/targets/qwen3_6_27b_tp2/CMakeLists.txt
- src/targets/qwen3_6_27b_tp2/export/ninfer/targets/qwen3_6_27b_tp2/package.h
- src/targets/qwen3_6_27b_tp2/impl/config.h
- src/targets/qwen3_6_27b_tp2/impl/load/bindings.cpp
- src/targets/qwen3_6_27b_tp2/impl/load/bindings.h
- src/targets/qwen3_6_27b_tp2/impl/package.cpp
- src/targets/qwen3_6_27b_tp2/impl/tp2_comm.cpp
- src/targets/qwen3_6_27b_tp2/impl/tp2_comm.h
- src/targets/qwen3_6_27b_tp2/impl/tp2_mailbox.cu
- src/targets/qwen3_6_27b_tp2/impl/tp2_mailbox.h
- src/targets/qwen3_6_27b_tp2/impl/variant.cpp
- src/targets/qwen3_6_27b_tp2/impl/variant.h
- tools/tp2/__init__.py
- tools/tp2/shard_qwen38_27b.py
- tools/tp2/tp2_proxy.py

## Modified files

- bench/ops/causal_softmax_attention_bench.cu
- bench/ops/linear_bench.cu
- include/ninfer/ops/softmax_attention.h
- src/CMakeLists.txt
- src/ops/attn_input_proj/fp8/fp8_attn_input_cutlass_sm70.cu
- src/ops/attn_input_proj/fp8/fp8_attn_input_output.cuh
- src/ops/attn_input_proj/fp8/fp8_attn_input_plan.cpp
- src/ops/attn_input_proj/fp8/fp8_attn_input_volta_qpn.cu
- src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu
- src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp
- src/ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.cpp
- src/ops/gdn_input_proj/fp8/fp8_gdn_input_cutlass_sm70.cu
- src/ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh
- src/ops/gdn_input_proj/fp8/fp8_gdn_input_plan.cpp
- src/ops/gdn_input_proj/fp8/fp8_gdn_input_volta_qpn.cu
- src/ops/gdn_input_proj/gdn_projected_conv.cu
- src/ops/kernel/mtp_pack.cuh
- src/ops/launcher/causal_conv1d.cu
- src/ops/launcher/causal_conv1d.h
- src/ops/launcher/gqa_attention_volta_flash.cu
- src/ops/launcher/mtp_pack.cu
- src/ops/linear/fp8/fp8_config.h
- src/ops/linear/nvfp4/nvfp4_config.h
- src/ops/linear/nvfp4/nvfp4_dispatch.cpp
- src/ops/linear/nvfp4/nvfp4_launch.h
- src/ops/linear/nvfp4/nvfp4_output.cuh
- src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cu
- src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh
- src/ops/linear/q4/q4_dispatch.cpp
- src/ops/linear/w8/w8_dispatch.cpp
- src/ops/linear_add/fp8/fp8_linear_add_decode.cu
- src/ops/linear_add/fp8/fp8_linear_add_plan.cpp
- src/ops/linear_add/fp8/fp8_linear_add_plan.h
- src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp
- src/ops/linear_attention/gated_delta_net/recurrent.cu
- src/ops/linear_attention/gated_delta_net/recurrent.cuh
- src/ops/linear_attention/gated_delta_net/replay.cpp
- src/ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp
- src/ops/linear_swiglu/fp8/fp8_linear_swiglu_qpn_split.cu
- src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
- src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_qpn_split.cu
- src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp
- src/ops/softmax_attention/dense/causal_cache/geometry.cuh
- src/ops/softmax_attention/dense/causal_cache/prompt.cu
- src/ops/softmax_attention/dense/causal_cache/small_t.cu
- src/ops/softmax_attention/dense/causal_cache/small_t.cuh
- src/ops/wrapper/attn_input_proj.cpp
- src/ops/wrapper/causal_conv1d_silu.cpp
- src/ops/wrapper/gdn_gating_proj.cpp
- src/ops/wrapper/gdn_input_proj.cpp
- src/ops/wrapper/linear_add.cpp
- src/ops/wrapper/linear_swiglu.cpp
- src/ops/wrapper/mtp_pack.cpp
- src/runtime/engine/engine.cpp
- src/runtime/engine/engine_core.h
- src/targets/qwen3_6/impl/runtime/text_context_impl.h
- src/targets/qwen3_6_27b/impl/variant.cpp
- src/targets/registry.cpp
- src/targets/registry.h
- tests/ops/linear/test_nvfp4_a16.cpp
- tests/ops/linear/test_w8_a16.cpp
- tests/ops/linear_add/test_nvfp4.cpp
- tests/ops/softmax_attention/causal_cache.cpp
- third_party/llama_cpp_fattn/fattn-mma-f16.cuh

## Tooling and docs added after 42ed9995

- tools/tp2/tp2_proxy.py: `--nccl-p2p {off,auto}` (default off keeps the previous behaviour, NCCL_P2P_DISABLE=1 for both ranks).
- tools/bench/v100/ctx_prompt.py: the synthetic code / zh-doc prompt generator behind every speed number.
- tools/bench/v100/decode_bench.py, matrix.sh: decode / first-token benchmark and the README table runner.
- tools/bench/v100/tp2_scenarios.py, tp2_fault.py: functional scenarios and fault injection for the TP2 service.
- deploy/Dockerfile, deploy/collect_libs.sh: runtime packaging of host-built binaries (replaces deploy/Dockerfile.tp2, whose base image was never published). The upstream root Dockerfile (CUDA 13 base, cannot target sm_70) is removed.
- docs/reproduce.md: from-scratch install and test checklist with expected outputs.
