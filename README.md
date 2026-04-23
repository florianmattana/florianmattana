# Florian Mattana
**GPU Performance Engineer**

CUDA kernel with a focus on the full compilation pipeline — CUDA to PTX to SASS — and low-level GPU design.

[Blog](https://florianmattana.com) · [LinkedIn](https://www.linkedin.com/in/florian-elio-mattana/) · [X](https://x.com/florian_mattana)

---

## SASS King

[github.com/florianmattana/sass-king](https://github.com/florianmattana/sass-king)

SASS is the machine code NVIDIA GPUs actually execute. CUDA compiles to PTX (documented). PTX assembles to SASS (architecture-specific and undocumented). Everything that matters for performance happens at the SASS level: instruction scheduling, register allocation, scoreboard management, fusion decisions, unrolling strategies. None of it is visible at the source.

The last systematic public work on SASS was Jia et al. (Citadel) in 2018–2019, covering Volta and Turing. Nothing comparable exists for Ampere, Hopper, or Blackwell. For SM120: zero. SASS King is an attempt to close that gap and lower the barrier for kernel developers who need to read SASS.

**Methodology.** Controlled variation: start from the simplest kernel, change one thing in the source, recompile, diff the SASS, document the observation. Paired with NCU profiling so every instruction-level claim is grounded in a measured outcome. Every statement is tagged `[OBS]` observed, `[INF]` inferred, or `[HYP]` hypothesis.

**Target architectures:** SM80 (A100), SM89 (RTX 4090), SM90a (H100), SM100a (B200), SM120 (RTX 5070 Ti / 5090). Work starts on SM120 because that is where I have direct hardware access. Community dumps and contributions cover the rest.

**Public status.** Phase 1 in progress: 6 kernel studies on SM120 (baseline vector add, FMA fusion, scoreboard grouping, unroll cascade, fixed-trip loops, shared memory scalar). Next in the roadmap: vectorized loads, warp reductions, division slowpath, register spill, first tensor core (`QMMA`). Later phases: classical algorithms (SGEMM, reductions, softmax, LayerNorm), annotated audits of real libraries (flash-attn, CUTLASS, FlashInfer, transformer_engine, llama.cpp), and a per-instruction reference across architectures.

**Contributions welcome.** SASS dumps from SM80 / SM89 / SM90a / SM100a move the project forward directly. The [CONTRIBUTING guide](https://github.com/florianmattana/sass-king/blob/main/CONTRIBUTING.md) lists the kernels to compile and the exact flags.

[Part 1 — Reading NVIDIA SASS from First Principles →](https://florianmattana.com/posts/sass_king/)

---

## FP4 Fused Attention for SM120

[github.com/florianmattana/fp4-fused-attention-sm120](https://github.com/florianmattana/fp4-fused-attention-sm120)

Fused GEMM → softmax → GEMM attention kernel for consumer Blackwell (SM120), written in inline PTX with warp-level `mma.sync`, FP4 E2M1 quantization, and UE8M0 block scaling. Work in progress, with validated MMA tests.

**Why it exists.** Existing FP4 attention implementations (FlashAttention-4, SageAttention3) target SM100 datacenter hardware. SM120 consumer GPUs are left on FP16 / FP8 fallbacks even though the Tensor Cores can run FP4.

**Reverse-engineered on SM120** (none of this is in the PTX ISA documentation):

* FP4 E2M1 via `kind::mxf8f6f4` stores each 4-bit value in an 8-bit container (bits 5–2, with padding). Throughput is half of SM100's `kind::mxf4nvf4`.
* `scale_vec::2X` is not available on SM120. Block scaling is limited to `scale_vec::1X` (one UE8M0 scale per 32 elements).
* The working MMA is `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0`.

[Full technical writeup →](https://florianmattana.com/posts/fp4-fused-attention-kernel-sm120/)

---

## Tensara MXFP4 · First Place

First place on the Tensara MXFP4 quantization problem. Hand-written CUDA kernel on B200, 72.48 μs final across all test sizes, ahead of a Triton baseline at 75.12 μs. The writeup walks through the spec work (OCP standard, floor vs ceil scale, round-to-nearest-even with `>=` vs `>` per midpoint), the naive kernel, and the warp-level rewrite that made memory access coalesced.

[Writeup →](https://florianmattana.com/posts/mxfp4_article/)

---

## NCU Kernel Audits

Profiling and diagnosing real-world GPU kernels from other engineers. First audit: warp-specialized persistent MXFP8 GEMM (2-CTA cluster, `tcgen05.mma`, SM100). Diagnosed latency-bound bottleneck, 93% No Eligible stall, false-positive divergence from warp specialization.

---

## Open-Source Contributions

| Project | Contribution | Status |
| --- | --- | --- |
| [model-kernels](https://github.com/ParagEkbote/model-kernels) | INT8 fused attention for diffusion transformers. 5 compilation + 2 precision fixes (max error 1.69 → 1.37), global max scale fix, segfault fix on a schema contract violation in `timestep_scales`. Benchmarked on Sana 1600M. | 2 PRs merged |
| [ThunderKittens](https://github.com/HazyResearch/ThunderKittens/pull/179) · Stanford HazyResearch | Narrowing-conversion fix in base-type packing | PR #179 |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | Benchmarking SM120 attention / GEMM / quantization kernels, studying the JIT pipeline | Exploring |

---

## Articles

* [**SASS King, Part 1: Reading NVIDIA SASS from First Principles**](https://florianmattana.com/posts/sass_king/) · April 2026 · 19 min
* [**I Wrote an MXFP4 Quantization Kernel and Ranked #1 on Tensara**](https://florianmattana.com/posts/mxfp4_article/) · April 2026 · 27 min
* [**Building an FP4 Fused Attention Kernel on Consumer Blackwell (SM120)**](https://florianmattana.com/posts/fp4-fused-attention-kernel-sm120/) · March 2026 · 39 min
* [**From Silicon to Thread Identity: How CUDA Threads Know Who They Are**](https://florianmattana.com/posts/from-silicon-to-thread-identity/) · February 2026 · 8 min
* [**Exploring PTX: A Close Look at Tile Optimization in CUDA**](https://florianmattana.com/posts/exploring-ptx-tile-optimization/) · January 2026 · 10 min

---

## Background

Previously: GPU software engineer at Melexis (A10G) on calibration pipelines across FP64 → FP8. Before that, Airbus (V100) on CUDA defect detection, and DPD Group (T4) migrating a parcel matching pipeline from CPU to GPU.
