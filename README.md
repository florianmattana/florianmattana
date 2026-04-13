# Florian Mattana

**GPU Kernel Engineer**

I write CUDA kernels at the PTX level for inference workloads on consumer Blackwell GPUs (SM120). I document what is not documented: MMA fragment layouts, FP4 container formats, scale factor distribution across lanes, and every SASS-level decision that affects performance.

`C/C++` `CUDA` `Inline PTX` `Nsight Compute` `Nsight Systems` `Tensor Cores` `FP4` `SM120` `Attention`

[**Blog**](https://florianmattana.com) · [**LinkedIn**](https://linkedin.com/in/florianmattana) · [**Twitter**](https://x.com/florian_mattana)

---

## Background

Production GPU work across three companies: CPU→GPU pipeline migration at Geopost (T4), CUDA defect detection at Airbus (V100, 33x speedup, GPU utilization 9% → 89%), semiconductor sensor calibration at Melexis (A10G, cuBLAS, custom kernels across FP64 to FP8 precision formats).

---

## Current Project

### [FP4 Fused Attention Kernel for Consumer Blackwell (SM120)](https://github.com/florianmattana/fp4-fused-attention-sm120)

Fused GEMM → softmax → GEMM attention kernel using FP4 E2M1 quantization with UE8M0 block scaling. Written entirely in inline PTX using `mma.sync.aligned.m16n8k32`. The score matrix stays in registers between both GEMMs.

The kernel is functionally complete: online softmax, K tile loop, multi-head, arbitrary HEAD_DIM. Correctness validated at cosine 1.0000 on all configurations. The MMA fragment layout, container format, and scale distribution for SM120 were reverse-engineered empirically since none of it is documented in the PTX ISA.

This is a pedagogical project. The goal is not to compete with SageAttention3 on throughput but to make every step of the FP4 attention pipeline visible at the instruction level. The full technical writeup (21 sections, 9500+ words) documents every bug, every wrong assumption, and every hardware surprise.

[**Full technical writeup →**](https://florianmattana.com/posts/fp4-fused-attention-kernel-sm120/)

---

## Projects

### [CUDA-Kernels](https://github.com/florianmattana/CUDA-Kernels)

From-scratch GPU kernel implementations with full NCU profiling breakdown for each one. GEMM, reduction, prefix scan, softmax, Flash Attention. Every kernel designed, profiled, and iterated at the PTX level on SM120.

### Tensara Leaderboard

First place on the MXFP4 quantization problem. Beat a Triton kernel with a hand-written CUDA solution on B200.

[**Full writeup →**](https://florianmattana.com/posts/mxfp4_article/)

### NCU Kernel Audits

Profiling and diagnosing real-world GPU kernels from other engineers. First audit: warp-specialized persistent MXFP8 GEMM (2-CTA cluster, tcgen05.mma, SM100). Diagnosed latency-bound bottleneck, 93% No Eligible stall, false-positive divergence from warp specialization.

### [GPU Profiling Guide](https://gist.github.com/florianmattana)

20,000+ words on Nsight Systems and Nsight Compute. Napkin math, roofline, bottleneck classification, source tab deep dive.

---

## Open-Source Contributions

| Project | What I Did |
|---|---|
| [**model-kernels**](https://github.com/ParagEkbote/model-kernels) | INT8 fused attention kernel for diffusion transformers. 5 compilation fixes, 2 precision fixes (max error 1.69 → 1.37), global max scale fix. Found and fixed a segfault caused by a schema contract violation in timestep_scales validation. Benchmarked on Sana 1600M (1024px), diagnosed performance gap (warp-level parallelism in softmax loop) and accuracy issues on head_dim=32. |
| [**ThunderKittens**](https://github.com/HazyResearch/ThunderKittens) · Stanford HazyResearch | Narrowing-conversion fix in base-type packing (PR #179) |
| [**CCCL**](https://github.com/NVIDIA/cccl/issues/8146) · NVIDIA | Issue #8146: requesting cuda::ptx wrappers for warp-level mma.sync on SM120 |

---

## Articles

[**Building an FP4 Fused Attention Kernel for Consumer Blackwell GPUs**](https://florianmattana.com/posts/fp4-fused-attention-kernel-sm120/)
From hardware instructions to working kernel: inline PTX, block scaling, register budget, MMA fragment mapping, NCU profiling.

[**Tensara MXFP4: How I Beat a Triton Kernel with Hand-Written CUDA**](https://florianmattana.com/posts/mxfp4_article/)
Full walkthrough of the competitive problem, approach, and profiling.

[**Exploring PTX: A Close Look at Tile Optimization in CUDA**](https://florianmattana.com/posts/exploring-ptx/)

[**From Silicon to Thread Identity: How CUDA Threads Know Who They Are**](https://florianmattana.com/posts/from-silicon-to-thread-identity/)

