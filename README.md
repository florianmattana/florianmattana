# Florian Mattana

**GPU Kernel Engineer | C/C++ | CUDA | Inline PTX | Nsight Compute**

I write CUDA kernels at the PTX level for inference workloads — GEMM, fused attention, quantization, softmax, reductions. Every kernel is profiled with Nsight Compute and benchmarked against cuBLAS.

Most open-source GPU libraries (FlashInfer, CUTLASS, cuda::ptx) target datacenter Blackwell (SM100). I work on **consumer Blackwell (SM120)** — different MMA instructions, different scaling constraints, almost no existing tooling. I write the PTX by hand because the wrappers don't exist yet.

---

## Current Project

**[FP4 Fused Attention for SM120](https://github.com/florianmattana/fp4-fused-attention-sm120)** — Fused GEMM-softmax-GEMM attention kernel for consumer Blackwell GPUs using FP4 E2M1 quantization with UE8M0 block scaling. Inline PTX `mma.sync` (not `tcgen05.mma` — SM100 only). Scores matrix stays in registers between the two GEMMs to avoid spilling.

Full technical writeup: [Building an FP4 Fused Attention Kernel for Consumer Blackwell GPUs](https://florianmattana.com/fp4-fused-attention-kernel-sm120)

---

## Open-Source Contributions

**[model-kernels](https://github.com/ParagEkbote/model-kernels)** — Fixed 5 compilation bugs + 2 precision bugs in INT8 fused attention. Max error reduced from 1.69 to 1.37. **2 PRs merged.**

**[ThunderKittens](https://github.com/HazyResearch/ThunderKittens/pull/179)** (Stanford HazyResearch) — Fixed narrowing-conversion bug in base-type packing.

**[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** — Benchmarking SM120 attention/GEMM/quantization kernels, studying JIT pipeline. 

---

## Projects

**[CUDA-Kernels](https://github.com/florianmattana/CUDA-Kernels)** — GEMM, reduction, prefix scan, softmax, Flash Attention built from scratch. Each kernel profiled with Nsight Compute. Best GEMM reaches 58.8% of cuBLAS on RTX 5070 Ti.

**[GPU Profiling Guide](https://gist.github.com/florianmattana)** — 20,000+ word guide covering Nsight Systems and Nsight Compute end-to-end: napkin math, roofline analysis, compute/memory/latency-bound classification, bottleneck quantification, Source tab deep dive.

---

## Articles

- [Building an FP4 Fused Attention Kernel for Consumer Blackwell GPUs](https://florianmattana.com/fp4-fused-attention-kernel-sm120)
- [Exploring PTX: A Close Look at Tile Optimization in CUDA](https://florianmattana.com/exploring-ptx-tile-optimization-cuda)
- [From Silicon to Thread Identity: How CUDA Threads Know Who They Are](https://florianmattana.com/from-silicon-to-thread-identity)

---

<p align="center">
  <a href="https://florianmattana.com">Blog</a> · 
  <a href="https://www.linkedin.com/in/florian-elio-mattana/">LinkedIn</a> · 
  <a href="https://x.com/florian_mattana">Twitter</a>
</p>
