GPU Kernel Engineer | C/C++ | CUDA | Nsight Compute

I write and optimize GPU kernels in CUDA C++ for inference workloads: GEMM, fused attention, quantization, softmax, reductions. Every kernel is profiled with Nsight Compute and benchmarked against cuBLAS. My best GEMM currently hits 58.8% of cuBLAS on an RTX 5070 Ti (Blackwell SM120).

I contribute to open-source CUDA projects. Recent work includes fixing compilation and numerical precision bugs in an INT8 fused attention kernel (2 PRs merged, max error reduced from 1.69 to 1.37) and a narrowing-conversion fix in ThunderKittens (Stanford HazyResearch, PR #179).

Currently building an FP4 fused attention kernel targeting consumer Blackwell GPUs using inline PTX.

Projects

cuda-kernels: GEMM, reduction, prefix scan, softmax, Flash Attention. Each kernel built from scratch with full Nsight Compute profiling at every optimization step.

Articles

Building an FP4 Fused Attention Kernel for Consumer Blackwell GPUs (SM120)

Exploring PTX: A Close Look at Tile Optimization in CUDA

From Silicon to Thread Identity: How CUDA Threads Know Who They Are

Links

- [Blog](https://florianmattana.com)
- [LinkedIn](https://www.linkedin.com/in/florian-elio-mattana/)
- [Twitter](https://x.com/florian_mattana)
