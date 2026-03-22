Florian Mattana
GPU Kernel Engineer | C/C++ | CUDA | Nsight Compute

I write and optimize CUDA kernels for inference: GEMM, fused attention, quantization, softmax, reductions. Every kernel is profiled with Nsight Compute and benchmarked against cuBLAS.

Currently building an FP4 fused attention kernel for consumer Blackwell GPUs (SM120) using inline PTX — documented from hardware instructions to working code.

Best GEMM so far: 58.8% of cuBLAS on RTX 5070 Ti.

Current Focus
FP4 Fused Attention for SM120 — A fused FP4 attention kernel targeting consumer Blackwell GPUs that existing implementations (SageAttention3, FlashAttention-4) don't support. Warp-level mma.sync with block scaling, online softmax, FP4 E2M1 quantization. Full technical writeup here.

Open-Source Contributions
Project	Contribution	Status
model-kernels	Fixed 5 compilation + 2 precision bugs in INT8 fused attention. Max error 1.69 → 1.37	2 PRs merged
ThunderKittens	Fixed narrowing-conversion bug in base-type packing	PR open
FlashInfer	Benchmarking SM120 attention/GEMM/quantization kernels, studying JIT pipeline	Exploring
Projects
Repo	Description
fp4-fused-attention-sm120	FP4 fused attention for consumer Blackwell. Inline PTX, block-scaled MMA, online softmax.
cuda-kernels	GEMM, reduction, prefix scan, softmax, Flash Attention. Built from scratch with Nsight Compute profiling.
Articles
Building an FP4 Fused Attention Kernel for Consumer Blackwell GPUs
Exploring PTX: A Close Look at Tile Optimization in CUDA
From Silicon to Thread Identity: How CUDA Threads Know Who They Are
