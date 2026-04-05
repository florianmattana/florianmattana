<div align="center">
Florian Mattana
GPU Kernel Engineer
I write CUDA kernels at the PTX level for inference workloads on consumer Blackwell GPUs (SM120)
where no wrappers, no libraries, and no tooling exist yet.
C/C++ CUDA Inline PTX Nsight Compute Nsight Systems Tensor Cores FP4 SM120
Blog · LinkedIn · Twitter
</div>

Why SM120
Most GPU kernel work targets datacenter chips — B200, B100, H100.
I target consumer Blackwell (RTX 5070 Ti, SM120): different MMA instructions, single-scale scale_vec::1X, FP4 packed in 8-bit containers, and zero cuda::ptx wrapper support.
If it runs on SM120, I wrote the PTX by hand.

Background
Production GPU work across three companies — CPU→GPU migration at Geopost (T4), CUDA defect detection at Airbus (V100, 33× speedup, GPU utilization 9% → 89%), semiconductor sensor calibration at Melexis (A10G, cuBLAS, custom kernels across FP64–FP8 precision formats).

Current Project
<table>
<tr>
<td width="120" align="center">
<br>
<a href="https://github.com/florianmattana/fp4-fused-attention-sm120">
<img src="https://img.shields.io/badge/FP4_Fused_Attention-SM120-76b900?style=for-the-badge&logo=nvidia" alt="FP4 Fused Attention"/>
</a>
</td>
<td>
Fused GEMM → softmax → GEMM attention kernel using FP4 E2M1 quantization with UE8M0 block scaling.
Inline PTX mma.sync — scores matrix stays in registers between both GEMMs. No public fused FP4 attention kernel exists for SM120 consumer GPUs.
📄 Full technical writeup →
</td>
</tr>
</table>

Projects
<table>
<tr>
<td width="50%" valign="top">
CUDA-Kernels
From-scratch GPU kernel implementations with full NCU profiling breakdown for each one. GEMM, reduction, prefix scan, softmax, Flash Attention — every kernel designed, profiled, and iterated at the PTX level on SM120.
</td>
<td width="50%" valign="top">
Tensara Leaderboard
First place on the MXFP4 quantization problem — beat a Triton kernel with a hand-written CUDA solution on B200.
📄 Full writeup →
</td>
</tr>
<tr>
<td width="50%" valign="top">
NCU Kernel Audits
Profiling and diagnosing real-world GPU kernels from other engineers. First audit: warp-specialized persistent MXFP8 GEMM (2-CTA cluster, tcgen05.mma, SM100). Diagnosed latency-bound bottleneck, 93% No Eligible stall, false-positive divergence from warp specialization.
</td>
<td width="50%" valign="top">
GPU Profiling Guide
20,000+ words on Nsight Systems & Nsight Compute.
Napkin math · Roofline · Bottleneck classification · Source tab deep dive.
</td>
</tr>
</table>

Open-Source
ProjectWhat I Did✅model-kernelsINT8 fused attention kernel for diffusion transformers — 5 compilation fixes, 2 precision fixes (max error 1.69 → 1.37), global max scale fix, pre-commit cleanup · 2 merged, 3 open PRs🔀ThunderKittens · Stanford HazyResearchNarrowing-conversion fix in base-type packing

Articles

Building an FP4 Fused Attention Kernel for Consumer Blackwell GPUs
From hardware instructions to working kernel — inline PTX, block scaling, register budget, MMA fragment mapping.


Tensara MXFP4: How I Beat a Triton Kernel with Hand-Written CUDA
Full walkthrough of the competitive problem, approach, and profiling.


Exploring PTX: A Close Look at Tile Optimization in CUDA


From Silicon to Thread Identity: How CUDA Threads Know Who They Are


<div align="center">
<sub>Hardware: RTX 5070 Ti · SM120 · 12 GB GDDR7 · 672 GB/s</sub>
</div>
