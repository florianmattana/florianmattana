<div align="center">

# Florian Mattana

**GPU Kernel Engineer**

I write CUDA kernels at the PTX level for inference workloads on consumer Blackwell GPUs (SM120)
where no wrappers, no libraries, and no tooling exist yet.

`C/C++` `CUDA` `Inline PTX` `Nsight Compute` `Nsight Systems` `Tensor Cores` `FP4` `SM120`

[Blog](https://florianmattana.com) · [LinkedIn](https://www.linkedin.com/in/florian-elio-mattana/) · [Twitter](https://x.com/florian_mattana)

</div>

---

### Why SM120

Most GPU kernel work targets datacenter chips — B200, B100, H100.
I target **consumer Blackwell** (RTX 5070 Ti, SM120): different MMA instructions, single-scale `scale_vec::1X`, FP4 packed in 8-bit containers, and zero `cuda::ptx` wrapper support.
If it runs on SM120, I wrote the PTX by hand.

---

### 🔬 Current Project

<table>
<tr>
<td width="120" align="center">
<br>
<a href="https://github.com/florianmattana/fp4-fused-attention-sm120">
<img src="https://img.shields.io/badge/FP4_Fused_Attention-SM120-76b900?style=for-the-badge&logo=nvidia" alt="FP4 Fused Attention"/>
</a>
</td>
<td>

**Fused GEMM → softmax → GEMM** attention kernel using FP4 E2M1 quantization with UE8M0 block scaling.
Inline PTX `mma.sync` — scores matrix stays in registers between both GEMMs.

📄 [Full technical writeup →](https://florianmattana.com/fp4-fused-attention-kernel-sm120)

</td>
</tr>
</table>

---

### 🔧 Projects

<table>
<tr>
<td width="50%" valign="top">

**[CUDA-Kernels](https://github.com/florianmattana/CUDA-Kernels)**
GEMM · Reduction · Prefix Scan · Softmax · Flash Attention
Built from scratch, profiled with NCU.

`Best GEMM → 58.8% of cuBLAS on RTX 5070 Ti`

</td>
<td width="50%" valign="top">

**[GPU Profiling Guide](https://gist.github.com/florianmattana)**
20,000+ words on Nsight Systems & Nsight Compute.
Napkin math · Roofline · Bottleneck classification · Source tab deep dive.

</td>
</tr>
</table>

---

### 🤝 Open-Source

| | Project | What I Did |
|:-:|---|---|
| ✅ | **[model-kernels](https://github.com/ParagEkbote/model-kernels)** | 5 compilation + 2 precision fixes on INT8 fused attention — max error 1.69 → 1.37 · **2 PRs merged** |
| 🔀 | **[ThunderKittens](https://github.com/HazyResearch/ThunderKittens/pull/179)** · Stanford HazyResearch | Narrowing-conversion fix in base-type packing |
| 🔍 | **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** | SM120 benchmarks · Identified [#34988](https://github.com/flashinfer-ai/flashinfer/issues/34988) — fused FP4 kernel slower than unfused |

---

### 📝 Articles

> [**Building an FP4 Fused Attention Kernel for Consumer Blackwell GPUs**](https://florianmattana.com/fp4-fused-attention-kernel-sm120)
> From hardware instructions to working kernel — inline PTX, block scaling, register budget, MMA fragment mapping.

> [**Exploring PTX: A Close Look at Tile Optimization in CUDA**](https://florianmattana.com/exploring-ptx-tile-optimization-cuda)

> [**From Silicon to Thread Identity: How CUDA Threads Know Who They Are**](https://florianmattana.com/from-silicon-to-thread-identity)

---

<div align="center">
<sub>Hardware: RTX 5070 Ti · SM120 · 12 GB GDDR7 · 672 GB/s</sub>
</div>
