# Hardware Requirements for Running EXAONE Models Locally

This document summarizes the hardware requirements for running the LGAI EXAONE models (32B and 1.2B) locally, including both full precision and quantized (16, 8, 4 bit) versions. These recommendations are based on official documentation, model architecture, and best practices for large language model inference.

---

## Model Overview

| Model Name                | Parameters | Context Length | Model File Size (fp16/bf16) |
|--------------------------|------------|---------------|-----------------------------|
| EXAONE-4.0-32B / Deep-32B| 32B        | 128K / 32K    | ~60–65 GB                   |
| EXAONE-4.0-1.2B          | 1.2B       | 64K           | ~2.5 GB                     |

---

## Hardware Requirements Table

### EXAONE 32B Model

| Precision      | VRAM (GPU) Required | System RAM | Disk Space Needed | Notes                                    |
|----------------|--------------------|------------|-------------------|------------------------------------------|
| Full (fp16/bf16)| 48–80 GB           | 128 GB     | 60–65 GB          | A100 80GB/H100 80GB or multi-GPU         |
| 16-bit (fp16)  | 48–80 GB           | 128 GB     | 60–65 GB          | Same as full precision                   |
| 8-bit          | 24–32 GB            | 64–128 GB  | 32–35 GB           | RTX 6000 Ada, RTX 4090, A6000            |
| 4-bit          | 16–24 GB            | 64–128 GB  | 8–10 GB            | RTX 4090, A6000, consumer GPUs possible  |

### EXAONE 1.2B Model

| Precision      | VRAM (GPU) Required | System RAM | Disk Space Needed | Notes                                    |
|----------------|--------------------|------------|-------------------|------------------------------------------|
| Full (fp16/bf16)| 4–6 GB              | 16 GB      | 2.5 GB            | Most modern GPUs, even laptops           |
| 16-bit (fp16)  | 4–6 GB              | 16 GB      | 2.5 GB            | Same as full precision                   |
| 8-bit          | 2–3 GB               | 8–16 GB    | 1.3 GB            | Consumer GPUs, some integrated GPUs      |
| 4-bit          | 1–2 GB               | 8–16 GB    | 0.7 GB            | Can run on many laptops, edge devices    |

---

## Cloud Provider Cost Table for EXAONE 1.2B Model (All Precisions)

The EXAONE 1.2B model (full, 16, 8, 4 bit) can run comfortably on any modern cloud GPU (T4, V100, A100, etc.). The cost per hour is the same for all precisions, as the model is small enough to fit on even the cheapest GPU VMs. Below are typical on-demand prices as of July 2025:

| Provider         | GPU Type   | VRAM  | Price per Hour (USD) | Notes                                 |
|------------------|-----------|-------|----------------------|---------------------------------------|
| Thunder Compute  | T4         | 16GB  | $0.27                | Cheapest, reliable, US region         |
| Thunder Compute  | A100 40GB  | 40GB  | $0.66                | SOTA, overkill for 1.2B, US region    |
| AWS              | T4         | 16GB  | $0.53                | g4dn.xlarge, US region                |
| AWS              | V100       | 16GB  | $3.06                | p3.2xlarge, US region                 |
| Google Cloud     | T4         | 16GB  | $0.35                | us-central1, on-demand                |
| Google Cloud     | V100       | 16GB  | $2.55                | europe-west4-b, on-demand             |
| Lambda Labs      | T4         | 16GB  | $0.50                | On-demand, US region                  |
| Lambda Labs      | A100 40GB  | 40GB  | $1.29                | On-demand, US region                  |
| RunPod           | A40        | 48GB  | $0.44                | Community cloud, US region            |
| RunPod           | A100 80GB  | 80GB  | $1.19                | Community cloud, US region            |
| Vast.ai          | T4         | 16GB  | $0.15 (median)        | Marketplace, reliability varies       |
| Paperspace       | RTX A6000  | 48GB  | $1.89                | On-demand, US region                  |
| OVHcloud         | L40S       | 48GB  | $1.69                | On-demand, Europe                     |
| Scaleway         | L40S       | 48GB  | $1.40                | On-demand, Europe                     |

**Note:** All precisions (full, 16, 8, 4 bit) for the 1.2B model will run on any of these GPUs. The cost is determined by the GPU type, not the model precision. For the lowest cost, choose a T4 or similar GPU.

---

## Additional Notes

- **CPU-only inference** is possible for 1.2B and quantized 32B, but will be very slow for 32B.
- **Multi-GPU setups** can be used to shard the 32B model if a single GPU does not have enough VRAM.
- **Quantization** (8-bit, 4-bit) reduces VRAM and RAM requirements, but may slightly impact output quality.
- **Disk space** values above are for a single model file; if you keep multiple versions, sum their sizes.
- **NVIDIA GPUs** are recommended for best performance and compatibility (CUDA, TensorRT, etc.).
- **Software requirements:**
  - Linux (Ubuntu 20.04/22.04 recommended)
  - Python 3.9+
  - `transformers>=4.41.0`, `torch` (with CUDA), quantization toolkits as needed

---

## Example Hardware Setups

### For EXAONE 32B (Full Precision)
- **GPU:** NVIDIA A100 80GB or H100 80GB
- **RAM:** 128 GB
- **Disk:** 65 GB SSD

### For EXAONE 32B (4-bit Quantized)
- **GPU:** NVIDIA RTX 4090 (24GB) or A6000 (48GB)
- **RAM:** 64–128 GB
- **Disk:** 10 GB SSD

### For EXAONE 1.2B (Any Precision)
- **GPU:** Any modern GPU with 4–6 GB VRAM (even some laptops)
- **RAM:** 16 GB
- **Disk:** 2.5 GB SSD

---
