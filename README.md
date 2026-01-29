# 🐙 Octopus-Inspired GPU Load Balancing

*Bio-inspired adaptive block assignment for image processing*

---

## TL;DR

I achieved **7-11x speedup over standard GPU practices** (grid-stride) on image-aware operations by applying insights from octopus neuroscience: adaptive pre-computed workload distribution.

| Test | Grid-Stride | Hybrid (Ours) | Speedup |
|------|-------------|---------------|---------|
| Uniform images | 49.39 ms | **6.53 ms** | **7.6x** |
| Mixed + 2K | 54.05 ms | **5.70 ms** | **9.5x** |
| Mixed + 4K | 56.74 ms | **6.04 ms** | **9.4x** |
| Mixed + 8K | 83.32 ms | **7.35 ms** | **11.3x** |

**4/4 tests: Hybrid wins** ✅

---

## The Journey: From 252x to Honest 11x

### What I Originally Claimed
> "252x speedup on GPU parallel processing!"

### What I Discovered
That 252x was comparing against a **weak baseline** (1 thread per image). When I compared against **proper GPU practices** (grid-stride), I initially **lost by 2.5x**.

### What I Built
A **hybrid approach** that combines:
- Block-per-image for small images (locality advantage)
- Adaptive subdivision for large images (load balance)

Result: **7-11x speedup over fair baseline** — an honest, reproducible improvement.

---

## The Octopus Insight

An octopus has ~500 million neurons distributed across 8 arms. Each arm operates semi-independently, yet they coordinate perfectly. When reaching for prey, all arms arrive simultaneously.

**How?** The octopus pre-computes force distribution so no arm waits for another.

**GPU translation:** Pre-compute work distribution so no thread waits.

---

## The Problem: Load Imbalance

Traditional approaches:

**Naive (1 thread/image):**
```
Thread 0: ████████████████ (16M pixels) → slowest, everyone waits
Thread 1: ████ (2M pixels)              → idle 87.5% of time
```

**Grid-Stride (standard GPU):**
```
Each thread processes interleaved pixels across ALL images
✅ Good memory coalescing
❌ Must search to find image boundaries (for image-aware ops)
❌ No locality within images
```

---

## The Solution: Hybrid Adaptive Assignment

```python
# Adaptive block assignment
threshold = 65536  # pixels per block

for each image:
    if image.size <= threshold:
        assign 1 block (locality)
    else:
        assign ceil(size / threshold) blocks (load balance)
```

**Result:**
- Small images: 1 block each → preserves locality
- Large images: subdivided → balanced workload
- All blocks: ~equal work → no waiting

---

## Benchmark Results

### Hybrid vs Fair Baseline (Grid-Stride)

Tested on **3x3 Gaussian blur** (image-aware operation requiring neighbor access):

| Test | Imbalance | Grid-Stride | Block/Image | **Hybrid** | Winner |
|------|-----------|-------------|-------------|------------|--------|
| Flickr Pure | 1.4x | 49.39 ms | 6.95 ms | **6.53 ms** | ✅ Hybrid |
| Flickr + 2K | 21.5x | 54.05 ms | 33.07 ms | **5.70 ms** | ✅ Hybrid |
| Flickr + 4K | 42.5x | 56.74 ms | 61.38 ms | **6.04 ms** | ✅ Hybrid |
| Flickr + 8K | 169x | 83.32 ms | 240.14 ms | **7.35 ms** | ✅ Hybrid |

**Key insight:** 
- Block-per-image wins on uniform workloads (7.6x)
- Block-per-image **loses** on imbalanced workloads (0.35x)
- Hybrid **wins all scenarios** (7-11x)

---

### Why Hybrid Wins

| Aspect | Grid-Stride | Block-per-Image | Hybrid |
|--------|-------------|-----------------|--------|
| Load balance (uniform) | ✅ | ✅ | ✅ |
| Load balance (imbalanced) | ✅ | ❌ | ✅ |
| Memory locality | ❌ | ✅ | ✅ |
| No search overhead | ❌ | ✅ | ✅ |
| Image boundary awareness | ❌ | ✅ | ✅ |

---

## When Does This Work?

### ✅ Good fit (use Hybrid):
- **Image-aware operations** (blur, edge detection, convolution)
- **Variable-size batches** (thumbnails + full-res)
- **Operations needing image boundaries** (per-image statistics, segmentation)

### ⚠️ Use Grid-Stride instead:
- **Per-pixel independent operations** (normalize, threshold, LUT)
- **No image structure needed**

### The Rule:
> For **image-aware** operations with **size variance**, Hybrid beats Grid-Stride by **7-11x**.

---

## Implementation

### Core Algorithm (~50 lines)

```python
def compute_hybrid_assignment(sizes, threshold=65536):
    """Adaptive block assignment."""
    block_to_image = []
    block_start = []
    block_end = []
    
    for img_id, size in enumerate(sizes):
        if size <= threshold:
            # Small image: 1 block
            block_to_image.append(img_id)
            block_start.append(0)
            block_end.append(size)
        else:
            # Large image: subdivide
            num_blocks = ceil(size / threshold)
            for b in range(num_blocks):
                block_to_image.append(img_id)
                block_start.append(b * threshold)
                block_end.append(min((b+1) * threshold, size))
    
    return block_to_image, block_start, block_end
```

### GPU Kernel

```python
@cuda.jit
def hybrid_kernel(images_flat, offsets, widths, heights,
                  block_to_image, block_start, block_end, output):
    block_id = cuda.blockIdx.x
    
    # O(1) lookup - no search!
    img_id = block_to_image[block_id]
    local_start = block_start[block_id]
    local_end = block_end[block_id]
    
    # Image info
    offset = offsets[img_id]
    w = widths[img_id]
    
    # Threads cooperate within block's range
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    
    for local_idx in range(local_start + tid, local_end, stride):
        # Process with full image context available
        y = local_idx // w
        x = local_idx % w
        # ... blur/edge/convolution logic
```

---

## Honest Comparison

| Claim | Baseline | Speedup | Status |
|-------|----------|---------|--------|
| vs Naive (1 thread/image) | Weak | 201x | ⚠️ Misleading |
| **vs Grid-Stride** | **Fair** | **7-11x** | ✅ **Honest** |

The 201x number is technically correct but compares against a strawman. The **7-11x vs grid-stride** is the academically defensible claim.

---

## Files

| File | Description |
|------|-------------|
| `hybrid_benchmark.py` | **Main benchmark** — Hybrid vs Grid-Stride vs Block-per-Image |
| `locality_benchmark.py` | Image-aware operations (blur) comparison |
| `fair_benchmark.py` | Per-pixel operations — shows where Grid-Stride wins |
| `web_image_benchmark.py` | Legacy benchmark (vs weak baseline) |
| `medical_benchmark.py` | Medical imaging tests |

## Quick Start

```bash
git clone https://github.com/matthewlam721/octopus-parallel.git
cd octopus-parallel

pip install numba numpy scipy pillow

# Download Flickr8k from Kaggle
# Place in ./Images/

# Run main benchmark
python hybrid_benchmark.py
```

---

## Future Work

### Optimizations Identified
- [ ] **Cost-based threshold T** — Use estimated ops instead of pixels
- [ ] **2D tiling** — Better for convolution with shared memory
- [ ] **Tune T vs threads/block** — Optimize for occupancy
- [ ] **Persistent kernels** — Reduce launch overhead

### Validation Roadmap
- [ ] Edge deployment (NVIDIA Jetson)
- [ ] Real algorithms (U-Net, segmentation)
- [ ] Video processing datasets
- [ ] Framework integration (PyTorch, JAX)

### Publication
- [ ] Conference paper (targeting MLSys, IPDPS workshop, or similar)

---

## What I Learned

1. **Honest benchmarks matter.** My initial 252x was vs weak baseline. Real contribution is 7-11x vs fair baseline.

2. **Simple isn't always optimal.** Block-per-image is simple but fails on imbalanced workloads. Hybrid adds complexity but wins all scenarios.

3. **Know when your approach wins.** Grid-stride beats us on per-pixel ops. We beat grid-stride on image-aware ops.

4. **Bio-inspiration works.** The octopus insight led to a real solution, not just a cute analogy.

---

## Conclusion

The octopus doesn't wait for its slowest arm. Neither should your GPU blocks.

For image-aware operations with variable-sized workloads, our hybrid adaptive assignment achieves **7-11x speedup** over standard grid-stride, with simple implementation and zero runtime overhead.

---

*Author: Matthew, UIUC MCS*  
*Contact: matthewlam721@gmail.com*  
*Repo: [github.com/matthewlam721/octopus-parallel](https://github.com/matthewlam721/octopus-parallel)*

---

## Appendix: Benchmark Results

### Hybrid Benchmark (Main Result)

```
======================================================================
HYBRID BENCHMARK SUMMARY
======================================================================

Test                  Grid-Stride  Block/Image       Hybrid          Winner
---------------------------------------------------------------------------
Flickr Pure                 49.39         6.95         6.53          hybrid
Flickr + 2K                 54.05        33.07         5.70          hybrid
Flickr + 4K                 56.74        61.38         6.04          hybrid
Flickr + 8K                 83.32       240.14         7.35          hybrid

======================================================================
Win count:
  Hybrid: 4/4
  Block-per-Image: 0/4
  Grid-Stride: 0/4

🐙 HYBRID WINS ALL TESTS!
```

### Locality Benchmark (Block-per-Image Analysis)

```
======================================================================
LOCALITY BENCHMARK SUMMARY
======================================================================

Test                      Naive  Grid-Stride  Block/Image       Winner
----------------------------------------------------------------------
Flickr Pure              466.95        51.33         6.43 block_per_image
Flickr + 2K             3113.87        55.26        34.49 block_per_image
Flickr + 4K             5909.22        58.77        63.18  grid_stride

~ Block-per-Image wins 2/3 tests
  (Loses when imbalance is high → need Hybrid)
```

### Fair Benchmark (Per-Pixel Operations)

```
======================================================================
FAIR BENCHMARK SUMMARY
======================================================================

Test                       Imbalance     vs Naive      vs Grid      Status
---------------------------------------------------------------------------
Flickr Pure (1000)             1.39x        8.80x        0.39x     ✗ LOSE
Flickr + 4K                   44.22x      156.48x        0.40x     ✗ LOSE
Flickr + 8K                  156.19x      438.98x        0.40x     ✗ LOSE

>>> For per-pixel ops, Grid-Stride wins. Use Hybrid for image-aware ops. <<<
```

*Tested on NVIDIA RTX 4090, January 2026*
