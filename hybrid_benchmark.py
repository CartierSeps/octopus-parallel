"""
Hybrid Benchmark: Adaptive Block Assignment
============================================
Combines best of both approaches:
- Small images: 1 block per image (locality win)
- Large images: Multiple blocks per image (load balance)

This is the "octopus hybrid" approach.

Author: Matthew
Date: January 2026
"""

from numba import cuda
import numpy as np
import time
from pathlib import Path
from PIL import Image
from scipy import stats
import math

# ============================================
# IMAGE LOADING
# ============================================

def load_images_2d(data_dir, max_images=None):
    """Load images keeping 2D structure info."""
    data_path = Path(data_dir)
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
        image_files.extend(data_path.glob(ext))
    
    image_files = sorted(image_files)[:max_images] if max_images else sorted(image_files)
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"Loading {len(image_files)} images...")
    
    all_pixels = []
    widths = []
    heights = []
    offsets = []
    sizes = []
    
    current_offset = 0
    
    for img_path in image_files:
        with Image.open(img_path) as img:
            img = img.convert('L')
            w, h = img.size
            pixels = np.array(img, dtype=np.float32).flatten() / 255.0
            
            all_pixels.append(pixels)
            widths.append(w)
            heights.append(h)
            offsets.append(current_offset)
            sizes.append(w * h)
            current_offset += w * h
    
    images_flat = np.concatenate(all_pixels)
    
    return (images_flat, 
            np.array(offsets, dtype=np.int64),
            np.array(sizes, dtype=np.int64),
            np.array(widths, dtype=np.int32),
            np.array(heights, dtype=np.int32))


def add_large_synthetic_2d(images_flat, offsets, sizes, widths, heights, large_size=(2048, 2048)):
    """Add large synthetic image."""
    w, h = large_size
    large_pixels = np.random.rand(w * h).astype(np.float32)
    
    new_flat = np.concatenate([images_flat, large_pixels])
    new_offsets = np.concatenate([offsets, [len(images_flat)]])
    new_sizes = np.concatenate([sizes, [w * h]])
    new_widths = np.concatenate([widths, [w]])
    new_heights = np.concatenate([heights, [h]])
    
    return new_flat, new_offsets, new_sizes, new_widths, new_heights


# ============================================
# HYBRID BLOCK ASSIGNMENT
# ============================================

def compute_hybrid_assignment(sizes, threshold=65536):
    """
    Adaptive block assignment:
    - Small images (size <= threshold): 1 block
    - Large images (size > threshold): multiple blocks
    
    Returns:
    - block_to_image: which image each block handles
    - block_start: starting pixel (local to image) for each block  
    - block_end: ending pixel (local to image) for each block
    """
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
            num_blocks = math.ceil(size / threshold)
            pixels_per_block = math.ceil(size / num_blocks)
            
            for b in range(num_blocks):
                block_to_image.append(img_id)
                start = b * pixels_per_block
                end = min((b + 1) * pixels_per_block, size)
                block_start.append(start)
                block_end.append(end)
    
    return (np.array(block_to_image, dtype=np.int32),
            np.array(block_start, dtype=np.int64),
            np.array(block_end, dtype=np.int64))


# ============================================
# KERNELS
# ============================================

@cuda.jit
def grid_stride_blur_kernel(images_flat, offsets, sizes, widths, heights, num_images, output):
    """Grid-stride with blur (baseline)."""
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = images_flat.shape[0]
    
    for pixel_idx in range(tid, n, stride):
        # Find which image (expensive search)
        img_id = 0
        for i in range(num_images):
            if i == num_images - 1:
                img_id = i
                break
            if pixel_idx >= offsets[i] and pixel_idx < offsets[i + 1]:
                img_id = i
                break
        
        offset = offsets[img_id]
        w = widths[img_id]
        h = heights[img_id]
        
        local_idx = pixel_idx - offset
        y = local_idx // w
        x = local_idx % w
        
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            output[pixel_idx] = images_flat[pixel_idx]
        else:
            total = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    neighbor_idx = offset + (y + dy) * w + (x + dx)
                    total += images_flat[neighbor_idx]
            output[pixel_idx] = total / 9.0


@cuda.jit
def block_per_image_blur_kernel(images_flat, offsets, sizes, widths, heights, output):
    """Block-per-image (previous approach)."""
    img_id = cuda.blockIdx.x
    
    if img_id >= sizes.shape[0]:
        return
    
    offset = offsets[img_id]
    w = widths[img_id]
    h = heights[img_id]
    num_pixels = sizes[img_id]
    
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    
    for local_idx in range(tid, num_pixels, stride):
        y = local_idx // w
        x = local_idx % w
        global_idx = offset + local_idx
        
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            output[global_idx] = images_flat[global_idx]
        else:
            total = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    neighbor_idx = offset + (y + dy) * w + (x + dx)
                    total += images_flat[neighbor_idx]
            output[global_idx] = total / 9.0


@cuda.jit
def hybrid_blur_kernel(images_flat, offsets, widths, heights,
                       block_to_image, block_start, block_end, output):
    """
    HYBRID: Adaptive block assignment.
    - Each block knows its image and pixel range
    - Small images: 1 block covers whole image
    - Large images: multiple blocks, each covers a chunk
    - Threads within block do coalesced access
    """
    block_id = cuda.blockIdx.x
    
    if block_id >= block_to_image.shape[0]:
        return
    
    # This block's assignment
    img_id = block_to_image[block_id]
    local_start = block_start[block_id]
    local_end = block_end[block_id]
    
    # Image info
    offset = offsets[img_id]
    w = widths[img_id]
    h = heights[img_id]
    
    # Threads cooperate within this block's range
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    
    for local_idx in range(local_start + tid, local_end, stride):
        y = local_idx // w
        x = local_idx % w
        global_idx = offset + local_idx
        
        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
            output[global_idx] = images_flat[global_idx]
        else:
            total = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    neighbor_idx = offset + (y + dy) * w + (x + dx)
                    total += images_flat[neighbor_idx]
            output[global_idx] = total / 9.0


# ============================================
# BENCHMARK
# ============================================

def run_hybrid_benchmark(images_flat, offsets, sizes, widths, heights, name, 
                         threshold=65536, warmup=3, runs=20):
    """Benchmark comparing all three approaches."""
    
    print(f"\n{'='*70}")
    print(f"HYBRID BENCHMARK: {name}")
    print(f"{'='*70}")
    
    num_images = len(sizes)
    total_pixels = len(images_flat)
    imbalance = max(sizes) / (total_pixels / num_images)
    
    print(f"\nDataset: {num_images} images, {total_pixels:,} pixels")
    print(f"Imbalance: {imbalance:.2f}x")
    print(f"Threshold: {threshold:,} pixels/block")
    
    # Compute hybrid assignment
    block_to_image, block_start, block_end = compute_hybrid_assignment(sizes, threshold)
    num_hybrid_blocks = len(block_to_image)
    
    print(f"\nBlock assignment:")
    print(f"  Block-per-Image: {num_images} blocks")
    print(f"  Hybrid: {num_hybrid_blocks} blocks")
    
    # Show breakdown
    small_images = sum(1 for s in sizes if s <= threshold)
    large_images = num_images - small_images
    print(f"  Small images (1 block each): {small_images}")
    print(f"  Large images (subdivided): {large_images}")
    
    # Setup
    threads_per_block = 256
    
    d_images = cuda.to_device(images_flat)
    d_offsets = cuda.to_device(offsets)
    d_sizes = cuda.to_device(sizes)
    d_widths = cuda.to_device(widths)
    d_heights = cuda.to_device(heights)
    d_output = cuda.device_array(total_pixels, dtype=np.float32)
    
    d_block_to_image = cuda.to_device(block_to_image)
    d_block_start = cuda.to_device(block_start)
    d_block_end = cuda.to_device(block_end)
    
    results = {}
    
    # ========================================
    # Test 1: Grid-Stride (baseline)
    # ========================================
    print(f"\nBenchmarking Grid-Stride...")
    grid_blocks = 256
    
    for _ in range(warmup):
        grid_stride_blur_kernel[grid_blocks, threads_per_block](
            d_images, d_offsets, d_sizes, d_widths, d_heights, num_images, d_output)
    cuda.synchronize()
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        grid_stride_blur_kernel[grid_blocks, threads_per_block](
            d_images, d_offsets, d_sizes, d_widths, d_heights, num_images, d_output)
        cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    results['grid_stride'] = np.mean(times) * 1000
    print(f"  Grid-Stride: {results['grid_stride']:.3f} ms")
    
    # ========================================
    # Test 2: Block-per-Image (previous)
    # ========================================
    print(f"\nBenchmarking Block-per-Image...")
    
    for _ in range(warmup):
        block_per_image_blur_kernel[num_images, threads_per_block](
            d_images, d_offsets, d_sizes, d_widths, d_heights, d_output)
    cuda.synchronize()
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        block_per_image_blur_kernel[num_images, threads_per_block](
            d_images, d_offsets, d_sizes, d_widths, d_heights, d_output)
        cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    results['block_per_image'] = np.mean(times) * 1000
    print(f"  Block-per-Image: {results['block_per_image']:.3f} ms")
    
    # ========================================
    # Test 3: HYBRID (new!)
    # ========================================
    print(f"\nBenchmarking Hybrid ({num_hybrid_blocks} blocks)...")
    
    for _ in range(warmup):
        hybrid_blur_kernel[num_hybrid_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights,
            d_block_to_image, d_block_start, d_block_end, d_output)
    cuda.synchronize()
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        hybrid_blur_kernel[num_hybrid_blocks, threads_per_block](
            d_images, d_offsets, d_widths, d_heights,
            d_block_to_image, d_block_start, d_block_end, d_output)
        cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    results['hybrid'] = np.mean(times) * 1000
    print(f"  Hybrid: {results['hybrid']:.3f} ms")
    
    # ========================================
    # Results
    # ========================================
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    baseline = results['grid_stride']
    
    print(f"\n  {'Method':<20} {'Time (ms)':>12} {'vs Grid-Stride':>15} {'Status':>10}")
    print(f"  {'-'*60}")
    
    for method, time_ms in results.items():
        speedup = baseline / time_ms
        if speedup > 1.05:
            status = "✓ WIN"
        elif speedup > 0.95:
            status = "~ TIE"
        else:
            status = "✗ LOSE"
        print(f"  {method:<20} {time_ms:>12.3f} {speedup:>14.2f}x  {status}")
    
    # Best method
    best = min(results, key=results.get)
    print(f"\n  >>> BEST: {best} <<<")
    
    return results


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("HYBRID BENCHMARK")
    print("Adaptive Block Assignment: Best of Both Worlds")
    print("=" * 70)
    print()
    print("Strategy:")
    print("  - Small images: 1 block (locality)")
    print("  - Large images: multiple blocks (load balance)")
    print()
    
    all_results = []
    
    # ========================================
    # Test 1: Flickr Pure (low imbalance)
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 1: Flickr Pure (500 images, low imbalance)")
    print("█" * 70)
    
    try:
        data = load_images_2d("Images", max_images=500)
        r = run_hybrid_benchmark(*data, "Flickr Pure (500)")
        all_results.append(('Flickr Pure', r))
    except Exception as e:
        print(f"Skipping: {e}")
    
    # ========================================
    # Test 2: Flickr + 2K (medium imbalance)
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 2: Flickr + 2K Image (medium imbalance)")
    print("█" * 70)
    
    try:
        data = load_images_2d("Images", max_images=500)
        data = add_large_synthetic_2d(*data, (2048, 2048))
        r = run_hybrid_benchmark(*data, "Flickr + 2K")
        all_results.append(('Flickr + 2K', r))
    except Exception as e:
        print(f"Skipping: {e}")
    
    # ========================================
    # Test 3: Flickr + 4K (high imbalance)
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 3: Flickr + 4K Image (high imbalance)")
    print("█" * 70)
    
    try:
        data = load_images_2d("Images", max_images=500)
        data = add_large_synthetic_2d(*data, (3840, 2160))
        r = run_hybrid_benchmark(*data, "Flickr + 4K")
        all_results.append(('Flickr + 4K', r))
    except Exception as e:
        print(f"Skipping: {e}")
    
    # ========================================
    # Test 4: Flickr + 8K (extreme imbalance)
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 4: Flickr + 8K Image (extreme imbalance)")
    print("█" * 70)
    
    try:
        data = load_images_2d("Images", max_images=500)
        data = add_large_synthetic_2d(*data, (7680, 4320))
        r = run_hybrid_benchmark(*data, "Flickr + 8K")
        all_results.append(('Flickr + 8K', r))
    except Exception as e:
        print(f"Skipping: {e}")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n\n" + "=" * 70)
    print("HYBRID BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"\n  {'Test':<20} {'Grid-Stride':>12} {'Block/Image':>12} {'Hybrid':>12} {'Winner':>15}")
    print(f"  {'-'*75}")
    
    for name, r in all_results:
        winner = min(r, key=r.get)
        print(f"  {name:<20} {r['grid_stride']:>12.2f} {r['block_per_image']:>12.2f} {r['hybrid']:>12.2f} {winner:>15}")
    
    print("\n" + "=" * 70)
    
    # Count wins
    hybrid_wins = sum(1 for _, r in all_results if r['hybrid'] == min(r.values()))
    block_wins = sum(1 for _, r in all_results if r['block_per_image'] == min(r.values()))
    grid_wins = sum(1 for _, r in all_results if r['grid_stride'] == min(r.values()))
    
    print(f"\nWin count:")
    print(f"  Hybrid: {hybrid_wins}/{len(all_results)}")
    print(f"  Block-per-Image: {block_wins}/{len(all_results)}")
    print(f"  Grid-Stride: {grid_wins}/{len(all_results)}")
    
    if hybrid_wins == len(all_results):
        print(f"\n🐙 HYBRID WINS ALL TESTS!")
        print("  → Adaptive block assignment = best of both worlds!")
    elif hybrid_wins > 0:
        print(f"\n🐙 Hybrid wins {hybrid_wins}/{len(all_results)} tests")
    
    print("\n🐙 Hybrid benchmark complete!")


if __name__ == "__main__":
    main()