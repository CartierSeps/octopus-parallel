"""
NLP Ragged Tensor Benchmark
===========================
Validates Hybrid approach on NLP-style variable-length data.

Simulates:
- Variable-length sentences (10-512 tokens)
- Preprocessing kernel (e.g., normalization, embedding transform)

Compares:
1. Grid-Stride Fair: O(total_tokens) mapping table
2. Hybrid: O(num_blocks) block metadata

Author: Matthew
Date: January 2026
"""

from numba import cuda
import numpy as np
import time
import math

# ============================================
# DATA GENERATION
# ============================================

def generate_nlp_batch(num_sentences=1000, length_distribution='realistic'):
    """
    Generate variable-length sentence batch.
    
    Args:
        num_sentences: Number of sentences
        length_distribution: 
            'realistic' - mixed short/medium/long (like real NLP data)
            'uniform' - uniform random lengths
            'extreme' - few very long sequences
    
    Returns:
        data: Flattened token embeddings (float32)
        offsets: Start index of each sentence
        lengths: Length of each sentence
    """
    
    if length_distribution == 'realistic':
        # Realistic NLP distribution:
        # 80% short (10-50), 15% medium (50-200), 5% long (200-512)
        lengths = []
        n_short = int(num_sentences * 0.80)
        n_medium = int(num_sentences * 0.15)
        n_long = num_sentences - n_short - n_medium
        
        lengths.extend(np.random.randint(10, 50, n_short))
        lengths.extend(np.random.randint(50, 200, n_medium))
        lengths.extend(np.random.randint(200, 512, n_long))
        np.random.shuffle(lengths)
        lengths = np.array(lengths, dtype=np.int64)
        
    elif length_distribution == 'uniform':
        lengths = np.random.randint(10, 512, num_sentences).astype(np.int64)
        
    elif length_distribution == 'extreme':
        # 95% short, 5% very long (simulates outliers)
        lengths = []
        n_short = int(num_sentences * 0.95)
        n_long = num_sentences - n_short
        
        lengths.extend(np.random.randint(10, 50, n_short))
        lengths.extend(np.random.randint(1000, 4096, n_long))  # Very long!
        np.random.shuffle(lengths)
        lengths = np.array(lengths, dtype=np.int64)
    
    else:
        raise ValueError(f"Unknown distribution: {length_distribution}")
    
    # Build offsets
    offsets = np.zeros(num_sentences, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths[:-1])
    
    total_tokens = np.sum(lengths)
    
    # Generate random embeddings (simulating token embeddings)
    # In real NLP, this would be embedding lookup results
    data = np.random.randn(total_tokens).astype(np.float32)
    
    return data, offsets, lengths


def generate_vision_patches(num_images=500, patch_distribution='realistic'):
    """
    Generate variable-size image patches.
    
    Simulates:
    - Object detection: cropped regions of different sizes
    - Multi-scale features: patches at different resolutions
    """
    
    if patch_distribution == 'realistic':
        # Realistic patch sizes (flattened)
        # Small: 32x32=1024, Medium: 64x64=4096, Large: 128x128=16384
        sizes = []
        n_small = int(num_images * 0.60)
        n_medium = int(num_images * 0.30)
        n_large = num_images - n_small - n_medium
        
        sizes.extend([1024] * n_small)   # 32x32
        sizes.extend([4096] * n_medium)  # 64x64
        sizes.extend([16384] * n_large)  # 128x128
        np.random.shuffle(sizes)
        sizes = np.array(sizes, dtype=np.int64)
        
    elif patch_distribution == 'extreme':
        # Mix of tiny and huge patches
        sizes = []
        n_tiny = int(num_images * 0.90)
        n_huge = num_images - n_tiny
        
        sizes.extend([256] * n_tiny)      # 16x16
        sizes.extend([65536] * n_huge)    # 256x256
        np.random.shuffle(sizes)
        sizes = np.array(sizes, dtype=np.int64)
    
    else:
        sizes = np.random.randint(1024, 16384, num_images).astype(np.int64)
    
    offsets = np.zeros(num_images, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes[:-1])
    
    total_pixels = np.sum(sizes)
    data = np.random.randn(total_pixels).astype(np.float32)
    
    return data, offsets, sizes


# ============================================
# SETUP FUNCTIONS
# ============================================

def setup_grid_stride_fair(offsets, lengths, total_elements):
    """Build element-to-sequence mapping (O(N) space and time)."""
    element_to_seq = np.zeros(total_elements, dtype=np.int32)
    
    for seq_id in range(len(lengths)):
        start = offsets[seq_id]
        end = start + lengths[seq_id]
        element_to_seq[start:end] = seq_id
    
    return element_to_seq


def setup_hybrid(lengths, threshold=1024):
    """Build block assignment (O(num_blocks) space and time)."""
    block_to_seq = []
    block_start = []
    block_end = []
    
    for seq_id, length in enumerate(lengths):
        if length <= threshold:
            block_to_seq.append(seq_id)
            block_start.append(0)
            block_end.append(length)
        else:
            num_blocks = math.ceil(length / threshold)
            elements_per_block = math.ceil(length / num_blocks)
            
            for b in range(num_blocks):
                block_to_seq.append(seq_id)
                start = b * elements_per_block
                end = min((b + 1) * elements_per_block, length)
                block_start.append(start)
                block_end.append(end)
    
    return (np.array(block_to_seq, dtype=np.int32),
            np.array(block_start, dtype=np.int64),
            np.array(block_end, dtype=np.int64))


# ============================================
# KERNELS
# ============================================

@cuda.jit
def grid_stride_fair_kernel(data, offsets, lengths, element_to_seq, output):
    """
    Grid-Stride with O(1) element_to_seq lookup.
    Simulates: Layer normalization per sequence.
    """
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)
    n = data.shape[0]
    
    for idx in range(tid, n, stride):
        seq_id = element_to_seq[idx]
        seq_start = offsets[seq_id]
        seq_len = lengths[seq_id]
        
        # Compute mean of sequence (simplified LayerNorm)
        # In real impl, this would be more complex
        local_idx = idx - seq_start
        
        # Simple transform: normalize by position
        # (Real LayerNorm needs reduction, this is simplified)
        output[idx] = data[idx] * (1.0 / (local_idx + 1))


@cuda.jit
def hybrid_kernel(data, offsets, lengths, 
                  block_to_seq, block_start, block_end, output):
    """
    Hybrid: Block-level assignment.
    Each block knows its sequence and range.
    """
    block_id = cuda.blockIdx.x
    
    if block_id >= block_to_seq.shape[0]:
        return
    
    seq_id = block_to_seq[block_id]
    local_start = block_start[block_id]
    local_end = block_end[block_id]
    
    seq_offset = offsets[seq_id]
    
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    
    for local_idx in range(local_start + tid, local_end, stride):
        global_idx = seq_offset + local_idx
        
        # Same transform as grid-stride
        output[global_idx] = data[global_idx] * (1.0 / (local_idx + 1))


# ============================================
# BENCHMARK
# ============================================

def run_ragged_benchmark(data, offsets, lengths, name, threshold=1024, 
                         warmup=3, runs=20):
    """Run benchmark comparing Grid-Stride-Fair vs Hybrid."""
    
    print(f"\n{'='*70}")
    print(f"RAGGED TENSOR BENCHMARK: {name}")
    print(f"{'='*70}")
    
    num_sequences = len(lengths)
    total_elements = len(data)
    max_len = np.max(lengths)
    min_len = np.min(lengths)
    avg_len = total_elements / num_sequences
    imbalance = max_len / avg_len
    
    print(f"\nDataset:")
    print(f"  Sequences: {num_sequences:,}")
    print(f"  Total elements: {total_elements:,}")
    print(f"  Length range: {min_len:,} - {max_len:,}")
    print(f"  Average length: {avg_len:.1f}")
    print(f"  Imbalance: {imbalance:.1f}x")
    
    threads_per_block = 256
    grid_blocks = 256
    
    # Common GPU arrays
    d_data = cuda.to_device(data)
    d_offsets = cuda.to_device(offsets)
    d_lengths = cuda.to_device(lengths)
    d_output = cuda.device_array(total_elements, dtype=np.float32)
    
    results = {}
    
    # ========================================
    # Grid-Stride Fair
    # ========================================
    print(f"\n[Grid-Stride Fair]")
    
    # Setup
    setup_times = []
    for _ in range(5):
        start = time.perf_counter()
        element_to_seq = setup_grid_stride_fair(offsets, lengths, total_elements)
        setup_times.append(time.perf_counter() - start)
    
    grid_setup_ms = np.mean(setup_times) * 1000
    grid_memory_bytes = element_to_seq.nbytes
    grid_memory_mb = grid_memory_bytes / (1024 * 1024)
    
    print(f"  Setup: {grid_setup_ms:.2f} ms")
    print(f"  Memory: {grid_memory_mb:.2f} MB")
    
    d_element_to_seq = cuda.to_device(element_to_seq)
    
    # Warmup
    for _ in range(warmup):
        grid_stride_fair_kernel[grid_blocks, threads_per_block](
            d_data, d_offsets, d_lengths, d_element_to_seq, d_output)
    cuda.synchronize()
    
    # Benchmark
    kernel_times = []
    for _ in range(runs):
        start = time.perf_counter()
        grid_stride_fair_kernel[grid_blocks, threads_per_block](
            d_data, d_offsets, d_lengths, d_element_to_seq, d_output)
        cuda.synchronize()
        kernel_times.append(time.perf_counter() - start)
    
    grid_kernel_ms = np.mean(kernel_times) * 1000
    grid_total_ms = grid_setup_ms + grid_kernel_ms
    
    print(f"  Kernel: {grid_kernel_ms:.2f} ms")
    print(f"  TOTAL: {grid_total_ms:.2f} ms")
    
    results['grid_fair'] = {
        'setup_ms': grid_setup_ms,
        'memory_mb': grid_memory_mb,
        'memory_bytes': grid_memory_bytes,
        'kernel_ms': grid_kernel_ms,
        'total_ms': grid_total_ms
    }
    
    del d_element_to_seq
    
    # ========================================
    # Hybrid
    # ========================================
    print(f"\n[Hybrid]")
    
    # Setup
    setup_times = []
    for _ in range(5):
        start = time.perf_counter()
        block_to_seq, block_start, block_end = setup_hybrid(lengths, threshold)
        setup_times.append(time.perf_counter() - start)
    
    hybrid_setup_ms = np.mean(setup_times) * 1000
    hybrid_memory_bytes = block_to_seq.nbytes + block_start.nbytes + block_end.nbytes
    hybrid_memory_mb = hybrid_memory_bytes / (1024 * 1024)
    num_blocks = len(block_to_seq)
    
    print(f"  Setup: {hybrid_setup_ms:.3f} ms")
    print(f"  Memory: {hybrid_memory_mb:.4f} MB ({num_blocks} blocks)")
    
    d_block_to_seq = cuda.to_device(block_to_seq)
    d_block_start = cuda.to_device(block_start)
    d_block_end = cuda.to_device(block_end)
    
    # Warmup
    for _ in range(warmup):
        hybrid_kernel[num_blocks, threads_per_block](
            d_data, d_offsets, d_lengths,
            d_block_to_seq, d_block_start, d_block_end, d_output)
    cuda.synchronize()
    
    # Benchmark
    kernel_times = []
    for _ in range(runs):
        start = time.perf_counter()
        hybrid_kernel[num_blocks, threads_per_block](
            d_data, d_offsets, d_lengths,
            d_block_to_seq, d_block_start, d_block_end, d_output)
        cuda.synchronize()
        kernel_times.append(time.perf_counter() - start)
    
    hybrid_kernel_ms = np.mean(kernel_times) * 1000
    hybrid_total_ms = hybrid_setup_ms + hybrid_kernel_ms
    
    print(f"  Kernel: {hybrid_kernel_ms:.2f} ms")
    print(f"  TOTAL: {hybrid_total_ms:.2f} ms")
    
    results['hybrid'] = {
        'setup_ms': hybrid_setup_ms,
        'memory_mb': hybrid_memory_mb,
        'memory_bytes': hybrid_memory_bytes,
        'kernel_ms': hybrid_kernel_ms,
        'total_ms': hybrid_total_ms
    }
    
    # ========================================
    # Comparison
    # ========================================
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    
    setup_speedup = grid_setup_ms / hybrid_setup_ms
    memory_ratio = grid_memory_bytes / hybrid_memory_bytes
    kernel_ratio = grid_kernel_ms / hybrid_kernel_ms
    total_speedup = grid_total_ms / hybrid_total_ms
    
    print(f"\n  {'Metric':<20} {'Grid-Fair':>15} {'Hybrid':>15} {'Ratio':>12}")
    print(f"  {'-'*65}")
    print(f"  {'Setup':<20} {grid_setup_ms:>14.2f}ms {hybrid_setup_ms:>14.3f}ms {setup_speedup:>11.0f}x")
    print(f"  {'Memory':<20} {grid_memory_mb:>14.2f}MB {hybrid_memory_mb:>14.4f}MB {memory_ratio:>11.0f}x")
    print(f"  {'Kernel':<20} {grid_kernel_ms:>14.2f}ms {hybrid_kernel_ms:>14.2f}ms {kernel_ratio:>11.2f}x")
    print(f"  {'-'*65}")
    print(f"  {'TOTAL':<20} {grid_total_ms:>14.2f}ms {hybrid_total_ms:>14.2f}ms {total_speedup:>11.2f}x")
    
    if total_speedup > 1.05:
        print(f"\n  >>> HYBRID WINS ({total_speedup:.1f}x faster) <<<")
    elif total_speedup > 0.95:
        print(f"\n  >>> TIE <<<")
    else:
        print(f"\n  >>> GRID-STRIDE WINS <<<")
    
    return {
        'name': name,
        'num_sequences': num_sequences,
        'total_elements': total_elements,
        'imbalance': imbalance,
        'results': results,
        'setup_speedup': setup_speedup,
        'memory_ratio': memory_ratio,
        'kernel_ratio': kernel_ratio,
        'total_speedup': total_speedup
    }


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("NLP / ML RAGGED TENSOR BENCHMARK")
    print("Validating Hybrid on Variable-Length Sequence Data")
    print("=" * 70)
    print()
    print("Use cases:")
    print("  - NLP: Variable-length sentences")
    print("  - Vision: Variable-size patches/crops")
    print("  - Audio: Variable-length segments")
    print()
    
    all_results = []
    
    # ========================================
    # Test 1: NLP Realistic
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 1: NLP Sentences (Realistic Distribution)")
    print("  80% short (10-50), 15% medium (50-200), 5% long (200-512)")
    print("█" * 70)
    
    data, offsets, lengths = generate_nlp_batch(1000, 'realistic')
    r = run_ragged_benchmark(data, offsets, lengths, "NLP Realistic (1K)")
    all_results.append(r)
    
    # ========================================
    # Test 2: NLP Large Scale
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 2: NLP Sentences (Large Scale - 10K)")
    print("█" * 70)
    
    data, offsets, lengths = generate_nlp_batch(10000, 'realistic')
    r = run_ragged_benchmark(data, offsets, lengths, "NLP Realistic (10K)")
    all_results.append(r)
    
    # ========================================
    # Test 3: NLP Extreme (outliers)
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 3: NLP with Extreme Outliers")
    print("  95% short (10-50), 5% very long (1000-4096)")
    print("█" * 70)
    
    data, offsets, lengths = generate_nlp_batch(1000, 'extreme')
    r = run_ragged_benchmark(data, offsets, lengths, "NLP Extreme (1K)")
    all_results.append(r)
    
    # ========================================
    # Test 4: Vision Patches
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 4: Vision Patches (Realistic)")
    print("  60% small (32x32), 30% medium (64x64), 10% large (128x128)")
    print("█" * 70)
    
    data, offsets, lengths = generate_vision_patches(500, 'realistic')
    r = run_ragged_benchmark(data, offsets, lengths, "Vision Patches (500)")
    all_results.append(r)
    
    # ========================================
    # Test 5: Vision Extreme
    # ========================================
    print("\n" + "█" * 70)
    print("TEST 5: Vision Patches (Extreme)")
    print("  90% tiny (16x16), 10% huge (256x256)")
    print("█" * 70)
    
    data, offsets, lengths = generate_vision_patches(500, 'extreme')
    r = run_ragged_benchmark(data, offsets, lengths, "Vision Extreme (500)")
    all_results.append(r)
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n\n" + "=" * 70)
    print("ML RAGGED TENSOR BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"\n  {'Test':<25} {'Elements':>12} {'Setup':>10} {'Memory':>10} {'Kernel':>10} {'TOTAL':>10}")
    print(f"  {'-'*80}")
    
    for r in all_results:
        print(f"  {r['name']:<25} {r['total_elements']:>11,} {r['setup_speedup']:>9.0f}x {r['memory_ratio']:>9.0f}x {r['kernel_ratio']:>9.2f}x {r['total_speedup']:>9.2f}x")
    
    print(f"\n  (Ratios = Grid-Fair / Hybrid, higher = Hybrid wins)")
    
    # Averages
    avg_setup = np.mean([r['setup_speedup'] for r in all_results])
    avg_memory = np.mean([r['memory_ratio'] for r in all_results])
    avg_kernel = np.mean([r['kernel_ratio'] for r in all_results])
    avg_total = np.mean([r['total_speedup'] for r in all_results])
    
    print(f"\n  {'AVERAGE':<25} {'-':>12} {avg_setup:>9.0f}x {avg_memory:>9.0f}x {avg_kernel:>9.2f}x {avg_total:>9.2f}x")
    
    # ========================================
    # CONCLUSION
    # ========================================
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    
    wins = sum(1 for r in all_results if r['total_speedup'] > 1.05)
    
    print(f"""
  Results: Hybrid wins {wins}/{len(all_results)} tests
  
  Average improvements:
    - Setup time: {avg_setup:.0f}x faster
    - Memory: {avg_memory:.0f}x less  
    - Kernel: {'~same' if 0.8 < avg_kernel < 1.2 else f'{avg_kernel:.2f}x'}
    - TOTAL: {avg_total:.2f}x faster
    """)
    
    if avg_total > 1.0:
        print("  VALIDATED: Hybrid approach works for ML ragged tensor preprocessing!")
        print("     Block-level metadata beats per-element mapping on total cost.")
    else:
        print("   NOT VALIDATED: Grid-Stride-Fair wins on this workload.")
    
    print("\n" + "=" * 70)
    print(" ML ragged tensor benchmark complete!")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    results = main()
