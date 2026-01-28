"""
Octopus-Inspired Parallel Load Balancing Benchmark V2
======================================================

OPTIMIZED VERSION:
- Flattened memory layout (coalesced access)
- Single contiguous data array
- Assignments are just [start, end] in global index space

Author: Matthew
Date: January 27, 2025
"""

from numba import cuda
import numpy as np
import time

# ============================================
# NAIVE APPROACH: One task per thread
# ============================================

@cuda.jit
def naive_kernel(task_offsets, task_sizes, flat_data, output):
    """
    Traditional approach: each thread processes one task.
    """
    tid = cuda.grid(1)
    num_tasks = task_sizes.shape[0]
    
    if tid < num_tasks:
        start = task_offsets[tid]
        end = start + task_sizes[tid]
        
        result = 0.0
        for i in range(start, end):
            result += flat_data[i] * 1.001
        output[tid] = result


# ============================================
# YOUR APPROACH V2: Flattened, coalesced access
# ============================================

@cuda.jit
def balanced_kernel_v2(flat_data, work_start, work_end, output):
    """
    OPTIMIZED: Each thread gets a contiguous range in flattened data.
    
    work_start[tid] = starting index in flat_data
    work_end[tid] = ending index in flat_data
    
    Memory access is now COALESCED - adjacent threads access adjacent memory.
    """
    tid = cuda.grid(1)
    
    if tid < work_start.shape[0]:
        start = work_start[tid]
        end = work_end[tid]
        
        result = 0.0
        for i in range(start, end):
            result += flat_data[i] * 1.001
        
        output[tid] = result


# ============================================
# PRE-COMPUTATION V2: Simple contiguous ranges
# ============================================

def compute_balanced_assignments_v2(task_sizes, num_threads):
    """
    OPTIMIZED: Just compute start/end in flattened space.
    No complex list-of-lists, just contiguous ranges.
    
    Returns:
        work_start: [num_threads] start index for each thread
        work_end: [num_threads] end index for each thread
    """
    total_work = sum(task_sizes)
    work_per_thread = total_work // num_threads
    remainder = total_work % num_threads
    
    work_start = np.zeros(num_threads, dtype=np.int64)
    work_end = np.zeros(num_threads, dtype=np.int64)
    
    current_pos = 0
    for tid in range(num_threads):
        work_start[tid] = current_pos
        # Threads 0..remainder-1 get one extra unit
        thread_work = work_per_thread + (1 if tid < remainder else 0)
        current_pos += thread_work
        work_end[tid] = current_pos
    
    return work_start, work_end


def flatten_tasks(task_sizes):
    """
    Flatten multiple tasks into single contiguous array.
    Returns offsets for naive kernel.
    """
    total = sum(task_sizes)
    flat_data = np.random.rand(total).astype(np.float32)
    
    offsets = np.zeros(len(task_sizes), dtype=np.int64)
    current = 0
    for i, size in enumerate(task_sizes):
        offsets[i] = current
        current += size
    
    return flat_data, offsets


# ============================================
# BENCHMARK V2
# ============================================

def run_benchmark_v2(task_sizes, num_threads=4, warmup_runs=3, benchmark_runs=10):
    """
    Run optimized benchmark with coalesced memory access.
    """
    print("=" * 60)
    print("OCTOPUS BENCHMARK V2 (OPTIMIZED MEMORY ACCESS)")
    print("=" * 60)
    
    num_tasks = len(task_sizes)
    total_work = sum(task_sizes)
    max_task = max(task_sizes)
    
    print(f"\nConfiguration:")
    print(f"  Tasks: {num_tasks}")
    print(f"  Task sizes: {[int(x) for x in task_sizes]}")
    print(f"  Total work units: {total_work:,}")
    print(f"  Max single task: {max_task:,}")
    print(f"  Threads: {num_threads}")
    print(f"  Work per thread (balanced): {total_work // num_threads:,}")
    
    # Imbalance ratio
    imbalance = max_task / (total_work / num_tasks)
    print(f"  Imbalance ratio: {imbalance:.2f}x")
    
    # Flatten data
    print(f"\nFlattening task data...")
    flat_data, task_offsets = flatten_tasks(task_sizes)
    task_sizes_np = np.array(task_sizes, dtype=np.int64)
    
    # Pre-compute balanced assignments
    print(f"Pre-computing balanced assignments...")
    precompute_start = time.perf_counter()
    work_start, work_end = compute_balanced_assignments_v2(task_sizes, num_threads)
    precompute_time = time.perf_counter() - precompute_start
    print(f"  Pre-computation time: {precompute_time*1000:.4f} ms")
    
    # Show distribution
    print(f"\nWork distribution:")
    print(f"  Naive (per task):")
    for i, size in enumerate(task_sizes):
        bar = "█" * min(int(size / max_task * 20), 20)
        print(f"    Task {i}: {bar} {int(size):,}")
    
    print(f"\n  Balanced (per thread):")
    for tid in range(num_threads):
        work = work_end[tid] - work_start[tid]
        bar = "█" * min(int(work / (total_work/num_threads) * 20), 20)
        print(f"    Thread {tid}: {bar} {work:,}")
    
    # Transfer to GPU
    print(f"\nTransferring to GPU...")
    d_flat_data = cuda.to_device(flat_data)
    d_task_offsets = cuda.to_device(task_offsets)
    d_task_sizes = cuda.to_device(task_sizes_np)
    d_work_start = cuda.to_device(work_start)
    d_work_end = cuda.to_device(work_end)
    
    d_output_naive = cuda.device_array(num_tasks, dtype=np.float32)
    d_output_balanced = cuda.device_array(num_threads, dtype=np.float32)
    
    # Kernel config
    threads_per_block = 256
    blocks_naive = (num_tasks + threads_per_block - 1) // threads_per_block
    blocks_balanced = (num_threads + threads_per_block - 1) // threads_per_block
    
    # Warmup
    print(f"\nWarmup ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        naive_kernel[blocks_naive, threads_per_block](
            d_task_offsets, d_task_sizes, d_flat_data, d_output_naive)
        balanced_kernel_v2[blocks_balanced, threads_per_block](
            d_flat_data, d_work_start, d_work_end, d_output_balanced)
    cuda.synchronize()
    
    # Benchmark NAIVE
    print(f"\nBenchmarking NAIVE ({benchmark_runs} runs)...")
    naive_times = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        naive_kernel[blocks_naive, threads_per_block](
            d_task_offsets, d_task_sizes, d_flat_data, d_output_naive)
        cuda.synchronize()
        naive_times.append(time.perf_counter() - start)
    
    naive_avg = np.mean(naive_times) * 1000
    naive_std = np.std(naive_times) * 1000
    
    # Benchmark BALANCED
    print(f"Benchmarking BALANCED ({benchmark_runs} runs)...")
    balanced_times = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        balanced_kernel_v2[blocks_balanced, threads_per_block](
            d_flat_data, d_work_start, d_work_end, d_output_balanced)
        cuda.synchronize()
        balanced_times.append(time.perf_counter() - start)
    
    balanced_avg = np.mean(balanced_times) * 1000
    balanced_std = np.std(balanced_times) * 1000
    
    # Verify correctness
    output_naive = d_output_naive.copy_to_host()
    output_balanced = d_output_balanced.copy_to_host()
    naive_total = np.sum(output_naive)
    balanced_total = np.sum(output_balanced)
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nTheoretical Analysis:")
    print(f"  Naive time ∝ max(task) = {max_task:,}")
    print(f"  Balanced time ∝ total/threads = {total_work // num_threads:,}")
    theoretical_speedup = max_task / (total_work / num_threads)
    print(f"  Theoretical max speedup: {theoretical_speedup:.2f}x")
    
    print(f"\nActual Timing:")
    print(f"  Naive:    {naive_avg:.3f} ms (±{naive_std:.3f})")
    print(f"  Balanced: {balanced_avg:.3f} ms (±{balanced_std:.3f})")
    
    speedup = naive_avg / balanced_avg
    print(f"\n  >>> SPEEDUP: {speedup:.2f}x <<<")
    
    if speedup > 1:
        improvement = (1 - balanced_avg/naive_avg) * 100
        print(f"  ✓ BALANCED is {speedup:.2f}x FASTER")
        print(f"  ✓ Time saved: {improvement:.1f}%")
        print(f"  ✓ Achieved {speedup/theoretical_speedup*100:.1f}% of theoretical max")
    else:
        print(f"  ✗ Naive was faster")
    
    print(f"\nCorrectness Check:")
    print(f"  Naive sum:    {naive_total:.2f}")
    print(f"  Balanced sum: {balanced_total:.2f}")
    diff = abs(naive_total - balanced_total) / naive_total * 100
    print(f"  Difference:   {diff:.4f}%")
    
    print("=" * 60)
    
    return {
        'naive_ms': naive_avg,
        'balanced_ms': balanced_avg,
        'speedup': speedup,
        'theoretical_speedup': theoretical_speedup,
        'imbalance_ratio': imbalance
    }


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GPU INFO")
    print("=" * 60)
    print(cuda.gpus)
    
    results = []
    
    # ----------------------------------------
    # TEST 1: Original example
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print(">>> TEST 1: Small scale, 8:4:2:6 ratio")
    print("█" * 60)
    r = run_benchmark_v2([800000, 400000, 200000, 600000], num_threads=4)
    results.append(('Test 1: 8:4:2:6', r))
    
    # ----------------------------------------
    # TEST 2: Extreme imbalance
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print(">>> TEST 2: Extreme imbalance (10:1:1:1)")
    print("█" * 60)
    r = run_benchmark_v2([2000000, 200000, 200000, 200000], num_threads=4)
    results.append(('Test 2: 10:1:1:1', r))
    
    # ----------------------------------------
    # TEST 3: Very extreme (100:1)
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print(">>> TEST 3: Very extreme imbalance (50:1)")
    print("█" * 60)
    r = run_benchmark_v2([5000000, 100000, 100000, 100000], num_threads=4)
    results.append(('Test 3: 50:1', r))
    
    # ----------------------------------------
    # TEST 4: More threads
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print(">>> TEST 4: 8 threads, high variance")
    print("█" * 60)
    r = run_benchmark_v2([1000000, 100000, 500000, 50000, 800000, 200000, 900000, 150000], 
                         num_threads=8)
    results.append(('Test 4: 8 threads', r))
    
    # ----------------------------------------
    # TEST 5: Many tasks, random
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print(">>> TEST 5: 16 random tasks, 8 threads")
    print("█" * 60)
    np.random.seed(42)
    tasks = list(np.random.randint(100000, 2000000, size=16))
    r = run_benchmark_v2(tasks, num_threads=8)
    results.append(('Test 5: Random', r))
    
    # ----------------------------------------
    # TEST 6: Large scale
    # ----------------------------------------
    print("\n\n" + "█" * 60)
    print(">>> TEST 6: Large scale (100M total work)")
    print("█" * 60)
    r = run_benchmark_v2([50000000, 10000000, 10000000, 10000000, 10000000, 10000000], 
                         num_threads=6)
    results.append(('Test 6: Large', r))
    
    # ----------------------------------------
    # SUMMARY
    # ----------------------------------------
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Test':<25} {'Imbalance':>10} {'Theoretical':>12} {'Actual':>10} {'Status':>10}")
    print("-" * 70)
    for name, r in results:
        status = "✓ WIN" if r['speedup'] > 1.05 else ("~ TIE" if r['speedup'] > 0.95 else "✗ LOSE")
        print(f"{name:<25} {r['imbalance_ratio']:>10.1f}x {r['theoretical_speedup']:>11.2f}x {r['speedup']:>9.2f}x {status:>10}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    wins = sum(1 for _, r in results if r['speedup'] > 1.05)
    print(f"\nBalanced approach wins: {wins}/{len(results)} tests")
    print(f"\nKey insight: Higher imbalance ratio → More speedup")
    print("\nBenchmark complete! 🐙")
