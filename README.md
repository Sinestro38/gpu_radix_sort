# Minimal GPU Radix Sort

A simple CUDA implementation of radix sort for integer key-value pairs. The core sorting kernel is just ~110 lines of code and demonstrates basic GPU radix sort concepts. Can easily be extended to other data types by adding few LOC normalizations into uint.

## What it does

- Sorts `(int key, int index)` pairs in descending order
- Handles arrays larger than GPU block size (each thread processes multiple elements)
- Uses 4-bit radix passes with shared memory for histograms and prefix scans

## Files

- `radix_sort.h` - Header with function declarations
- `radix_sort.cu` - Main CUDA kernel (~110 LOC)
- `test_radix_sort.cu` - Correctness tests with CPU verification
- `benchmark_radix_sort.cu` - Performance measurement tools
- `benchmark_large.cu` - Additional benchmarks for larger arrays

## Usage

### Test correctness
```bash
nvcc -std=c++17 test_radix_sort.cu radix_sort.cu -o test -O3
./test
```

### Measure performance
```bash
# Simple benchmark
./run_benchmark_simple.sh

# Or manually
nvcc -std=c++17 benchmark_radix_sort.cu radix_sort.cu -o benchmark -O3
./benchmark
```

Example output:
```bash
$ nvcc -std=c++17 benchmark_large.cu radix_sort.cu -o radix_sort_benchmark -Xcompiler "-Wall -Wextra" -O3 -arch=sm_90 --use_fast_math
$ ./radix_sort_benchmark 
GPU: NVIDIA GH200 480GB (Compute 9.0)
Memory: 94 GB, Bandwidth: 4022.8 GB/s

RADIX SORT PERFORMANCE ANALYSIS
================================
Configuration       Total Keys  Time (ms)   GKeys/sec   GB/s        Efficiency %   
--------------------------------------------------------------------------------
1x1M                1048576     42.44       0.02        0.20        0.00           
1x4M                4194304     164.23      0.03        0.20        0.01           
1x16M               16777216    717.21      0.02        0.19        0.00           
1x64M               67108864    2913.94     0.02        0.18        0.00           
64x64K              4194304     13.15       0.32        2.55        0.06           
128x32K             4194304     12.40       0.34        2.71        0.07           
256x16K             4194304     12.62       0.33        2.66        0.07           
512x8K              4194304     13.20       0.32        2.54        0.06           
1024x4K             4194304     12.67       0.33        2.65        0.07           
2048x2K             4194304     14.35       0.29        2.34        0.06           
4096x1K             4194304     18.04       0.23        1.86        0.05           
8192x512            4194304     25.45       0.16        1.32        0.03           
16384x256           4194304     40.26       0.10        0.83        0.02           
128x128K            16777216    40.69       0.41        3.30        0.08           
256x64K             16777216    42.29       0.40        3.17        0.08           
512x32K             16777216    45.47       0.37        2.95        0.07           
1024x16K            16777216    39.48       0.42        3.40        0.08           
================================================================================
PEAK PERFORMANCE: 1024x16K - 0.42 GKeys/sec
================================================================================

NOTES:
- Efficiency % = (Achieved GB/s / Theoretical Bandwidth) * 100
- Higher batch counts often achieve better GPU utilization
- Performance depends on memory access patterns and GPU occupancy
```

## Implementation Notes

- Uses MSB-flipped normalization to handle signed integers
- Processes 4 bits per radix pass (8 total passes for 32-bit integers)
- Each thread handles multiple elements when array size > block size
- Basic shared memory optimization for counting and prefix scans

This is a minimal implementation for fun - production radix sorts would include additional optimizations like better memory coalescing, multiple keys per thread, and specialized handling for different data distributions. 
