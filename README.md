# Minimal GPU Radix Sort

A simple CUDA implementation of radix sort for integer key-value pairs. The core sorting kernel is just ~110 lines of code and demonstrates basic GPU radix sort concepts. Can easily be extended to other data types by adding few LOC normalizations to uint.

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
ubuntu@192-222-56-240:~/tmp$ ./run_benchmark.sh
Compiling radix sort benchmark...
Compilation successful!

Running benchmark...
========================================
GPU INFORMATION:
  Device: NVIDIA GH200 480GB
  Compute Capability: 9.0
  Global Memory: 94 GB
  Shared Memory per Block: 48 KB
  Max Threads per Block: 1024
  Multiprocessors: 132
  Memory Clock Rate: 2619 MHz
  Memory Bus Width: 6144 bits
  Theoretical Memory Bandwidth: 4022.8 GB/s

Starting radix sort benchmark...
Block size: 256, Bits per pass: 4
Data type: int32 keys with int32 indices
Sort order: Descending

  Running Small-1K... Done
  Running Small-4K... Done
  Running Small-16K... Done
  Running Small-64K... Done
  Running Medium-256K... Done
  Running Medium-1M... Done
  Running Medium-4M... Done
  Running Large-16M... Done
  Running Large-64M... Done
  Running Batch4x256K... Done
  Running Batch8x128K... Done
  Running Batch16x64K... Done
  Running Batch32x32K... Done
  Running Batch64x16K... Done
  Running Batch128x8K... Done
  Running Batch256x4K... Done
  Running Batch512x2K... Done
  Running Batch1024x16K... Done

========================================================================================================================
RADIX SORT BENCHMARK RESULTS
========================================================================================================================
Configuration            Batch Size  Array Size  Total Keys  Avg Time(ms)GKeys/sec   GB/s        Std Dev(ms) 
------------------------------------------------------------------------------------------------------------------------
Small-1K                 1           1024        1024        1.74        0.00        0.00        0.16        
Small-4K                 1           4096        4096        1.79        0.00        0.02        0.03        
Small-16K                1           16384       16384       2.25        0.01        0.06        0.02        
Small-64K                1           65536       65536       3.94        0.02        0.13        0.02        
Medium-256K              1           262144      262144      12.29       0.02        0.17        0.64        
Medium-1M                1           1048576     1048576     42.55       0.02        0.20        0.23        
Medium-4M                1           4194304     4194304     164.48      0.03        0.20        0.74        
Large-16M                1           16777216    16777216    713.69      0.02        0.19        0.19        
Large-64M                1           67108864    67108864    2917.16     0.02        0.18        0.49        
Batch4x256K              4           262144      1048576     15.56       0.07        0.54        0.09        
Batch8x128K              8           131072      1048576     11.00       0.10        0.76        0.03        
Batch16x64K              16          65536       1048576     8.83        0.12        0.95        0.25        
Batch32x32K              32          32768       1048576     7.88        0.13        1.07        0.88        
Batch64x16K              64          16384       1048576     7.03        0.15        1.19        0.05        
Batch128x8K              128         8192        1048576     6.83        0.15        1.23        0.12        
Batch256x4K              256         4096        1048576     6.97        0.15        1.20        0.03        
Batch512x2K              512         2048        1048576     7.31        0.14        1.15        0.04        
Batch1024x16K            1024        16384       16777216    40.02       0.42        3.35        1.06        
========================================================================================================================

BEST PERFORMANCE: Batch1024x16K - 0.42 GKeys/sec (3.35 GB/s)

NOTES:
- GKeys/sec = Billion key-value pairs sorted per second
- GB/s = Gigabytes per second throughput (8 bytes per key-value pair)
- Times are averaged over multiple runs with warmup
```

## Implementation Notes

- Uses MSB-flipped normalization to handle signed integers
- Processes 4 bits per radix pass (8 total passes for 32-bit integers)
- Each thread handles multiple elements when array size > block size
- Basic shared memory optimization for counting and prefix scans

This is a minimal implementation for fun - production radix sorts would include additional optimizations like better memory coalescing, multiple keys per thread, and specialized handling for different data distributions. 