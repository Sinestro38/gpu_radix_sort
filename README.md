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

## Implementation Notes

- Uses MSB-flipped normalization to handle signed integers
- Processes 4 bits per radix pass (8 total passes for 32-bit integers)
- Each thread handles multiple elements when array size > block size
- Basic shared memory optimization for counting and prefix scans

This is a minimal implementation for fun - production radix sorts would include additional optimizations like better memory coalescing, multiple keys per thread, and specialized handling for different data distributions. 