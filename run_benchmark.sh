#!/bin/bash

echo "Compiling radix sort benchmark..."

# Compile the benchmark
nvcc -std=c++17 benchmark_radix_sort.cu radix_sort.cu -o radix_sort_benchmark \
     -Xcompiler "-Wall -Wextra" -O3 -arch=sm_90 --use_fast_math

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo ""
    echo "Running benchmark..."
    echo "========================================"
    ./radix_sort_benchmark
else
    echo "Compilation failed!"
    exit 1
fi 