#pragma once

#include <cuda_runtime.h>
#include <cstdio>

// device helper function for normalization (unsigned int result enables comparison)

/**
 * @brief normalizes an int key for unsigned integer comparison.
 * maps int values to unsigned integers by flipping the most significant bit.
 * this ensures that [INT_MIN, INT_MAX] maps to [0, UINT_MAX] when sorted as
 * unsigned integers in ascending order.
 */
__device__ __forceinline__ unsigned int normalize_key(int key_val) {
    return ((unsigned int)key_val) ^ 0x80000000u;
}


/**
 * @brief launches the CUDA radix sort for (int key, int index) pairs.
 * sorts batches of keys and their associated indices in descending order.
 *
 * @tparam BLOCK_SIZE CUDA block size (threads per block).
 * @tparam NUM_BITS_PER_PASS number of bits to sort by in each pass.
 *
 * @param h_input_keys pointer to the input keys on the host.
 * @param h_output_keys pointer to store the sorted keys on the host.
 * @param h_output_key_ids pointer to store the sorted key indices on the host.
 * @param batch_size number of batches to sort.
 * @param num_elements_per_batch number of key-index pairs in each batch.
 */
template <int BLOCK_SIZE = 256, int NUM_BITS_PER_PASS = 4>
void launch_radix_sort_int_descending(
    const int* h_input_keys,            // initial keys on host
    int* h_output_keys,                 // buffer for sorted keys on host
    int* h_output_key_ids,      // buffer for sorted key_ids on host
    int batch_size,
    int num_elements_per_batch
);

// CUDA error checking macro
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                      cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0) 
