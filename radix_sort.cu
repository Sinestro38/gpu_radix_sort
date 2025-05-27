#include "radix_sort.h"
#include <cuda_runtime.h>
#include <stdexcept> // for std::runtime_error
#include <string>    // for std::to_string

// Kernel to perform one pass of radix sort on (int key, int index) pairs
// sorts in descending order of original key values.
// each thread can process multiple elements when num_elements_per_batch > BLOCK_SIZE
template <int CURRENT_BIT, int BLOCK_SIZE, int NUM_BITS_PER_PASS>
__global__ void radix_sort_pairs_kernel_int_desc(
    const int* __restrict__ d_input_keys,
    int* __restrict__ d_output_keys,
    const int* __restrict__ d_input_key_ids, // type is int
    int* __restrict__ d_output_key_ids,   // type is int
    int num_elements_per_batch)
{
    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int elems_per_thread = (num_elements_per_batch + BLOCK_SIZE - 1) / BLOCK_SIZE; // ceildiv

    constexpr int NUM_BUCKETS = 1 << NUM_BITS_PER_PASS;
    constexpr bool ASCENDING_NORMALIZED = false; // sort normalized keys descending for original keys descending

    // adjust pointers for the current batch
    const int* current_batch_input_keys = d_input_keys + batch_id * num_elements_per_batch;
    int* current_batch_output_keys = d_output_keys + batch_id * num_elements_per_batch;
    const int* current_batch_input_key_ids = d_input_key_ids + batch_id * num_elements_per_batch;
    int* current_batch_output_key_ids = d_output_key_ids + batch_id * num_elements_per_batch;

    // shared memory declarations
    extern __shared__ int s_mem[];
    int* s_counts = s_mem; // BLOCK_SIZE * NUM_BUCKETS
    int* total_counts = (int*)&s_counts[BLOCK_SIZE * NUM_BUCKETS]; // NUM_BUCKETS
    int* total_offsets = (int*)&total_counts[NUM_BUCKETS]; // NUM_BUCKETS + 1
    int* s_thread_offsets = (int*)&total_offsets[NUM_BUCKETS + 1]; // BLOCK_SIZE * NUM_BUCKETS

    // 1. initialize local counts for this thread
    int local_counts[NUM_BUCKETS];
    for (int i = 0; i < NUM_BUCKETS; ++i) {
        local_counts[i] = 0;
    }

    // 2. process elements and compute local counts
    // each thread processes elems_per_thread elements in a contiguous chunk
    for (int i = 0; i < elems_per_thread; ++i) {
        int element_idx = tid * elems_per_thread + i;
        if (element_idx < num_elements_per_batch) {
            int key = current_batch_input_keys[element_idx];
            unsigned int normalized_key = normalize_key(key);
            unsigned int radix = (normalized_key >> CURRENT_BIT) & (NUM_BUCKETS - 1);
            local_counts[radix]++;
        }
    }

    // 3. store local_counts into shared memory s_counts
    for (int i = 0; i < NUM_BUCKETS; ++i) {
        s_counts[tid * NUM_BUCKETS + i] = local_counts[i];
    }
    __syncthreads();

    // 4. compute total_counts by summing s_counts across threads
    if (tid < NUM_BUCKETS) {
        int sum = 0;
        for (int t = 0; t < BLOCK_SIZE; ++t) {
            sum += s_counts[t * NUM_BUCKETS + tid]; // tid here is bucket_id
        }
        total_counts[tid] = sum;
    }
    __syncthreads();

    // 5. perform exclusive scan over total_counts to get total_offsets
    if (tid == 0) {
        total_offsets[0] = 0;
        for (int i = 1; i < NUM_BUCKETS + 1; ++i) {
            total_offsets[i] = total_offsets[i - 1] + total_counts[i - 1];
        }
    }
    __syncthreads();

    // 6. compute per-thread starting offsets (s_thread_offsets)
    for (int radix = 0; radix < NUM_BUCKETS; ++radix) {
        int thread_count_for_radix = s_counts[tid * NUM_BUCKETS + radix];
        s_thread_offsets[tid * NUM_BUCKETS + radix] = thread_count_for_radix;
        __syncthreads();

        // parallel prefix scan (similar to Mojo implementation)
        for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
            int val = 0;
            if (tid >= offset) {
                val = s_thread_offsets[(tid - offset) * NUM_BUCKETS + radix];
            }
            __syncthreads();
            s_thread_offsets[tid * NUM_BUCKETS + radix] += val;
            __syncthreads();
        }
        
        // convert to exclusive scan
        int exclusive_prefix_sum = (tid == 0) ? 0 : s_thread_offsets[(tid - 1) * NUM_BUCKETS + radix];
        __syncthreads();
        s_thread_offsets[tid * NUM_BUCKETS + radix] = exclusive_prefix_sum;
        __syncthreads();
    }

    // 7. scatter elements to output
    // reset local_counts to use as local_offsets for each thread
    for (int i = 0; i < NUM_BUCKETS; ++i) {
        local_counts[i] = 0; // re-using as local_offsets
    }

    // process elements again and scatter to output
    for (int i = 0; i < elems_per_thread; ++i) {
        int element_idx = tid * elems_per_thread + i;
        if (element_idx < num_elements_per_batch) {
            int key = current_batch_input_keys[element_idx];
            int key_id_val = current_batch_input_key_ids[element_idx];
            int current_pass_key_id = (CURRENT_BIT == 0) ? element_idx : key_id_val;

            unsigned int normalized_key = normalize_key(key);
            unsigned int radix = (normalized_key >> CURRENT_BIT) & (NUM_BUCKETS - 1);

            int global_offset_base;
            if (ASCENDING_NORMALIZED) { // ascending sort of normalized keys
                global_offset_base = total_offsets[radix];
            } else { // descending sort of normalized keys
                global_offset_base = total_offsets[NUM_BUCKETS] - total_offsets[radix + 1];
            }

            int thread_specific_offset_in_bucket = s_thread_offsets[tid * NUM_BUCKETS + radix];
            int local_offset_within_thread_bucket = local_counts[radix];

            int global_offset = global_offset_base + thread_specific_offset_in_bucket + local_offset_within_thread_bucket;

            current_batch_output_keys[global_offset] = key;
            current_batch_output_key_ids[global_offset] = current_pass_key_id;

            local_counts[radix]++;
        }
    }
}


// host launcher function for int keys, descending order
template <int BLOCK_SIZE, int NUM_BITS_PER_PASS>
void launch_radix_sort_int_descending(
    const int* h_input_keys,
    int* h_output_keys,
    int* h_output_key_ids,
    int batch_size,
    int num_elements_per_batch)
{
    size_t total_elements = (size_t)batch_size * num_elements_per_batch;
    if (total_elements == 0) return;

    int* d_keys_ping;
    int* d_keys_pong;
    int* d_key_ids_ping; // indices are int
    int* d_key_ids_pong; // indices are int

    CUDA_CHECK(cudaMalloc(&d_keys_ping, total_elements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_keys_pong, total_elements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_key_ids_ping, total_elements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_key_ids_pong, total_elements * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_keys_ping, h_input_keys, total_elements * sizeof(int), cudaMemcpyHostToDevice));
    // d_key_ids_ping will be effectively initialized by the kernel in the first pass (CURRENT_BIT == 0)
    // to be the original index (0 to N-1).

    dim3 grid_dim(batch_size);
    dim3 block_dim(BLOCK_SIZE);
    
    constexpr int num_total_bits = sizeof(int) * 8;
    int num_passes = 0;

    constexpr int NUM_BUCKETS = 1 << NUM_BITS_PER_PASS;
    size_t shmem_size = (2 * BLOCK_SIZE * NUM_BUCKETS + 2 * NUM_BUCKETS + 1) * sizeof(int);

    for (int current_bit_val = 0; current_bit_val < num_total_bits; current_bit_val += NUM_BITS_PER_PASS) {
        #define LAUNCH_KERNEL_FOR_BIT_INT_DESC(BIT) \
            radix_sort_pairs_kernel_int_desc<BIT, BLOCK_SIZE, NUM_BITS_PER_PASS><<<grid_dim, block_dim, shmem_size>>>( \
                d_keys_ping, d_keys_pong, d_key_ids_ping, d_key_ids_pong, num_elements_per_batch)

        // dispatch based on current_bit_val for int (32-bit)
        if (current_bit_val == 0) LAUNCH_KERNEL_FOR_BIT_INT_DESC(0);
        else if (current_bit_val == 4) LAUNCH_KERNEL_FOR_BIT_INT_DESC(4);
        else if (current_bit_val == 8) LAUNCH_KERNEL_FOR_BIT_INT_DESC(8);
        else if (current_bit_val == 12) LAUNCH_KERNEL_FOR_BIT_INT_DESC(12);
        else if (current_bit_val == 16) LAUNCH_KERNEL_FOR_BIT_INT_DESC(16);
        else if (current_bit_val == 20) LAUNCH_KERNEL_FOR_BIT_INT_DESC(20);
        else if (current_bit_val == 24) LAUNCH_KERNEL_FOR_BIT_INT_DESC(24);
        else if (current_bit_val == 28) LAUNCH_KERNEL_FOR_BIT_INT_DESC(28);
        else {
             cudaDeviceSynchronize();
             std::string err_msg = "Unsupported current_bit value: " + std::to_string(current_bit_val) + " for fixed dispatcher (int).";
             CUDA_CHECK(cudaGetLastError());
             throw std::runtime_error(err_msg);
        }
        #undef LAUNCH_KERNEL_FOR_BIT_INT_DESC
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::swap(d_keys_ping, d_keys_pong);
        std::swap(d_key_ids_ping, d_key_ids_pong);
        num_passes++;
    }

    CUDA_CHECK(cudaMemcpy(h_output_keys, d_keys_ping, total_elements * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_key_ids, d_key_ids_ping, total_elements * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_keys_ping));
    CUDA_CHECK(cudaFree(d_keys_pong));
    CUDA_CHECK(cudaFree(d_key_ids_ping));
    CUDA_CHECK(cudaFree(d_key_ids_pong));
}

// explicit template instantiation for int keys, int indices, descending sort.
// default BLOCK_SIZE=256, NUM_BITS_PER_PASS=4
template void launch_radix_sort_int_descending<256, 4>(
    const int*, int*, int*, int, int); 