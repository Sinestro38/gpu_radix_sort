#include "radix_sort.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip> // For std::fixed and std::setprecision
#include <limits>  // For std::numeric_limits
#include <stdexcept> // For std::runtime_error

// Helper to print arrays for debugging
template <typename T>
void print_array(const std::string& name, const T* arr, int size, int batch_id = -1, int elements_per_batch = -1) {
    std::cout << name;
    if (batch_id != -1) {
        std::cout << " [Batch " << batch_id << "]";
    }
    std::cout << ": [ ";
    int start = 0;
    int end = size;
    if (batch_id != -1 && elements_per_batch != -1) {
        start = batch_id * elements_per_batch;
        end = start + elements_per_batch;
    }
    for (int i = start; i < end; ++i) {
        std::cout << arr[i];
        if (i < end - 1) std::cout << ", ";
    }
    std::cout << " ]" << std::endl;
}

// For comparing pairs in CPU sort (int keys, descending order)
struct IntKeyValuePair {
    int key;
    int id;
    unsigned int normalized_key; // MSB-flipped key for comparison

    IntKeyValuePair(int k, int i) : key(k), id(i) {
        normalized_key = ((unsigned int)k) ^ 0x80000000u;
    }
};

bool compareIntKeyValuePairsDesc(const IntKeyValuePair& a, const IntKeyValuePair& b) {
    // For descending sort of original keys, we sort normalized keys descendingly.
    if (a.normalized_key != b.normalized_key) {
        return a.normalized_key > b.normalized_key;
    }
    // Stability: if normalized keys are equal, sort by original index (smaller id first)
    return a.id < b.id;
}

void test_simple_case() {
    std::cout << "\n--- Testing Simple Case: 1x8 ---" << std::endl;
    
    // Simple test case: 8 elements, 1 batch
    std::vector<int> h_input_keys = {5, 2, 8, 1, 9, 3, 7, 4};
    std::vector<int> h_output_keys_gpu(8);
    std::vector<int> h_output_key_ids_gpu(8);
    
    std::cout << "Input keys: ";
    for (int i = 0; i < 8; i++) {
        std::cout << h_input_keys[i] << " ";
    }
    std::cout << std::endl;
    
    // Show normalized keys for debugging
    std::cout << "Normalized keys: ";
    for (int i = 0; i < 8; i++) {
        unsigned int norm = ((unsigned int)h_input_keys[i]) ^ 0x80000000u;
        std::cout << std::hex << norm << " ";
    }
    std::cout << std::dec << std::endl;
    
    // GPU Radix Sort
    launch_radix_sort_int_descending<256, 4>(
        h_input_keys.data(),
        h_output_keys_gpu.data(),
        h_output_key_ids_gpu.data(),
        1, // batch_size
        8  // num_elements_per_batch
    );
    
    std::cout << "GPU output keys: ";
    for (int i = 0; i < 8; i++) {
        std::cout << h_output_keys_gpu[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "GPU output IDs: ";
    for (int i = 0; i < 8; i++) {
        std::cout << h_output_key_ids_gpu[i] << " ";
    }
    std::cout << std::endl;
    
    // CPU sort for comparison
    std::vector<IntKeyValuePair> pairs;
    for (int i = 0; i < 8; i++) {
        pairs.emplace_back(h_input_keys[i], i);
    }
    std::stable_sort(pairs.begin(), pairs.end(), compareIntKeyValuePairsDesc);
    
    std::cout << "CPU expected keys: ";
    for (int i = 0; i < 8; i++) {
        std::cout << pairs[i].key << " ";
    }
    std::cout << std::endl;
    
    std::cout << "CPU expected IDs: ";
    for (int i = 0; i < 8; i++) {
        std::cout << pairs[i].id << " ";
    }
    std::cout << std::endl;
}

void test_radix_sort_int_descending(
    int batch_size,
    int num_elements_per_batch)
{
    std::cout << "\n--- Testing Radix Sort for Type: int, IndexType: int, Order: Descending"
              << ", Batch: " << batch_size << "x" << num_elements_per_batch
              << " ---" << std::endl;

    size_t total_elements = (size_t)batch_size * num_elements_per_batch;
    std::vector<int> h_input_keys(total_elements);
    // h_input_key_ids is not sent to GPU, kernel generates 0..N-1 for first pass.
    // But we need it for CPU verification.
    std::vector<int> h_original_ids_for_cpu(total_elements);
    std::vector<int> h_output_keys_gpu(total_elements);
    std::vector<int> h_output_key_ids_gpu(total_elements);

    std::vector<int> h_expected_keys(total_elements);
    std::vector<int> h_expected_key_ids(total_elements);

    // Initialize input data
    std::mt19937 gen(123); // Fixed seed
    std::uniform_int_distribution<int> dis(-10000, 10000);
    for (size_t i = 0; i < total_elements; ++i) {
        h_input_keys[i] = dis(gen);
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < num_elements_per_batch; ++i) {
            h_original_ids_for_cpu[b * num_elements_per_batch + i] = i;
        }
    }

    // print_array("Host Input Keys (Initial)", h_input_keys.data(), total_elements);

    // GPU Radix Sort (descending for int)
    launch_radix_sort_int_descending<
        256, // BLOCK_SIZE
        4    // NUM_BITS_PER_PASS
    >(
        h_input_keys.data(),
        h_output_keys_gpu.data(),
        h_output_key_ids_gpu.data(),
        batch_size,
        num_elements_per_batch
    );

    // CPU Sort for verification (per batch, descending)
    for (int b = 0; b < batch_size; ++b) {
        int start_idx = b * num_elements_per_batch;

        std::vector<IntKeyValuePair> batch_pairs;
        for (int i = 0; i < num_elements_per_batch; ++i) {
            batch_pairs.emplace_back(h_input_keys[start_idx + i], h_original_ids_for_cpu[start_idx + i]);
        }

        std::stable_sort(batch_pairs.begin(), batch_pairs.end(), 
                         compareIntKeyValuePairsDesc); 

        for (int i = 0; i < num_elements_per_batch; ++i) {
            h_expected_keys[start_idx + i] = batch_pairs[i].key;
            h_expected_key_ids[start_idx + i] = batch_pairs[i].id;
        }
    }

    // Verify results
    bool success = true;
    for (int b = 0; b < batch_size; ++b) {
      bool batch_success = true;
        for (int i = 0; i < num_elements_per_batch; ++i) {
            size_t idx = (size_t)b * num_elements_per_batch + i;
            if (h_expected_keys[idx] != h_output_keys_gpu[idx] || h_expected_key_ids[idx] != h_output_key_ids_gpu[idx]) {
                if(batch_success) { // Print headers only once per failed batch
                    std::cerr << "Verification FAILED for Batch " << b << std::endl;
                    print_array("GPU Output Keys", h_output_keys_gpu.data(), num_elements_per_batch, b, num_elements_per_batch);
                    print_array("GPU Output IDs", h_output_key_ids_gpu.data(), num_elements_per_batch, b, num_elements_per_batch);
                    print_array("CPU Expected Keys", h_expected_keys.data(), num_elements_per_batch, b, num_elements_per_batch);
                    print_array("CPU Expected IDs", h_expected_key_ids.data(), num_elements_per_batch, b, num_elements_per_batch);
                }
                std::cerr << "  Mismatch at index " << i << " (global " << idx << "): "
                          << "GPU Key: " << h_output_keys_gpu[idx] << " (ID: " << h_output_key_ids_gpu[idx] << "), "
                          << "CPU Key: " << h_expected_keys[idx] << " (ID: " << h_expected_key_ids[idx] << ")"
                          << std::endl;
                success = false;
                batch_success = false;
            }
        }
    }

    if (success) {
        std::cout << "Verification PASSED!" << std::endl;
    } else {
        std::cout << "Verification FAILED!" << std::endl;
    }
}

int main() {
    try {
        test_simple_case();  // Start with simple case
        test_radix_sort_int_descending(1, 16);   // Small test
        test_radix_sort_int_descending(1, 32);   // 32 elements
        test_radix_sort_int_descending(1, 64);   // 64 elements  
        test_radix_sort_int_descending(1, 128);  // 128 elements
        test_radix_sort_int_descending(1, 256);  // Equal to block size
        test_radix_sort_int_descending(1, 512);  // Larger than block size
        test_radix_sort_int_descending(1, 1024); // Much larger than block size
        test_radix_sort_int_descending(4, 1024); // Multiple batches with large arrays
        
        std::cout << "\nAll integer descending sort tests completed successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 