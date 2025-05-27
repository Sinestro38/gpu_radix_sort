#include "radix_sort.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>

// CUDA timing utilities
class CudaTimer {
private:
    cudaEvent_t start_event, stop_event;
    
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_event));
    }
    
    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        return milliseconds;
    }
};

// Generate random test data
void generate_test_data(std::vector<int>& keys, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dis(std::numeric_limits<int>::min(), 
                                          std::numeric_limits<int>::max());
    
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = dis(gen);
    }
}

// Run benchmark for a specific configuration
double benchmark_configuration(int batch_size, int num_elements_per_batch, int num_runs = 10) {
    size_t total_elements = (size_t)batch_size * num_elements_per_batch;
    std::vector<int> h_input_keys(total_elements);
    std::vector<int> h_output_keys(total_elements);
    std::vector<int> h_output_key_ids(total_elements);
    
    // Generate test data
    generate_test_data(h_input_keys);
    
    CudaTimer timer;
    std::vector<double> times;
    times.reserve(num_runs);
    
    // Warmup runs
    for (int i = 0; i < 3; ++i) {
        launch_radix_sort_int_descending<256, 4>(
            h_input_keys.data(),
            h_output_keys.data(),
            h_output_key_ids.data(),
            batch_size,
            num_elements_per_batch
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Benchmark runs
    for (int i = 0; i < num_runs; ++i) {
        timer.start();
        
        launch_radix_sort_int_descending<256, 4>(
            h_input_keys.data(),
            h_output_keys.data(),
            h_output_key_ids.data(),
            batch_size,
            num_elements_per_batch
        );
        
        float time_ms = timer.stop();
        times.push_back(time_ms);
    }
    
    // Return minimum time (best performance)
    return *std::min_element(times.begin(), times.end());
}

// Print GPU information
void print_gpu_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "GPU: " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")" << std::endl;
    std::cout << "Memory: " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB, ";
    std::cout << "Bandwidth: " << std::fixed << std::setprecision(1) 
              << (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6) << " GB/s" << std::endl;
    std::cout << std::endl;
}

int main() {
    try {
        print_gpu_info();
        
        std::cout << "RADIX SORT PERFORMANCE ANALYSIS" << std::endl;
        std::cout << "================================" << std::endl;
        std::cout << std::left 
                  << std::setw(20) << "Configuration"
                  << std::setw(12) << "Total Keys"
                  << std::setw(12) << "Time (ms)"
                  << std::setw(12) << "GKeys/sec"
                  << std::setw(12) << "GB/s"
                  << std::setw(15) << "Efficiency %"
                  << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        // Test configurations focusing on finding peak performance
        struct TestConfig {
            int batch_size;
            int array_size;
            std::string name;
        };
        
        std::vector<TestConfig> configs = {
            // Large single arrays
            {1, 1048576, "1x1M"},
            {1, 4194304, "1x4M"},
            {1, 16777216, "1x16M"},
            {1, 67108864, "1x64M"},
            
            // Optimal batch sizes (based on initial results)
            {64, 65536, "64x64K"},
            {128, 32768, "128x32K"},
            {256, 16384, "256x16K"},
            {512, 8192, "512x8K"},
            {1024, 4096, "1024x4K"},
            {2048, 2048, "2048x2K"},
            
            // Very high batch counts
            {4096, 1024, "4096x1K"},
            {8192, 512, "8192x512"},
            {16384, 256, "16384x256"},
            
            // Large total throughput tests
            {128, 131072, "128x128K"},
            {256, 65536, "256x64K"},
            {512, 32768, "512x32K"},
            {1024, 16384, "1024x16K"},
        };
        
        double best_gkeys_per_sec = 0.0;
        std::string best_config;
        
        // Get theoretical memory bandwidth for efficiency calculation
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        double theoretical_bandwidth = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
        
        for (const auto& config : configs) {
            try {
                size_t total_keys = (size_t)config.batch_size * config.array_size;
                
                // Skip configurations that might be too large for memory
                size_t memory_needed = total_keys * 4 * 4; // 4 buffers * 4 bytes per int
                if (memory_needed > prop.totalGlobalMem * 0.8) {
                    std::cout << std::left << std::setw(20) << config.name
                              << std::setw(12) << total_keys
                              << "SKIPPED (too large)" << std::endl;
                    continue;
                }
                
                double time_ms = benchmark_configuration(config.batch_size, config.array_size);
                double gkeys_per_sec = (total_keys / 1e9) / (time_ms / 1000.0);
                double gb_per_sec = (total_keys * 8.0 / 1e9) / (time_ms / 1000.0);
                double efficiency = (gb_per_sec / theoretical_bandwidth) * 100.0;
                
                std::cout << std::left << std::fixed << std::setprecision(2)
                          << std::setw(20) << config.name
                          << std::setw(12) << total_keys
                          << std::setw(12) << time_ms
                          << std::setw(12) << gkeys_per_sec
                          << std::setw(12) << gb_per_sec
                          << std::setw(15) << efficiency
                          << std::endl;
                
                if (gkeys_per_sec > best_gkeys_per_sec) {
                    best_gkeys_per_sec = gkeys_per_sec;
                    best_config = config.name;
                }
                
            } catch (const std::exception& e) {
                std::cout << std::left << std::setw(20) << config.name
                          << "ERROR: " << e.what() << std::endl;
            }
        }
        
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "PEAK PERFORMANCE: " << best_config 
                  << " - " << std::fixed << std::setprecision(2) 
                  << best_gkeys_per_sec << " GKeys/sec" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        std::cout << "\nNOTES:" << std::endl;
        std::cout << "- Efficiency % = (Achieved GB/s / Theoretical Bandwidth) * 100" << std::endl;
        std::cout << "- Higher batch counts often achieve better GPU utilization" << std::endl;
        std::cout << "- Performance depends on memory access patterns and GPU occupancy" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 