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

// Benchmark configuration
struct BenchmarkConfig {
    int batch_size;
    int num_elements_per_batch;
    int num_warmup_runs;
    int num_benchmark_runs;
    std::string description;
    
    BenchmarkConfig(int bs, int ne, int warmup, int bench, const std::string& desc)
        : batch_size(bs), num_elements_per_batch(ne), num_warmup_runs(warmup), 
          num_benchmark_runs(bench), description(desc) {}
};

// Benchmark results
struct BenchmarkResult {
    BenchmarkConfig config;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double std_dev_ms;
    double gkeys_per_sec;
    double throughput_gb_per_sec;
    
    BenchmarkResult(const BenchmarkConfig& cfg) : config(cfg) {}
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
BenchmarkResult run_benchmark(const BenchmarkConfig& config) {
    BenchmarkResult result(config);
    
    size_t total_elements = (size_t)config.batch_size * config.num_elements_per_batch;
    std::vector<int> h_input_keys(total_elements);
    std::vector<int> h_output_keys(total_elements);
    std::vector<int> h_output_key_ids(total_elements);
    
    // Generate test data
    generate_test_data(h_input_keys);
    
    CudaTimer timer;
    std::vector<double> times;
    times.reserve(config.num_benchmark_runs);
    
    std::cout << "  Running " << config.description << "..." << std::flush;
    
    // Warmup runs
    for (int i = 0; i < config.num_warmup_runs; ++i) {
        launch_radix_sort_int_descending<256, 4>(
            h_input_keys.data(),
            h_output_keys.data(),
            h_output_key_ids.data(),
            config.batch_size,
            config.num_elements_per_batch
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Benchmark runs
    for (int i = 0; i < config.num_benchmark_runs; ++i) {
        timer.start();
        
        launch_radix_sort_int_descending<256, 4>(
            h_input_keys.data(),
            h_output_keys.data(),
            h_output_key_ids.data(),
            config.batch_size,
            config.num_elements_per_batch
        );
        
        float time_ms = timer.stop();
        times.push_back(time_ms);
    }
    
    // Calculate statistics
    result.min_time_ms = *std::min_element(times.begin(), times.end());
    result.max_time_ms = *std::max_element(times.begin(), times.end());
    
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    result.avg_time_ms = sum / times.size();
    
    // Calculate standard deviation
    double variance = 0.0;
    for (double t : times) {
        variance += (t - result.avg_time_ms) * (t - result.avg_time_ms);
    }
    result.std_dev_ms = std::sqrt(variance / times.size());
    
    // Calculate throughput metrics
    double total_keys = static_cast<double>(total_elements);
    result.gkeys_per_sec = (total_keys / 1e9) / (result.avg_time_ms / 1000.0);
    
    // Throughput in GB/s (assuming 8 bytes per key-value pair: 4 bytes key + 4 bytes index)
    double total_bytes = total_keys * 8.0; // 4 bytes key + 4 bytes index
    result.throughput_gb_per_sec = (total_bytes / 1e9) / (result.avg_time_ms / 1000.0);
    
    std::cout << " Done" << std::endl;
    
    return result;
}

// Print benchmark results
void print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(120, '=') << std::endl;
    std::cout << "RADIX SORT BENCHMARK RESULTS" << std::endl;
    std::cout << std::string(120, '=') << std::endl;
    
    // Print header
    std::cout << std::left 
              << std::setw(25) << "Configuration"
              << std::setw(12) << "Batch Size"
              << std::setw(12) << "Array Size"
              << std::setw(12) << "Total Keys"
              << std::setw(12) << "Avg Time(ms)"
              << std::setw(12) << "GKeys/sec"
              << std::setw(12) << "GB/s"
              << std::setw(12) << "Std Dev(ms)"
              << std::endl;
    
    std::cout << std::string(120, '-') << std::endl;
    
    // Print results
    for (const auto& result : results) {
        size_t total_keys = (size_t)result.config.batch_size * result.config.num_elements_per_batch;
        
        std::cout << std::left << std::fixed << std::setprecision(2)
                  << std::setw(25) << result.config.description
                  << std::setw(12) << result.config.batch_size
                  << std::setw(12) << result.config.num_elements_per_batch
                  << std::setw(12) << total_keys
                  << std::setw(12) << result.avg_time_ms
                  << std::setw(12) << result.gkeys_per_sec
                  << std::setw(12) << result.throughput_gb_per_sec
                  << std::setw(12) << result.std_dev_ms
                  << std::endl;
    }
    
    std::cout << std::string(120, '=') << std::endl;
    
    // Find best performance
    auto best_gkeys = std::max_element(results.begin(), results.end(),
        [](const BenchmarkResult& a, const BenchmarkResult& b) {
            return a.gkeys_per_sec < b.gkeys_per_sec;
        });
    
    if (best_gkeys != results.end()) {
        std::cout << "\nBEST PERFORMANCE: " << best_gkeys->config.description 
                  << " - " << std::fixed << std::setprecision(2) 
                  << best_gkeys->gkeys_per_sec << " GKeys/sec ("
                  << best_gkeys->throughput_gb_per_sec << " GB/s)" << std::endl;
    }
}

// Print GPU information
void print_gpu_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "GPU INFORMATION:" << std::endl;
    std::cout << "  Device: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Global Memory: " << (prop.totalGlobalMem / (1024*1024*1024)) << " GB" << std::endl;
    std::cout << "  Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB" << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Memory Clock Rate: " << (prop.memoryClockRate / 1000) << " MHz" << std::endl;
    std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    
    // Calculate theoretical memory bandwidth
    double bandwidth_gb_s = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6;
    std::cout << "  Theoretical Memory Bandwidth: " << std::fixed << std::setprecision(1) 
              << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << std::endl;
}

int main() {
    try {
        print_gpu_info();
        
        // Define benchmark configurations
        std::vector<BenchmarkConfig> configs = {
            // Small arrays (single batch)
            BenchmarkConfig(1, 1024, 5, 20, "Small-1K"),
            BenchmarkConfig(1, 4096, 5, 20, "Small-4K"),
            BenchmarkConfig(1, 16384, 5, 20, "Small-16K"),
            BenchmarkConfig(1, 65536, 5, 20, "Small-64K"),
            
            // Medium arrays (single batch)
            BenchmarkConfig(1, 262144, 3, 15, "Medium-256K"),
            BenchmarkConfig(1, 1048576, 3, 15, "Medium-1M"),
            BenchmarkConfig(1, 4194304, 3, 10, "Medium-4M"),
            
            // Large arrays (single batch)
            BenchmarkConfig(1, 16777216, 2, 10, "Large-16M"),
            BenchmarkConfig(1, 67108864, 2, 5, "Large-64M"),
            
            // Batched configurations
            BenchmarkConfig(4, 262144, 3, 15, "Batch4x256K"),
            BenchmarkConfig(8, 131072, 3, 15, "Batch8x128K"),
            BenchmarkConfig(16, 65536, 3, 15, "Batch16x64K"),
            BenchmarkConfig(32, 32768, 3, 15, "Batch32x32K"),
            BenchmarkConfig(64, 16384, 3, 15, "Batch64x16K"),
            
            // High batch count
            BenchmarkConfig(128, 8192, 3, 10, "Batch128x8K"),
            BenchmarkConfig(256, 4096, 3, 10, "Batch256x4K"),
            BenchmarkConfig(512, 2048, 3, 10, "Batch512x2K"),
            BenchmarkConfig(1024, 16384, 3, 10, "Batch1024x16K"),
        };
        
        std::vector<BenchmarkResult> results;
        results.reserve(configs.size());
        
        std::cout << "Starting radix sort benchmark..." << std::endl;
        std::cout << "Block size: 256, Bits per pass: 4" << std::endl;
        std::cout << "Data type: int32 keys with int32 indices" << std::endl;
        std::cout << "Sort order: Descending" << std::endl << std::endl;
        
        // Run benchmarks
        for (const auto& config : configs) {
            try {
                BenchmarkResult result = run_benchmark(config);
                results.push_back(result);
            } catch (const std::exception& e) {
                std::cerr << "  Error in " << config.description << ": " << e.what() << std::endl;
            }
        }
        
        // Print results
        print_results(results);
        
        // Additional analysis
        std::cout << "\nNOTES:" << std::endl;
        std::cout << "- GKeys/sec = Billion key-value pairs sorted per second" << std::endl;
        std::cout << "- GB/s = Gigabytes per second throughput (8 bytes per key-value pair)" << std::endl;
        std::cout << "- Times are averaged over multiple runs with warmup" << std::endl;
        std::cout << "- Performance may vary based on data distribution and GPU utilization" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 