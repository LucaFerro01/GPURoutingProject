#include "packet_soa.h"
#include "routing_device.h"
#include "routing.h"
#include "bloom_filter.h"
#include "gpu_verbose.h"
#include <cuda_runtime.h>
#include <vector>
#include <set>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void forward_kernel_bloom(
    const uint32_t* dst_ip,
    uint8_t* ttl,
    uint16_t* checksum,
    int* out_if,
    const uint32_t* bloom,
    const RouteEntryDevice* rtable,
    int rtable_size,
    const uint8_t* unique_prefix_lens,
    int num_unique_lens,
    int N
);

void gpu_forward_bloom(PacketSoA& soa, const std::vector<RouteEntry>& rtable)
{
    int N = soa.N;

    // --- Build Bloom filter in pinned memory for faster transfer ---
    uint32_t* h_bloom;
    CUDA_CHECK(cudaMallocHost(&h_bloom, BLOOM_WORDS * sizeof(uint32_t)));
    
    std::vector<uint32_t> prefixes(rtable.size());
    std::vector<uint8_t> prefix_len(rtable.size());
    
    for (size_t i = 0; i < rtable.size(); i++) {
        prefixes[i] = rtable[i].prefix;
        prefix_len[i] = rtable[i].prefix_len;
    }
    
    build_bloom_filter(h_bloom, prefixes.data(), prefix_len.data(), rtable.size());

    // --- Extract unique prefix lengths and sort from longest to shortest ---
    std::set<uint8_t, std::greater<uint8_t>> unique_lens_set;
    for (size_t i = 0; i < rtable.size(); i++) {
        unique_lens_set.insert(rtable[i].prefix_len);
    }
    std::vector<uint8_t> unique_prefix_lens(unique_lens_set.begin(), unique_lens_set.end());
    
    if (g_verbose_gpu_timing) {
        std::cout << "Unique prefix lengths: " << unique_prefix_lens.size() << " (out of 33 possible)\n";
    }

    // --- Allocate device memory ---
    uint32_t* d_dst_ip;
    uint8_t*  d_ttl;
    uint16_t* d_checksum;
    int*      d_out_if;
    uint32_t* d_bloom;
    RouteEntryDevice* d_rtable;
    uint8_t* d_unique_lens;

    CUDA_CHECK(cudaMalloc(&d_dst_ip, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ttl, N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_checksum, N * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_out_if, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bloom, BLOOM_WORDS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rtable, rtable.size() * sizeof(RouteEntryDevice)));
    CUDA_CHECK(cudaMalloc(&d_unique_lens, unique_prefix_lens.size() * sizeof(uint8_t)));

    // --- Copy routing table to pinned memory ---
    RouteEntryDevice* h_rtable;
    CUDA_CHECK(cudaMallocHost(&h_rtable, rtable.size() * sizeof(RouteEntryDevice)));
    
    for (size_t i = 0; i < rtable.size(); i++) {
        h_rtable[i] = { rtable[i].prefix, rtable[i].prefix_len, rtable[i].out_if };
    }

    // --- Create CUDA streams for async operations ---
    cudaStream_t stream1, stream2, stream3;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));
    CUDA_CHECK(cudaStreamCreate(&stream3));

    // --- Timing for H2D transfer ---
    cudaEvent_t h2d_start, h2d_stop;
    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_stop));
    CUDA_CHECK(cudaEventRecord(h2d_start));

    // --- Async copy data to GPU using pinned memory ---
    if (soa.pinned_dst_ip) {
        CUDA_CHECK(cudaMemcpyAsync(d_dst_ip, soa.pinned_dst_ip, N * sizeof(uint32_t), 
                                    cudaMemcpyHostToDevice, stream1));
        CUDA_CHECK(cudaMemcpyAsync(d_ttl, soa.pinned_ttl, N * sizeof(uint8_t), 
                                    cudaMemcpyHostToDevice, stream1));
        CUDA_CHECK(cudaMemcpyAsync(d_checksum, soa.pinned_checksum, N * sizeof(uint16_t), 
                                    cudaMemcpyHostToDevice, stream2));
    } else {
        CUDA_CHECK(cudaMemcpy(d_dst_ip, soa.dst_ip.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ttl, soa.ttl.data(), N * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_checksum, soa.hdr_checksum.data(), N * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }
    
    // Async transfer of bloom filter and routing table
    CUDA_CHECK(cudaMemcpyAsync(d_bloom, h_bloom, BLOOM_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice, stream2));
    CUDA_CHECK(cudaMemcpyAsync(d_rtable, h_rtable, rtable.size() * sizeof(RouteEntryDevice), cudaMemcpyHostToDevice, stream3));
    CUDA_CHECK(cudaMemcpy(d_unique_lens, unique_prefix_lens.data(), unique_prefix_lens.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Wait for all async transfers
    if (soa.pinned_dst_ip) {
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        CUDA_CHECK(cudaStreamSynchronize(stream2));
        CUDA_CHECK(cudaStreamSynchronize(stream3));
    } else {
        CUDA_CHECK(cudaStreamSynchronize(stream2));
        CUDA_CHECK(cudaStreamSynchronize(stream3));
    }
    
    CUDA_CHECK(cudaEventRecord(h2d_stop));
    CUDA_CHECK(cudaEventSynchronize(h2d_stop));
    float h2d_ms;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));

    // --- Kernel launch ---
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    forward_kernel_bloom<<<grid, block>>>(d_dst_ip, d_ttl, d_checksum, d_out_if, 
                                          d_bloom, d_rtable, rtable.size(), 
                                          d_unique_lens, unique_prefix_lens.size(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    // --- Timing for D2H transfer ---
    cudaEvent_t d2h_start, d2h_stop;
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));
    CUDA_CHECK(cudaEventRecord(d2h_start));

    // --- Async copy back using pinned memory ---
    if (soa.pinned_out_if) {
        CUDA_CHECK(cudaMemcpyAsync(soa.pinned_ttl, d_ttl, N * sizeof(uint8_t), 
                                    cudaMemcpyDeviceToHost, stream1));
        CUDA_CHECK(cudaMemcpyAsync(soa.pinned_out_if, d_out_if, N * sizeof(int), 
                                    cudaMemcpyDeviceToHost, stream2));
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        CUDA_CHECK(cudaStreamSynchronize(stream2));
        soa.copyFromPinned();
    } else {
        CUDA_CHECK(cudaMemcpy(soa.ttl.data(), d_ttl, N * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(soa.out_if.data(), d_out_if, N * sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    CUDA_CHECK(cudaEventRecord(d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(d2h_stop));
    float d2h_ms;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop));

    // Print detailed timing
    if (g_verbose_gpu_timing) {
        std::cout << "GPU Bloom Timing (pinned=" << (soa.pinned_dst_ip ? "YES" : "NO") << "):" << std::endl;
        std::cout << "  H2D transfer: " << h2d_ms << " ms" << std::endl;
        std::cout << "  Kernel exec:  " << kernel_ms << " ms" << std::endl;
        std::cout << "  D2H transfer: " << d2h_ms << " ms" << std::endl;
        std::cout << "  Total GPU:    " << (h2d_ms + kernel_ms + d2h_ms) << " ms\n" << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));
    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_dst_ip));
    CUDA_CHECK(cudaFree(d_ttl));
    CUDA_CHECK(cudaFree(d_checksum));
    CUDA_CHECK(cudaFree(d_out_if));
    CUDA_CHECK(cudaFree(d_bloom));
    CUDA_CHECK(cudaFree(d_rtable));
    CUDA_CHECK(cudaFree(d_unique_lens));
    
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaStreamDestroy(stream3));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Free pinned memory
    cudaFreeHost(h_bloom);
    cudaFreeHost(h_rtable);
}
