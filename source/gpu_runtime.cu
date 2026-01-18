#include "packet_soa.h"
#include "routing_device.h"
#include "routing.h"
#include <cuda_runtime.h>
#include <vector>
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

__global__ void forward_kernel(
    const uint32_t* dst_ip,
    uint8_t* ttl,
    uint16_t* checksum,
    int* out_if,
    const RouteEntryDevice* rtable,
    int rtable_size,
    int N
);

void gpu_forward(PacketSoA& soa, const std::vector<RouteEntry>& rtable)
{
    int N = soa.N;

    // --- Allocate device memory ---
    uint32_t* d_dst_ip;
    uint8_t*  d_ttl;
    uint16_t* d_checksum;
    int*      d_out_if;
    RouteEntryDevice* d_rtable;

    CUDA_CHECK(cudaMalloc(&d_dst_ip, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_ttl, N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_checksum, N * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_out_if, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rtable, rtable.size() * sizeof(RouteEntryDevice)));

    // --- Copy routing table ---
    std::vector<RouteEntryDevice> rdev(rtable.size());
    for (size_t i = 0; i < rtable.size(); i++) {
        rdev[i] = { rtable[i].prefix, rtable[i].prefix_len, rtable[i].out_if };
    }

    // --- Create CUDA streams for async operations ---
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    // --- Async copy from pinned memory to GPU (faster!) ---
    if (soa.pinned_dst_ip) {
        CUDA_CHECK(cudaMemcpyAsync(d_dst_ip, soa.pinned_dst_ip, N * sizeof(uint32_t), 
                                    cudaMemcpyHostToDevice, stream1));
        CUDA_CHECK(cudaMemcpyAsync(d_ttl, soa.pinned_ttl, N * sizeof(uint8_t), 
                                    cudaMemcpyHostToDevice, stream1));
        CUDA_CHECK(cudaMemcpyAsync(d_checksum, soa.pinned_checksum, N * sizeof(uint16_t), 
                                    cudaMemcpyHostToDevice, stream2));
    } else {
        // Fallback to synchronous copy if no pinned memory
        CUDA_CHECK(cudaMemcpy(d_dst_ip, soa.dst_ip.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ttl, soa.ttl.data(), N * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_checksum, soa.hdr_checksum.data(), N * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }
    
    CUDA_CHECK(cudaMemcpy(d_rtable, rdev.data(), rdev.size() * sizeof(RouteEntryDevice), cudaMemcpyHostToDevice));

    // --- Kernel launch ---
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // Wait for async transfers to complete before kernel
    if (soa.pinned_dst_ip) {
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        CUDA_CHECK(cudaStreamSynchronize(stream2));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    forward_kernel<<<grid, block>>>(d_dst_ip, d_ttl, d_checksum, d_out_if, d_rtable, rdev.size(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "GPU kernel time: " << ms << " ms\n";

    // --- Async copy back using pinned memory ---
    if (soa.pinned_out_if) {
        CUDA_CHECK(cudaMemcpyAsync(soa.pinned_ttl, d_ttl, N * sizeof(uint8_t), 
                                    cudaMemcpyDeviceToHost, stream1));
        CUDA_CHECK(cudaMemcpyAsync(soa.pinned_out_if, d_out_if, N * sizeof(int), 
                                    cudaMemcpyDeviceToHost, stream2));
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        CUDA_CHECK(cudaStreamSynchronize(stream2));
        
        // Copy from pinned to regular memory
        soa.copyFromPinned();
    } else {
        // Fallback to synchronous copy
        CUDA_CHECK(cudaMemcpy(soa.ttl.data(), d_ttl, N * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(soa.out_if.data(), d_out_if, N * sizeof(int), cudaMemcpyDeviceToHost));
    }

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_dst_ip));
    CUDA_CHECK(cudaFree(d_ttl));
    CUDA_CHECK(cudaFree(d_checksum));
    CUDA_CHECK(cudaFree(d_out_if));
    CUDA_CHECK(cudaFree(d_rtable));
    
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
