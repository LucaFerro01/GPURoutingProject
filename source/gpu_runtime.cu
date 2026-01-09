#include "packet_soa.h"
#include "routing_device.h"
#include "routing.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

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

    cudaMalloc(&d_dst_ip,    N * sizeof(uint32_t));
    cudaMalloc(&d_ttl,       N * sizeof(uint8_t));
    cudaMalloc(&d_checksum,  N * sizeof(uint16_t));
    cudaMalloc(&d_out_if,    N * sizeof(int));
    cudaMalloc(&d_rtable,    rtable.size() * sizeof(RouteEntryDevice));

    // --- Copy packet data ---
    cudaMemcpy(d_dst_ip, soa.dst_ip.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ttl,    soa.ttl.data(),    N * sizeof(uint8_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_checksum, soa.hdr_checksum.data(),
               N * sizeof(uint16_t), cudaMemcpyHostToDevice);

    // --- Copy routing table ---
    std::vector<RouteEntryDevice> rdev(rtable.size());
    for (size_t i = 0; i < rtable.size(); i++) {
        rdev[i] = { rtable[i].prefix, rtable[i].prefix_len, rtable[i].out_if };
    }
    cudaMemcpy(d_rtable, rdev.data(),
               rdev.size() * sizeof(RouteEntryDevice),
               cudaMemcpyHostToDevice);

    // --- Kernel launch ---
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    forward_kernel<<<grid, block>>>(d_dst_ip, d_ttl, d_checksum,
                                    d_out_if, d_rtable,
                                    rdev.size(), N);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "GPU kernel time: " << ms << " ms\n";

    // --- Copy back ---
    cudaMemcpy(soa.ttl.data(), d_ttl, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(soa.out_if.data(), d_out_if, N * sizeof(int), cudaMemcpyDeviceToHost);

    // --- Cleanup ---
    cudaFree(d_dst_ip);
    cudaFree(d_ttl);
    cudaFree(d_checksum);
    cudaFree(d_out_if);
    cudaFree(d_rtable);
}
