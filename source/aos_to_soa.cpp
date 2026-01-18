#include "packet_soa.h"
#include "ip_types.h"
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

PacketSoA aos_to_soa(const std::vector<Packet>& packets)
{
    PacketSoA soa(packets.size());

    for (size_t i = 0; i < packets.size(); i++) {
        const IPv4Header& h = packets[i].hdr;

        soa.dst_ip[i] = ntohl(h.dst_ip);   // convert ONCE, not do in the GPU
        soa.ttl[i] = h.ttl;
        soa.hdr_checksum[i] = h.hdr_checksum;
        soa.out_if[i] = packets[i].out_if;
    }

    // Allocate pinned memory and copy data for faster GPU transfers
    soa.allocatePinned();
    soa.copyToPinned();

    return soa;
}

void PacketSoA::allocatePinned() {
    cudaError_t err;
    
    err = cudaMallocHost(&pinned_dst_ip, N * sizeof(uint32_t));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    err = cudaMallocHost(&pinned_ttl, N * sizeof(uint8_t));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    err = cudaMallocHost(&pinned_checksum, N * sizeof(uint16_t));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    
    err = cudaMallocHost(&pinned_out_if, N * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void PacketSoA::freePinned() {
    if (pinned_dst_ip) cudaFreeHost(pinned_dst_ip);
    if (pinned_ttl) cudaFreeHost(pinned_ttl);
    if (pinned_checksum) cudaFreeHost(pinned_checksum);
    if (pinned_out_if) cudaFreeHost(pinned_out_if);
    
    pinned_dst_ip = nullptr;
    pinned_ttl = nullptr;
    pinned_checksum = nullptr;
    pinned_out_if = nullptr;
}

void PacketSoA::copyToPinned() {
    if (!pinned_dst_ip) return;
    
    std::copy(dst_ip.begin(), dst_ip.end(), pinned_dst_ip);
    std::copy(ttl.begin(), ttl.end(), pinned_ttl);
    std::copy(hdr_checksum.begin(), hdr_checksum.end(), pinned_checksum);
    std::copy(out_if.begin(), out_if.end(), pinned_out_if);
}

void PacketSoA::copyFromPinned() {
    if (!pinned_out_if) return;
    
    std::copy(pinned_ttl, pinned_ttl + N, ttl.begin());
    std::copy(pinned_out_if, pinned_out_if + N, out_if.begin());
}
