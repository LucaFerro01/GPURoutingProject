#include "ip_types.h"
#include "routing.h"
#include "packet_soa.h"
#include <iostream>
#include <chrono>
#include <omp.h>
#include <cassert>
#include <random>

#define N_Packets 10000000

std::vector<Packet> generate_packets(size_t N);
void forward_packet_cpu(Packet& p, const std::vector<RouteEntry>& rtable);
void forward_packets_cpu_parallel(std::vector<Packet>&, const std::vector<RouteEntry>&);
PacketSoA aos_to_soa(const std::vector<Packet>& packets);
void gpu_forward(PacketSoA& soa, const std::vector<RouteEntry>& rtable);
void gpu_forward_bloom(PacketSoA& soa, const std::vector<RouteEntry>& rtable);

// Generate a realistic routing table with various prefix lengths
std::vector<RouteEntry> generate_routing_table(size_t num_entries) {
    std::vector<RouteEntry> rtable;
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<uint32_t> dist_ip(0, 0xFFFFFFFF);
    
    // Add some common prefixes with different lengths
    // Typical distribution: many /24, some /16, few /8
    size_t num_8 = num_entries / 20;   // 5% /8
    size_t num_16 = num_entries / 5;   // 20% /16
    size_t num_24 = num_entries - num_8 - num_16; // 75% /24
    
    int out_if = 1;
    
    // Generate /8 prefixes
    for (size_t i = 0; i < num_8; i++) {
        uint32_t prefix = (dist_ip(rng) & 0xFF000000);
        rtable.push_back({prefix, 8, out_if++});
    }
    
    // Generate /16 prefixes
    for (size_t i = 0; i < num_16; i++) {
        uint32_t prefix = (dist_ip(rng) & 0xFFFF0000);
        rtable.push_back({prefix, 16, out_if++});
    }
    
    // Generate /24 prefixes
    for (size_t i = 0; i < num_24; i++) {
        uint32_t prefix = (dist_ip(rng) & 0xFFFFFF00);
        rtable.push_back({prefix, 24, out_if++});
    }
    
    // Add default route
    rtable.push_back({0, 0, 9999});
    
    return rtable;
}

int main()
{
    // Generate routing table - change this number to test different sizes
    // Try: 3, 100, 1000, 5000, 10000
    const size_t NUM_ROUTES = 5000;
    
    std::vector<RouteEntry> rtable;
    if (NUM_ROUTES <= 3) {
        // Small test table
        rtable = {
            { ntohl(inet_addr("10.0.0.0")),   8,  1 },
            { ntohl(inet_addr("192.168.0.0")),8, 2 },
            { 0, 0, 3 } // default route
        };
    } else {
        rtable = generate_routing_table(NUM_ROUTES);
    }
    
    std::cout << "Routing table size: " << rtable.size() << " entries\n" << std::endl;

    auto packets = generate_packets(N_Packets);

    std::vector<Packet> packetsSerial = packets;
    std::vector<Packet> packetsParallel = packets;

    // --- Serial Part --- 
    auto startSerial = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < packetsSerial.size(); i++) {
        forward_packet_cpu(packetsSerial[i], rtable);
    }
    auto endSerial = std::chrono::high_resolution_clock::now();
    
    std::cout << std::endl;

    double msSerial = std::chrono::duration<double, std::milli>(endSerial - startSerial).count();
    std::cout << "CPU forwarding time: " << msSerial << " ms" << std::endl;
    // --- End Serial Part

    // --- Parallel Part
    auto startParallel = std::chrono::high_resolution_clock::now();
    forward_packets_cpu_parallel(packetsParallel, rtable);
    auto endParallel = std::chrono::high_resolution_clock::now();

    double msParallel = std::chrono::duration<double, std::milli>(endParallel - startParallel).count();
    std::cout << "CPU Parallel forwarding time: " << msParallel << " ms" << std::endl;
    // --- End Parallel Part

    // --- GPU part (LPM standard) ---
    auto startGPU = std::chrono::high_resolution_clock::now();
    PacketSoA soa = aos_to_soa(packets);
    gpu_forward(soa, rtable);
    auto endGPU = std::chrono::high_resolution_clock::now();
    double msGPU = std::chrono::duration<double, std::milli>(endGPU - startGPU).count();
    std::cout << "GPU LPM forwarding total time: " << msGPU << " ms" << std::endl;
    // --- End GPU part

    // --- GPU Bloom filter part ---
    auto startGPUBloom = std::chrono::high_resolution_clock::now();
    PacketSoA soaBloom = aos_to_soa(packets);
    gpu_forward_bloom(soaBloom, rtable);
    auto endGPUBloom = std::chrono::high_resolution_clock::now();
    double msGPUBloom = std::chrono::duration<double, std::milli>(endGPUBloom - startGPUBloom).count();
    std::cout << "GPU Bloom forwarding total time: " << msGPUBloom << " ms" << std::endl;
    // --- End GPU Bloom part

    // Check operations
    std::cout << "\nPrimi 5 risultati per debug:\n";
    for(size_t i = 0; i < 5 && i < packets.size(); i++)
    {
        std::cout << "Packet " << i 
                  << " - CPU out_if=" << packetsSerial[i].out_if
                  << ", GPU LPM out_if=" << soa.out_if[i]
                  << ", GPU Bloom out_if=" << soaBloom.out_if[i]
                  << ", dst_ip=0x" << std::hex << ntohl(packetsSerial[i].hdr.dst_ip) << std::dec
                  << "\n";
    }
    std::cout << std::endl;

    for(size_t i = 0; i < packets.size(); i++)
    {
        // Check the serial and parallel packets was the same
        assert(packetsSerial[i].out_if == packetsParallel[i].out_if);
        // Check GPU LPM result matches CPU serial result
        assert(soa.out_if[i] == packetsSerial[i].out_if);
        // Check GPU Bloom result matches CPU serial result
        assert(soaBloom.out_if[i] == packetsSerial[i].out_if);
        assert(soa.ttl[i] == packetsSerial[i].hdr.ttl);
        assert(soa.dst_ip[i] == ntohl(packetsSerial[i].hdr.dst_ip));
    }

    std::cout << "\n All tests passed!\n";

    return 0;
}
