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
PacketSoA aos_to_soa_no_pinned(const std::vector<Packet>& packets);
void gpu_forward(PacketSoA& soa, const std::vector<RouteEntry>& rtable);
void gpu_forward_bloom(PacketSoA& soa, const std::vector<RouteEntry>& rtable);
void set_gpu_verbose(bool verbose);

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
    const size_t NUM_ROUTES = 100;
    const int NUM_BATCHES = 10;  // Number of batches to process
    
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
    
    std::cout << "Routing table size: " << rtable.size() << " entries" << std::endl;
    std::cout << "Number of batches: " << NUM_BATCHES << "\n" << std::endl;

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

    std::cout << "\n=== GPU Tests WITH Pinned Memory ===" << std::endl;

    // --- GPU part (LPM standard) with pinned memory ---
    auto startGPU = std::chrono::high_resolution_clock::now();
    auto startAoS = std::chrono::high_resolution_clock::now();
    PacketSoA soa = aos_to_soa(packets);
    auto endAoS = std::chrono::high_resolution_clock::now();
    double msAoS = std::chrono::duration<double, std::milli>(endAoS - startAoS).count();
    
    auto startForward = std::chrono::high_resolution_clock::now();
    gpu_forward(soa, rtable);
    auto endForward = std::chrono::high_resolution_clock::now();
    double msForward = std::chrono::duration<double, std::milli>(endForward - startForward).count();
    
    auto endGPU = std::chrono::high_resolution_clock::now();
    double msGPU = std::chrono::duration<double, std::milli>(endGPU - startGPU).count();
    std::cout << "  AoS->SoA + pinned alloc: " << msAoS << " ms" << std::endl;
    std::cout << "  GPU forward function:    " << msForward << " ms" << std::endl;
    std::cout << "GPU LPM forwarding total time: " << msGPU << " ms" << std::endl;
    // --- End GPU part

    // --- GPU Bloom filter part with pinned memory ---
    auto startGPUBloom = std::chrono::high_resolution_clock::now();
    auto startAoSBloom = std::chrono::high_resolution_clock::now();
    PacketSoA soaBloom = aos_to_soa(packets);
    auto endAoSBloom = std::chrono::high_resolution_clock::now();
    double msAoSBloom = std::chrono::duration<double, std::milli>(endAoSBloom - startAoSBloom).count();
    
    auto startForwardBloom = std::chrono::high_resolution_clock::now();
    gpu_forward_bloom(soaBloom, rtable);
    auto endForwardBloom = std::chrono::high_resolution_clock::now();
    double msForwardBloom = std::chrono::duration<double, std::milli>(endForwardBloom - startForwardBloom).count();
    
    auto endGPUBloom = std::chrono::high_resolution_clock::now();
    double msGPUBloom = std::chrono::duration<double, std::milli>(endGPUBloom - startGPUBloom).count();
    std::cout << "  AoS->SoA + pinned alloc: " << msAoSBloom << " ms" << std::endl;
    std::cout << "  GPU forward function:    " << msForwardBloom << " ms" << std::endl;
    std::cout << "GPU Bloom forwarding total time: " << msGPUBloom << " ms" << std::endl;
    // --- End GPU Bloom part

    std::cout << "\n=== GPU Tests WITHOUT Pinned Memory ===" << std::endl;

    // --- GPU LPM without pinned memory ---
    auto startGPUNoPinned = std::chrono::high_resolution_clock::now();
    auto startAoSNoPinned = std::chrono::high_resolution_clock::now();
    PacketSoA soaNoPinned = aos_to_soa_no_pinned(packets);
    auto endAoSNoPinned = std::chrono::high_resolution_clock::now();
    double msAoSNoPinned = std::chrono::duration<double, std::milli>(endAoSNoPinned - startAoSNoPinned).count();
    
    auto startForwardNoPinned = std::chrono::high_resolution_clock::now();
    gpu_forward(soaNoPinned, rtable);
    auto endForwardNoPinned = std::chrono::high_resolution_clock::now();
    double msForwardNoPinned = std::chrono::duration<double, std::milli>(endForwardNoPinned - startForwardNoPinned).count();
    
    auto endGPUNoPinned = std::chrono::high_resolution_clock::now();
    double msGPUNoPinned = std::chrono::duration<double, std::milli>(endGPUNoPinned - startGPUNoPinned).count();
    std::cout << "  AoS->SoA (no pinned):    " << msAoSNoPinned << " ms" << std::endl;
    std::cout << "  GPU forward function:    " << msForwardNoPinned << " ms" << std::endl;
    std::cout << "GPU LPM forwarding total time: " << msGPUNoPinned << " ms" << std::endl;

    // --- GPU Bloom without pinned memory ---
    auto startGPUBloomNoPinned = std::chrono::high_resolution_clock::now();
    PacketSoA soaBloomNoPinned = aos_to_soa_no_pinned(packets);
    gpu_forward_bloom(soaBloomNoPinned, rtable);
    auto endGPUBloomNoPinned = std::chrono::high_resolution_clock::now();
    double msGPUBloomNoPinned = std::chrono::duration<double, std::milli>(endGPUBloomNoPinned - startGPUBloomNoPinned).count();
    std::cout << "GPU Bloom forwarding total time: " << msGPUBloomNoPinned << " ms" << std::endl;

    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << "GPU LPM speedup with pinned memory: " 
              << ((msGPUNoPinned - msGPU) / msGPUNoPinned * 100.0) << "%" << std::endl;
    std::cout << "GPU Bloom speedup with pinned memory: " 
              << ((msGPUBloomNoPinned - msGPUBloom) / msGPUBloomNoPinned * 100.0) << "%" << std::endl;

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

    std::cout << "Verifying correctness..." << std::endl;
    for(size_t i = 0; i < packets.size(); i++)
    {
        // Check the serial and parallel packets was the same
        assert(packetsSerial[i].out_if == packetsParallel[i].out_if);
        // Check GPU LPM result matches CPU serial result (both pinned and non-pinned)
        assert(soa.out_if[i] == packetsSerial[i].out_if);
        assert(soaNoPinned.out_if[i] == packetsSerial[i].out_if);
        // Check GPU Bloom result matches CPU serial result (both pinned and non-pinned)
        assert(soaBloom.out_if[i] == packetsSerial[i].out_if);
        assert(soaBloomNoPinned.out_if[i] == packetsSerial[i].out_if);
        assert(soa.ttl[i] == packetsSerial[i].hdr.ttl);
        assert(soa.dst_ip[i] == ntohl(packetsSerial[i].hdr.dst_ip));
    }

    std::cout << "\n All tests passed!\n";

    // ==========================================================================
    // MULTI-BATCH TEST: Simulating continuous processing
    // ==========================================================================
    std::cout << "\n\n========================================" << std::endl;
    std::cout << "=== MULTI-BATCH TEST (streaming simulation) ===" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Disable verbose GPU timing for cleaner output during batch processing
    set_gpu_verbose(false);

    // Test 1: WITH pinned memory (allocate ONCE, reuse multiple times)
    std::cout << "--- WITH Pinned Memory (reused " << NUM_BATCHES << " times) ---" << std::endl;
    
    auto startMultiPinned = std::chrono::high_resolution_clock::now();
    
    // Allocate pinned memory ONCE
    PacketSoA soaReusable = aos_to_soa(packets);
    
    auto afterAllocPinned = std::chrono::high_resolution_clock::now();
    double allocPinnedTime = std::chrono::duration<double, std::milli>(afterAllocPinned - startMultiPinned).count();
    
    // Process multiple batches using the SAME pinned memory
    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        // In a real scenario, you would update the data here
        // For this test, we just reprocess the same data
        gpu_forward_bloom(soaReusable, rtable);
    }
    
    auto endMultiPinned = std::chrono::high_resolution_clock::now();
    double totalMultiPinned = std::chrono::duration<double, std::milli>(endMultiPinned - startMultiPinned).count();
    double processingOnlyPinned = std::chrono::duration<double, std::milli>(endMultiPinned - afterAllocPinned).count();
    
    std::cout << "  Initial allocation: " << allocPinnedTime << " ms" << std::endl;
    std::cout << "  Processing " << NUM_BATCHES << " batches: " << processingOnlyPinned << " ms" << std::endl;
    std::cout << "  Average per batch: " << (processingOnlyPinned / NUM_BATCHES) << " ms" << std::endl;
    std::cout << "  Total time: " << totalMultiPinned << " ms\n" << std::endl;

    // Test 2: WITHOUT pinned memory (allocate every time)
    std::cout << "--- WITHOUT Pinned Memory (allocate " << NUM_BATCHES << " times) ---" << std::endl;
    
    auto startMultiNoPinned = std::chrono::high_resolution_clock::now();
    
    // Process multiple batches, allocating each time
    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        PacketSoA soaBatch = aos_to_soa_no_pinned(packets);
        gpu_forward_bloom(soaBatch, rtable);
    }
    
    auto endMultiNoPinned = std::chrono::high_resolution_clock::now();
    double totalMultiNoPinned = std::chrono::duration<double, std::milli>(endMultiNoPinned - startMultiNoPinned).count();
    
    std::cout << "  Total time: " << totalMultiNoPinned << " ms" << std::endl;
    std::cout << "  Average per batch: " << (totalMultiNoPinned / NUM_BATCHES) << " ms\n" << std::endl;

    // Results
    std::cout << "=== MULTI-BATCH RESULTS ===" << std::endl;
    double speedup = ((totalMultiNoPinned - totalMultiPinned) / totalMultiNoPinned) * 100.0;
    std::cout << "Pinned memory speedup: " << speedup << "%" << std::endl;
    std::cout << "Time saved: " << (totalMultiNoPinned - totalMultiPinned) << " ms" << std::endl;
    
    if (speedup > 0) {
        std::cout << "✅ Pinned memory IS beneficial for multi-batch processing!" << std::endl;
    } else {
        std::cout << "⚠️  Pinned memory overhead still too high even for " << NUM_BATCHES << " batches" << std::endl;
    }
    std::cout << "\n========================================\n" << std::endl;

    return 0;
}
