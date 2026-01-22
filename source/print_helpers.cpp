#include "print_helpers.h"

#include <arpa/inet.h>
#include <iostream>

void print_routing_info(size_t routes, int batches)
{
    std::cout << "Routing table size: " << routes << " entries" << std::endl;
    std::cout << "Number of batches: " << batches << "\n" << std::endl;
}

void print_cpu_results(double msSerial, double msParallel)
{
    std::cout << std::endl;
    std::cout << "CPU forwarding time: " << msSerial << " ms" << std::endl;
    std::cout << "CPU Parallel forwarding time: " << msParallel << " ms" << std::endl;
}

void print_gpu_header(const char* title)
{
    std::cout << "\n=== " << title << " ===" << std::endl;
}

void print_gpu_lpm_results(double msAoS, double msForward, double msTotal)
{
    std::cout << "  AoS->SoA + pinned alloc: " << msAoS << " ms" << std::endl;
    std::cout << "  GPU forward function:    " << msForward << " ms" << std::endl;
    std::cout << "GPU LPM forwarding total time: " << msTotal << " ms" << std::endl;
}

void print_gpu_bloom_results(double msAoS, double msForward, double msTotal)
{
    std::cout << "  AoS->SoA + pinned alloc: " << msAoS << " ms" << std::endl;
    std::cout << "  GPU forward function:    " << msForward << " ms" << std::endl;
    std::cout << "GPU Bloom forwarding total time: " << msTotal << " ms" << std::endl;
}

void print_gpu_lpm_no_pinned(double msAoS, double msForward, double msTotal)
{
    std::cout << "  AoS->SoA (no pinned):    " << msAoS << " ms" << std::endl;
    std::cout << "  GPU forward function:    " << msForward << " ms" << std::endl;
    std::cout << "GPU LPM forwarding total time: " << msTotal << " ms" << std::endl;
}

void print_performance_comparison(double msGPU, double msGPUNoPinned,
                                  double msGPUBloom, double msGPUBloomNoPinned)
{
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << "GPU LPM speedup with pinned memory: "
              << ((msGPUNoPinned - msGPU) / msGPUNoPinned * 100.0) << "%" << std::endl;
    std::cout << "GPU Bloom speedup with pinned memory: "
              << ((msGPUBloomNoPinned - msGPUBloom) / msGPUBloomNoPinned * 100.0) << "%" << std::endl;
}

void print_debug_samples(const std::vector<Packet>& packetsSerial,
                         const PacketSoA& soa,
                         const PacketSoA& soaBloom)
{
    std::cout << "\nPrimi 5 risultati per debug:\n";
    for (size_t i = 0; i < 5 && i < packetsSerial.size(); i++) {
        std::cout << "Packet " << i
                  << " - CPU out_if=" << packetsSerial[i].out_if
                  << ", GPU LPM out_if=" << soa.out_if[i]
                  << ", GPU Bloom out_if=" << soaBloom.out_if[i]
                  << ", dst_ip=0x" << std::hex << ntohl(packetsSerial[i].hdr.dst_ip) << std::dec
                  << "\n";
    }
    std::cout << std::endl;
}

void print_verification_header()
{
    std::cout << "Verifying correctness..." << std::endl;
}

void print_verification_success()
{
    std::cout << "\n All tests passed!\n";
}

void print_multibatch_header()
{
    std::cout << "\n\n========================================" << std::endl;
    std::cout << "=== MULTI-BATCH TEST (streaming simulation) ===" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

void print_multibatch_pinned_header(int batches)
{
    std::cout << "--- WITH Pinned Memory (reused " << batches << " times) ---" << std::endl;
}

void print_multibatch_pinned_results(double allocPinnedTime,
                                     double processingOnlyPinned,
                                     double totalMultiPinned,
                                     int batches)
{
    std::cout << "  Initial allocation: " << allocPinnedTime << " ms" << std::endl;
    std::cout << "  Processing " << batches << " batches: " << processingOnlyPinned << " ms" << std::endl;
    std::cout << "  Average per batch: " << (processingOnlyPinned / batches) << " ms" << std::endl;
    std::cout << "  Total time: " << totalMultiPinned << " ms\n" << std::endl;
}

void print_multibatch_no_pinned_header(int batches)
{
    std::cout << "--- WITHOUT Pinned Memory (allocate " << batches << " times) ---" << std::endl;
}

void print_multibatch_no_pinned_results(double totalMultiNoPinned, int batches)
{
    std::cout << "  Total time: " << totalMultiNoPinned << " ms" << std::endl;
    std::cout << "  Average per batch: " << (totalMultiNoPinned / batches) << " ms\n" << std::endl;
}

void print_multibatch_summary(double totalMultiPinned, double totalMultiNoPinned)
{
    std::cout << "=== MULTI-BATCH RESULTS ===" << std::endl;
    double speedup = ((totalMultiNoPinned - totalMultiPinned) / totalMultiNoPinned) * 100.0;
    std::cout << "Pinned memory speedup: " << speedup << "%" << std::endl;
    std::cout << "Time saved: " << (totalMultiNoPinned - totalMultiPinned) << " ms" << std::endl;

    if (speedup > 0) {
        std::cout << "Pinned memory IS beneficial for multi-batch processing!" << std::endl;
    } else {
        std::cout << "Pinned memory overhead still too high even for multi-batch processing" << std::endl;
    }
    std::cout << "\n========================================\n" << std::endl;
}
