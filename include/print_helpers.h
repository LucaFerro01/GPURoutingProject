#pragma once

#include "ip_types.h"
#include "packet_soa.h"
#include <vector>

void print_routing_info(size_t routes, int batches);
void print_cpu_results(double msSerial, double msParallel);
void print_gpu_header(const char* title);
void print_gpu_lpm_results(double msAoS, double msForward, double msTotal);
void print_gpu_bloom_results(double msAoS, double msForward, double msTotal);
void print_gpu_lpm_no_pinned(double msAoS, double msForward, double msTotal);
void print_performance_comparison(double msGPU, double msGPUNoPinned,
                                  double msGPUBloom, double msGPUBloomNoPinned);
void print_debug_samples(const std::vector<Packet>& packetsSerial,
                         const PacketSoA& soa,
                         const PacketSoA& soaBloom);
void print_verification_header();
void print_verification_success();
void print_multibatch_header();
void print_multibatch_pinned_header(int batches);
void print_multibatch_pinned_results(double allocPinnedTime,
                                     double processingOnlyPinned,
                                     double totalMultiPinned,
                                     int batches);
void print_multibatch_no_pinned_header(int batches);
void print_multibatch_no_pinned_results(double totalMultiNoPinned, int batches);
void print_multibatch_summary(double totalMultiPinned, double totalMultiNoPinned);
