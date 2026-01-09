#include "ip_types.h"
#include "routing.h"
#include "packet_soa.h"
#include <iostream>
#include <chrono>
#include <omp.h>
#include <cassert>

#define N_Packets 10000000

std::vector<Packet> generate_packets(size_t N);
void forward_packet_cpu(Packet& p, const std::vector<RouteEntry>& rtable);
void forward_packets_cpu_parallel(std::vector<Packet>&, const std::vector<RouteEntry>&);
PacketSoA aos_to_soa(const std::vector<Packet>& packets);
void gpu_forward(PacketSoA& soa, const std::vector<RouteEntry>& rtable);

int main()
{
    // Example routing table
    std::vector<RouteEntry> rtable = {
        { ntohl(inet_addr("10.0.0.0")),   8,  1 },
        { ntohl(inet_addr("192.168.0.0")),8, 2 },
        { 0, 0, 3 } // default route
    };

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

    // --- GPU part (for now only SOA part)
    auto startGPU = std::chrono::high_resolution_clock::now();
    PacketSoA soa = aos_to_soa(packets);
    gpu_forward(soa, rtable);
    auto endGPU = std::chrono::high_resolution_clock::now();
    double msGPU = std::chrono::duration<double, std::milli>(endGPU - startGPU).count();
    std::cout << "GPU forwarding total time (incl. mem transfer): " << msGPU << " ms" << std::endl;
    // --- End GPU part



    // Check operations
    std::cout << "\nPrimi 5 risultati per debug:\n";
    for(size_t i = 0; i < 5 && i < packets.size(); i++)
    {
        std::cout << "Packet " << i 
                  << " - CPU out_if=" << packetsSerial[i].out_if
                  << ", GPU out_if=" << soa.out_if[i]
                  << ", dst_ip=0x" << std::hex << ntohl(packetsSerial[i].hdr.dst_ip) << std::dec
                  << "\n";
    }
    std::cout << std::endl;

    for(size_t i = 0; i < packets.size(); i++)
    {
        // Check the serial and parallel packets was the same
        assert(packetsSerial[i].out_if == packetsParallel[i].out_if);
        // Check GPU result matches CPU serial result
        assert(soa.out_if[i] == packetsSerial[i].out_if);
        assert(soa.ttl[i] == packetsSerial[i].hdr.ttl);
        assert(soa.dst_ip[i] == ntohl(packetsSerial[i].hdr.dst_ip));
    }

    return 0;
}
