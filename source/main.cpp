#include "ip_types.h"
#include "routing.h"
#include <iostream>
#include <chrono>
#include <omp.h>
#include <cassert>

#define N_Packets 1000000

std::vector<Packet> generate_packets(size_t N);
void forward_packet_cpu(Packet& p, const std::vector<RouteEntry>& rtable);
void forward_packets_cpu_parallel(std::vector<Packet>&, const std::vector<RouteEntry>&);

int main()
{
    // Example routing table
    std::vector<RouteEntry> rtable = {
        { ntohl(inet_addr("10.0.0.0")),   8,  1 },
        { ntohl(inet_addr("192.168.0.0")),8, 2 },
        { 0, 0, 3 } // default route
    };

    auto packetsSerial = generate_packets(N_Packets);
    std::vector<Packet> packetsParallel = packetsSerial;

    // --- Serial Part --- 
    auto startSerial = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < packetsSerial.size(); i++) {
        forward_packet_cpu(packetsSerial[i], rtable);
    }
    auto endSerial = std::chrono::high_resolution_clock::now();

    // for(size_t i = 0; i < packets.size(); i++) {
    //     std::cout << "Packet " << i
    //         << " -> out_if=" << packets[i].out_if
    //         << " TTL=" << int(packets[i].hdr.ttl)
    //         << " checksum=0x" << std::hex << ntohs(packets[i].hdr.hdr_checksum)
    //         << std::dec << "\n";
    // }
    
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

    // Check for see if the output operations was the same between the two type of operations
    for(size_t i = 0; i < packetsSerial.size(); i++)
        assert(packetsSerial[i].out_if == packetsParallel[i].out_if);

    return 0;
}
