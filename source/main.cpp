#include "ip_types.h"
#include "routing.h"
#include "packet_soa.h"
#include <iostream>
#include <chrono>
#include <omp.h>
#include <cassert>

#define N_Packets 1000000

std::vector<Packet> generate_packets(size_t N);
void forward_packet_cpu(Packet& p, const std::vector<RouteEntry>& rtable);
void forward_packets_cpu_parallel(std::vector<Packet>&, const std::vector<RouteEntry>&);
PacketSoA aos_to_soa(const std::vector<Packet>& packets);

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

    // --- GPU part (for now only SOA part)
    PacketSoA soa = aos_to_soa(packets);

    // Check operations
    for(size_t i = 0; i < packets.size(); i++)
    {
        // Check the serial and parallel packets was the same
        assert(packetsSerial[i].out_if == packetsParallel[i].out_if);
        // Check the correct creatio of the SOA object
        assert(soa.ttl[i] == packets[i].hdr.ttl);
        assert(soa.out_if[i] == packets[i].out_if);
        assert(soa.dst_ip[i] == ntohl(packets[i].hdr.dst_ip));
    }

    return 0;
}
