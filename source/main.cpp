#include "ip_types.h"
#include "routing.h"
#include <iostream>
#include <chrono>
#define N_Packets 1000

std::vector<Packet> generate_packets(size_t N);
void forward_packet_cpu(Packet& p, const std::vector<RouteEntry>& rtable);

int main()
{
    // Example routing table
    std::vector<RouteEntry> rtable = {
        { ntohl(inet_addr("10.0.0.0")),   8,  1 },
        { ntohl(inet_addr("192.168.0.0")),8, 2 },
        { 0, 0, 3 } // default route
    };

    auto packets = generate_packets(N_Packets);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < packets.size(); i++) {
        forward_packet_cpu(packets[i], rtable);
    }
    auto end = std::chrono::high_resolution_clock::now();

    for(size_t i = 0; i < packets.size(); i++) {
        std::cout << "Packet " << i
            << " -> out_if=" << packets[i].out_if
            << " TTL=" << int(packets[i].hdr.ttl)
            << " checksum=0x" << std::hex << ntohs(packets[i].hdr.hdr_checksum)
            << std::dec << "\n";
    }
    
    std::cout << std::endl;

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "CPU forwarding time: " << ms << " ms\n";


    return 0;
}
