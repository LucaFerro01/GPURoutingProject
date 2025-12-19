#include "ip_types.h"
#include "routing.h"
#include <iostream>

std::vector<Packet> generate_packets(size_t N);
void forward_packet_cpu(Packet& p, const std::vector<RouteEntry>& rtable);

int main()
{
    // Example routing table
    std::vector<RouteEntry> rtable = {
        { ntohl(inet_addr("10.0.0.0")),   8,  1 },
        { ntohl(inet_addr("192.168.0.0")),16, 2 },
        { 0, 0, 3 } // default route
    };

    auto packets = generate_packets(5);

    for (size_t i = 0; i < packets.size(); i++) {
        forward_packet_cpu(packets[i], rtable);

        std::cout << "Packet " << i
                  << " -> out_if=" << packets[i].out_if
                  << " TTL=" << int(packets[i].hdr.ttl)
                  << " checksum=0x" << std::hex << ntohs(packets[i].hdr.hdr_checksum)
                  << std::dec << "\n";
    }

    return 0;
}
