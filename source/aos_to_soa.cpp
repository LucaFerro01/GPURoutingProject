#include "packet_soa.h"
#include "ip_types.h"
#include <arpa/inet.h>

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

    return soa;
}
