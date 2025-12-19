#include "ip_types.h"
#include "routing.h"

void forward_packet_cpu(Packet& p, const std::vector<RouteEntry>& rtable)
{
    IPv4Header& h = p.hdr;

    // Check IPv4
    uint8_t version = h.ver_ihl >> 4;
    if (version != 4) {
        p.out_if = -1;
        return;
    }

    // TTL decrement
    if (h.ttl <= 1) {
        p.out_if = -1; // dropped
        return;
    }
    h.ttl--;

    // Recompute checksum
    h.hdr_checksum = 0;
    h.hdr_checksum = ipv4_checksum(&h, sizeof(IPv4Header));

    // Convert destination IP to host order
    uint32_t dst_host = ntohl(h.dst_ip);

    // Longest prefix match
    int route_idx = longest_prefix_match(dst_host, rtable);
    if (route_idx < 0) {
        p.out_if = -1; // drop
        return;
    }

    p.out_if = rtable[route_idx].out_if;
}
