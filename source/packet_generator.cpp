#include "ip_types.h"
#include <random>

std::vector<Packet> generate_packets(size_t N)
{
    std::vector<Packet> packets(N);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<uint32_t> ipdist(0, 0xFFFFFFFF);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; i++) {
        Packet& p = packets[i];
        IPv4Header& h = p.hdr;

        h.ver_ihl = (4 << 4) | 5; // IPv4, IHL = 5 (20 bytes)
        h.tos = 0;
        h.tot_len = htons(20);
        h.id = htons(0);
        h.frag_off = htons(0);
        h.ttl = 64;
        h.protocol = 17; // UDP

        h.src_ip = htonl(ipdist(rng));
        h.dst_ip = htonl(ipdist(rng));

        // minimal payload
        p.payload = {0xDE, 0xAD, 0xBE, 0xEF};

        // checksum
        h.hdr_checksum = 0;
        h.hdr_checksum = ipv4_checksum(&h, sizeof(IPv4Header));
    }

    return packets;
}
