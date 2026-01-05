#pragma once
#include <cstdint>
#include <vector>

struct PacketSoA {
    size_t N;

    // IPv4 fields
    std::vector<uint32_t> dst_ip;        // host order
    std::vector<uint8_t>  ttl;
    std::vector<uint16_t> hdr_checksum;

    // Output
    std::vector<int> out_if;

    PacketSoA(size_t n)
        : N(n),
          dst_ip(n),
          ttl(n),
          hdr_checksum(n),
          out_if(n, -1) {}
};
