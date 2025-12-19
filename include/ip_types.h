#pragma once
#include <cstdint>
#include <vector>
#include <cstring>
#include <winsock.h>

// IPv4 header is 20 bytes without options
#pragma pack(push,1)
struct IPv4Header {
    uint8_t  ver_ihl;        // version (4 bits) + IHL (4 bits)
    uint8_t  tos;
    uint16_t tot_len;
    uint16_t id;
    uint16_t frag_off;
    uint8_t  ttl;
    uint8_t  protocol;
    uint16_t hdr_checksum;
    uint32_t src_ip;
    uint32_t dst_ip;
};
#pragma pack(pop)

struct Packet {
    IPv4Header hdr;
    std::vector<uint8_t> payload;

    // Output metadata (e.g., out interface or drop flag)
    int out_if = -1;
};

// Compute IPv4 checksum (one's complement)
inline uint16_t ipv4_checksum(const void* data, size_t len_bytes)
{
    uint32_t sum = 0;
    const uint16_t* ptr = reinterpret_cast<const uint16_t*>(data);

    for (size_t i = 0; i < len_bytes / 2; i++) {
        sum += ntohs(ptr[i]);
        if (sum & 0x10000)
            sum = (sum & 0xFFFF) + 1;
    }

    // If odd length: shouldn't happen for IPv4 header
    if (len_bytes & 1) {
        uint8_t last = *(reinterpret_cast<const uint8_t*>(data) + len_bytes - 1);
        sum += last << 8;
        if (sum & 0x10000)
            sum = (sum & 0xFFFF) + 1;
    }

    return htons(~sum & 0xFFFF);
}
