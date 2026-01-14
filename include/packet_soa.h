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

    // Puntatori per pinned memory (opzionale)
    uint32_t* pinned_dst_ip = nullptr;
    uint8_t*  pinned_ttl = nullptr;
    uint16_t* pinned_checksum = nullptr;
    int*      pinned_out_if = nullptr;
    
    void allocatePinned();
    void freePinned();
    void copyToPinned();
    void copyFromPinned();

    PacketSoA(size_t n)
        : N(n),
          dst_ip(n),
          ttl(n),
          hdr_checksum(n),
          out_if(n, -1) {}
};
