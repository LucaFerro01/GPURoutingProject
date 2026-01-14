#ifndef BLOOM_FILTER_H
#define BLOOM_FILTER_H

#include <cstdint>
#include <cstring>

#define BLOOM_BITS (1 << 20)  // 1M bits
#define BLOOM_WORDS (BLOOM_BITS / 32)

// Hash functions (CPU e GPU compatibili)
inline uint32_t hash1(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    return x;
}

inline uint32_t hash2(uint32_t x) {
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

// CPU functions
void bloom_insert(uint32_t* bloom, uint32_t network);

void build_bloom_filter(
    uint32_t* bloom,
    const uint32_t* prefixes,
    const uint8_t* prefix_len,
    int n
);

#endif // BLOOM_FILTER_H
