#include "bloom_filter.h"

void bloom_insert(uint32_t* bloom, uint32_t network)
{
    uint32_t h1 = hash1(network) % BLOOM_BITS;
    uint32_t h2 = hash2(network) % BLOOM_BITS;

    bloom[h1 >> 5] |= (1u << (h1 & 31));
    bloom[h2 >> 5] |= (1u << (h2 & 31));
}

void build_bloom_filter(
    uint32_t* bloom,
    const uint32_t* prefixes,
    const uint8_t* prefix_len,
    int n
)
{
    memset(bloom, 0, BLOOM_WORDS * sizeof(uint32_t));

    for (int i = 0; i < n; ++i) {
        uint32_t mask = prefix_len[i] == 0 ? 0 : 0xFFFFFFFF << (32 - prefix_len[i]);
        uint32_t net = prefixes[i] & mask;
        bloom_insert(bloom, net);
    }
}
