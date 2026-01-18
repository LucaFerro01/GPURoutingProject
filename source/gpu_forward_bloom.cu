#include <cuda.h>
#include <cuda_runtime.h>
#include "routing_device.h"

#define BLOOM_BITS (1 << 20)

// Hash functions (device)
__device__ __forceinline__
uint32_t hash1_dev(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    return x;
}

__device__ __forceinline__
uint32_t hash2_dev(uint32_t x) {
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__
bool bloom_test(const uint32_t* bloom, uint32_t key)
{
    uint32_t h1 = hash1_dev(key) % BLOOM_BITS;
    uint32_t h2 = hash2_dev(key) % BLOOM_BITS;

    if (!(bloom[h1 >> 5] & (1u << (h1 & 31)))) return false;
    if (!(bloom[h2 >> 5] & (1u << (h2 & 31)))) return false;

    return true;
}

__device__ uint16_t simpleChecksumBloom(uint32_t dst_ip, uint8_t ttl)
{
    uint32_t sum = 0;
    sum += (dst_ip >> 16) & 0xFFFF;
    sum += dst_ip & 0xFFFF;
    sum += ttl;

    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    return static_cast<uint16_t>(~sum);
}

__global__ void forward_kernel_bloom(
    const uint32_t* dst_ip,
    uint8_t* ttl,
    uint16_t* checksum,
    int* out_if,
    const uint32_t* bloom,
    const RouteEntryDevice* rtable,
    int rtable_size,
    const uint8_t* unique_prefix_lens,
    int num_unique_lens,
    int N
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    // TTL check
    if(ttl[i] < 1){
        out_if[i] = -1;
        return;
    }
    ttl[i] -= 1;

    // Recompute checksum
    checksum[i] = simpleChecksumBloom(dst_ip[i], ttl[i]);

    // Bloom-accelerated LPM
    uint32_t dip = dst_ip[i];
    int best = -1;
    int best_len = -1;

    // Iterate only through existing prefix lengths (from longest to shortest)
    for (int idx = 0; idx < num_unique_lens; ++idx) {
        int plen = unique_prefix_lens[idx];
        uint32_t mask = plen == 0 ? 0 : 0xFFFFFFFF << (32 - plen);
        uint32_t net = dip & mask;

        // Quick check with Bloom filter
        if (!bloom_test(bloom, net)) continue;

        // Verify real match in routing table
        for (int r = 0; r < rtable_size; r++) {
            if (rtable[r].prefix_len == plen) {
                uint32_t rt_mask = plen == 0 ? 0 : 0xFFFFFFFF << (32 - plen);
                if ((rtable[r].prefix & rt_mask) == net) {
                    best = r;
                    best_len = plen;
                    goto found; // Early exit on first match
                }
            }
        }
    }

found:
    out_if[i] = (best >= 0) ? rtable[best].out_if : -1;
}
