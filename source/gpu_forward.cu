#include <cuda.h>
#include <cuda_runtime.h>
#include "routing_device.h"

__device__ __forceinline__
uint32_t prefixMask(uint32_t prefix_len)
{
    return prefix_len == 0 ? 0 : 0xFFFFFFFF << (32 - prefix_len);
}

__device__ uint16_t simpleChecksum(uint32_t dst_ip, uint8_t ttl)
{
    uint32_t sum = 0;
    sum += (dst_ip >> 16) & 0xFFFF; // High 16 bits
    sum += dst_ip & 0xFFFF;         // Low 16 bits
    sum += ttl;                     // TTL

    // Fold 32-bit sum to 16 bits
    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    return static_cast<uint16_t>(~sum);
}

__global__ void forward_kernel(
    const uint32_t* dst_ip,
    uint8_t* ttl,
    uint16_t* checksum,
    int* out_if,
    const RouteEntryDevice* rtable,
    int rtable_size,
    int N
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    // TTL
    if(ttl[i] < 1){
        out_if[i] = -1; // Drop packet
        return;
    }
    ttl[i] -= 1;

    // Recompute checksum (simplified model)
    checksum[i] = simpleChecksum(dst_ip[i], ttl[i]);

    // Longest Prefix Match
    int best = -1;
    int best_len = -1;
    
    uint32_t dip = dst_ip[i];

    for(int r = 0; r < rtable_size; r++){
        uint32_t mask = prefixMask(rtable[r].prefix_len);
        if((dip & mask) == (rtable[r].prefix & mask)){
            if (rtable[r].prefix_len > best_len) {
                best_len = rtable[r].prefix_len;
                best = r;
            }
        }
    }

    out_if[i] = (best >= 0) ? rtable[best].out_if : -1;
}