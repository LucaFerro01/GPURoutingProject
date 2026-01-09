#pragma once
#include <cstdint>
#include <vector>
#include <arpa/inet.h>

struct RouteEntry {
    uint32_t prefix;       // network prefix (host order)
    uint8_t  prefix_len;   // 0..32
    int      out_if;       // output interface index
};

// Compute mask for prefix length
inline uint32_t prefix_mask(uint8_t prefix_len)
{
    if (prefix_len == 0) return 0;
    return ~uint32_t(0) << (32 - prefix_len);  // già in host order
}

// Longest Prefix Match (linear, simple but correct)
inline int longest_prefix_match(uint32_t dst_ip_host,
                                const std::vector<RouteEntry>& rtable)
{
    int best_index = -1;
    int best_len   = -1;

    for (size_t i = 0; i < rtable.size(); i++) {
        uint32_t mask = prefix_mask(rtable[i].prefix_len);
        if ((dst_ip_host & mask) == (rtable[i].prefix & mask)) {
            if (rtable[i].prefix_len > best_len) {
                best_len = rtable[i].prefix_len;
                best_index = static_cast<int>(i);
            }
        }
    }
    return best_index;
}
