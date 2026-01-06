#include <cstdint>

struct RouteEntryDevice
{
    uint32_t prefix;
    uint8_t prefix_len;
    int out_if;
};
