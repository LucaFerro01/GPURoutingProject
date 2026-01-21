#include "gpu_verbose.h"

// Global flag definition
bool g_verbose_gpu_timing = true;

void set_gpu_verbose(bool verbose) {
    g_verbose_gpu_timing = verbose;
}
