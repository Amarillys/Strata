// Keep the shared host's local KT entry points linkable on Metal. KT GPU kernels
// and x86 AVX2 expert rows are not implemented by this backend.
#include "strata/kernels/kt_kernels.hpp"
#include "strata/kernels/cpu/kt_avx2.hpp"

namespace strata::metal_engine {
void missing(const char* name);
}

namespace strata::kernels {
void kt_dequant_rows(int, const void*, int64_t, int64_t, void*, int, void*, int64_t, int, const int32_t*) {
    strata::metal_engine::missing("strata::kernels::kt_dequant_rows");
}
}

namespace strata::kernels::cpu {
void kt256_gu_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) {
    strata::metal_engine::missing("strata::kernels::cpu::kt256_gu_rows");
}
void kt256_down_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) {
    strata::metal_engine::missing("strata::kernels::cpu::kt256_down_rows");
}
}
