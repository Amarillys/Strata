#pragma once
#include "strata/kernels/cpu/native_expert.hpp"

namespace strata::kernels::cpu {
// Call only after cpu_avx2_ok(). Q8_0 activations; same integer dot and float
// accumulation order as the scalar KT adapter, sharing decode across tokens.
void kt256_gu_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act,
                   int nt, float* const* ff, int r0, int r1);
void kt256_down_rows(const NativeFmt& f, const uint8_t* blob, const void* const* act,
                     int nt, float* const* out, int r0, int r1);
}
