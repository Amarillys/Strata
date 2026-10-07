#pragma once
#include "strata/kernels/iq_kernels.hpp"
#include "strata/artifact/kt.hpp"
namespace strata::kernels {
// dst_type: 0=f32, 1=f16, 30=bf16. Row-aware, including compact KT tails.
void kt_dequant_rows(int type, const void* src, int64_t cols, int64_t rows, void* dst,
                     int dst_type, void* stream, int64_t ld = 0, int interleave = 1,
                     const int32_t* tokens = nullptr);
void kt_mmvq(int type, const void* w, const void* x, float* y, int cols, int rows, int tokens, void* stream);
void kt_expert_grouped(const NativeExpertLayout& L, const unsigned long long* ptr, const int32_t* start,
                       const int32_t* count, const int32_t* dst, const int32_t* tok, int64_t groups,
                       int64_t entries, const void* x, void* scratch, float* out, void* stream);
}
