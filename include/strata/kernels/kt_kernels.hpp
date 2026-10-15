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
// The prompt path's KT MMQ (STRATA_PREFILL_KT_MMQ=1): the experts blobs[first, n) of an MMQ group as GGUF Q8_0 blocks
// ({f16 d; int8 qs[32]}: the trellis codes as they are, d = row scale * block scale) for llama.cpp's Q8_0 MMQ - gate rows
// then up rows into gu_dst + q * gu_stride, down rows into d_dst + q * d_stride.  Expert n - 1 also zeroes zero_bytes
// after both of its slots (MMQ reads past a 640-value row).  One launch per call.
constexpr int kKtQ8GroupMax = 16;
void kt_to_q8_0_group(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, size_t gu_row, size_t d_row,
                      size_t up_off, size_t down_off, const uint8_t* const* blobs, int first, int n,
                      void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, size_t zero_bytes, void* stream);
}
