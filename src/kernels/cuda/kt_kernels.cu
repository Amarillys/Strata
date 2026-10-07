// Native KT CUDA adapter. Integer trellis follows IK's IQKT CUDA kernels (MIT).
// Copyright (C) 2024 Iwan Kawrakow; see third_party/ik_kt/LICENSE for the adapted code.
#include "strata/kernels/kt_kernels.hpp"
#include "strata/kernels/dp4a.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>

namespace strata::kernels {
namespace {
struct Q8 { __half2 ds; int8_t qs[32]; };
static_assert(sizeof(Q8) == 36);
void check() {
    auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
// Generate four trellis values directly in the byte layout consumed by DP4A.
// IQ3 stores magnitudes and separate signs; IQ4 stores signed trellis values.
// The byte sum itself is another DP4A, replacing four scalar extracts/adds.
template<bool MAGNITUDE> __device__ __forceinline__ uint32_t codes4(uint32_t& seed) {
    uint32_t packed = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        seed *= 0xCBAC1FEDu;
        int v = STRATA_DP4A((int)(seed & 0x3f3f3f3fu), 0x01010101, -126);
        if constexpr (MAGNITUDE) v = abs(v);
        packed |= (uint32_t(v) & 255u) << (8 * k);
    }
    return packed;
}
__device__ __forceinline__ int signed_dot4(uint32_t& seed, uint32_t signs, int x, int sum) {
    const uint32_t v = codes4<true>(seed);
    return STRATA_DP4A((int)__vsub4(v ^ signs, signs), x, sum);
}
// KT rows and all 32-bit fields below are aligned to four bytes, including tails.
__device__ __forceinline__ uint32_t word(const uint8_t* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}
template<int TYPE> __device__ __forceinline__ float dot32(
        const uint8_t* row, int n, int group, const Q8& x) {
    const int b = group / 8, ib = group % 8, nt = n % 256 / 32;
    const bool tail = b == n / 256;
    const uint8_t* p = row + 4 + b * (TYPE == kt::IQ3 ? 100 : 128);
    const int* q8 = reinterpret_cast<const int*>(x.qs);
    int dot = 0, ls;
    const float scale = *reinterpret_cast<const float*>(row);
    if constexpr (TYPE == kt::IQ3) {
        ls = tail ? ((p[12 * nt + ib / 2] >> (4 * (ib & 1))) & 15)
                  : ((p[ib % 4] >> (4 * (ib / 4))) & 15);
        const auto* seeds = reinterpret_cast<const uint16_t*>(p + (tail ? 0 : 4) + 8 * ib);
        const uint32_t mask = 0x01010101u << ib;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t seed = seeds[j] + 4096;
            uint32_t s0, s1;
            if (tail) {
                const uint32_t sb = p[8 * nt + 4 * ib + j];
                s0 = __vcmpne4(((sb & 15u) * 0x00204081u) & 0x01010101u, 0);
                s1 = __vcmpne4(((sb >> 4) * 0x00204081u) & 0x01010101u, 0);
            } else {
                s0 = __vcmpne4(word(p + 68 + 8 * j) & mask, 0);
                s1 = __vcmpne4(word(p + 72 + 8 * j) & mask, 0);
            }
            dot = signed_dot4(seed, s0, q8[2 * j], dot);
            dot = signed_dot4(seed, s1, q8[2 * j + 1], dot);
        }
        return scale * ls * 1.01f * __low2float(x.ds) * dot;
    } else {
        const uint32_t sh = word(p + (tail ? 16 * ib : 4 * ib));
        ls = int((sh & 255) >> 1) - 64;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t lo = p[(tail ? 16 * ib + 4 : 32 + 8 * ib) + j];
            const uint32_t hi = tail ? ((p[16 * ib + 12 + j / 2] >> (4 * (j & 1))) & 15)
                                     : ((p[96 + 8 * (ib % 4) + j] >> (4 * (ib / 4))) & 15);
            uint32_t seed = lo + (hi << 8) + (((sh >> (8 + 3 * j)) & 7) << 12) + ((sh & 1) << 15) + 4096;
            dot = STRATA_DP4A((int)codes4<false>(seed), q8[j], dot);
        }
        return scale * ls * __low2float(x.ds) * dot;
    }
}
template<int TYPE> __device__ float row_dot(const uint8_t* row, int n, const Q8* x) {
    float sum = 0;
    for (int b = threadIdx.x; b < n / 32; b += 32) {
        sum += dot32<TYPE>(row, n, b, x[b]);
    }
    for (int o = 16; o > 0; o /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    return sum;
}
// Store BF16 bits directly, as dequant_bf16.cu does. This also builds through
// Strata's HIP compatibility headers, which do not provide cuda_bf16.h.
struct Bf16 {
    uint16_t bits;
    __device__ explicit Bf16(float f) {
        uint32_t u = __float_as_uint(f);
        bits = (u & 0x7fffffffu) > 0x7f800000u ? uint16_t((u >> 16) | 64u)
             : uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
};
static_assert(sizeof(Bf16) == sizeof(uint16_t));
__device__ __forceinline__ uint16_t bits16(__half v) { return __half_as_ushort(v); }
__device__ __forceinline__ uint16_t bits16(Bf16 v) { return v.bits; }
template<class T> __device__ __forceinline__ void store_codes4(T* dst, float d, uint32_t q) {
    const T a = T(d * int(int8_t(q))), b = T(d * int(int8_t(q >> 8)));
    const T c = T(d * int(int8_t(q >> 16))), e = T(d * int(int8_t(q >> 24)));
    // Prefill's normal rows are aligned. Preserve arbitrary leading dimensions
    // and destination offsets for the row/gather API as well.
    if ((reinterpret_cast<uintptr_t>(dst) & (4 * sizeof(T) - 1)) == 0) {
        if constexpr (std::is_same_v<T, float>) {
            *reinterpret_cast<float4*>(dst) = make_float4(a, b, c, e);
        } else {
            *reinterpret_cast<uint2*>(dst) = make_uint2(
                uint32_t(bits16(a)) | (uint32_t(bits16(b)) << 16),
                uint32_t(bits16(c)) | (uint32_t(bits16(e)) << 16));
        }
    } else {
        dst[0] = a; dst[1] = b; dst[2] = c; dst[3] = e;
    }
}
// Four lanes share a 32-value group. Each reconstructs eight adjacent values,
// without repeating an IQ3 seed's recurrence or staging 32 scalar codes.
template<int TYPE, class T> __global__ void dq(const uint8_t* src, size_t rb, int cols, int rows,
                                    T* dst, int64_t ld, int interleave, const int32_t* tokens) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)rows * (cols / 8)) return;
    const int r = (int)(i / (cols / 8)), chunk = (int)(i % (cols / 8));
    const int group = chunk / 4, j = chunk % 4, b = group / 8, ib = group % 8;
    const int nt = cols % 256 / 32;
    const bool tail = b == cols / 256;
    const uint8_t* row = src + (size_t)(tokens ? tokens[r] : r) * rb;
    const uint8_t* p = row + 4 + b * (TYPE == kt::IQ3 ? 100 : 128);
    const float scale = *reinterpret_cast<const float*>(row);
    T* out = dst + (int64_t)r * ld * interleave + chunk * 8;
    if constexpr (TYPE == kt::IQ3) {
        const int ls = tail ? ((p[12 * nt + ib / 2] >> (4 * (ib & 1))) & 15)
                            : ((p[ib % 4] >> (4 * (ib / 4))) & 15);
        const float d = scale * ls * 1.01f;
        uint32_t seed = *reinterpret_cast<const uint16_t*>(p + (tail ? 0 : 4) + 8 * ib + 2 * j) + 4096;
        uint32_t s0, s1;
        if (tail) {
            const uint32_t sb = p[8 * nt + 4 * ib + j];
            s0 = __vcmpne4(((sb & 15u) * 0x00204081u) & 0x01010101u, 0);
            s1 = __vcmpne4(((sb >> 4) * 0x00204081u) & 0x01010101u, 0);
        } else {
            const uint32_t mask = 0x01010101u << ib;
            s0 = __vcmpne4(word(p + 68 + 8 * j) & mask, 0);
            s1 = __vcmpne4(word(p + 72 + 8 * j) & mask, 0);
        }
        const uint32_t v0 = codes4<true>(seed), v1 = codes4<true>(seed);
        store_codes4(out, d, __vsub4(v0 ^ s0, s0));
        store_codes4(out + 4, d, __vsub4(v1 ^ s1, s1));
    } else {
        const uint32_t sh = word(p + (tail ? 16 * ib : 4 * ib));
        const float d = scale * (int((sh & 255) >> 1) - 64);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int k = 2 * j + h;
            const uint32_t lo = p[(tail ? 16 * ib + 4 : 32 + 8 * ib) + k];
            const uint32_t hi = tail ? ((p[16 * ib + 12 + k / 2] >> (4 * (k & 1))) & 15)
                                     : ((p[96 + 8 * (ib % 4) + k] >> (4 * (ib / 4))) & 15);
            uint32_t seed = lo + (hi << 8) + (((sh >> (8 + 3 * k)) & 7) << 12) + ((sh & 1) << 15) + 4096;
            store_codes4(out + 4 * h, d, codes4<false>(seed));
        }
    }
}
template<int TYPE> __global__ void mv(const uint8_t* w, size_t rb, const Q8* x, float* y, int cols, int rows, int tokens) {
    const int r = blockIdx.x * 4 + threadIdx.y;
    if (r >= rows) return;
    for (int t = 0; t < tokens; ++t) {
        const float v = row_dot<TYPE>(w + (size_t)r * rb, cols, x + (size_t)t * (cols / 32));
        if (threadIdx.x == 0) y[(size_t)t * rows + r] = v;
    }
}
template<bool DOWN, int TYPE> __global__ void expert(NativeExpertLayout L, const unsigned long long* ptr,
        const int32_t* start, const int32_t* count, const int32_t* dst, const int32_t* tok,
        const Q8* x, float* h, const Q8* hq, float* out) {
    const int r = blockIdx.x * 4 + threadIdx.y;
    if (r >= (DOWN ? L.n_embd : L.n_ff)) return;
    for (int g = blockIdx.y; g < *count; g += gridDim.y) {
        const uint8_t* blob = (const uint8_t*)ptr[g];
        for (int e = start[g]; e < start[g + 1]; ++e) {
            if constexpr (DOWN) {
                const float v = row_dot<TYPE>(blob + L.down_off + r * L.d_row, (int)L.n_ff,
                                       hq + (size_t)e * (L.n_ff / 32));
                if (threadIdx.x == 0) out[(size_t)dst[e] * L.n_embd + r] = v;
            } else {
                const Q8* a = x + (size_t)tok[e] * (L.n_embd / 32);
                const float v = row_dot<TYPE>(blob + r * L.gu_row, (int)L.n_embd, a);
                const float u = row_dot<TYPE>(blob + L.up_off + r * L.gu_row, (int)L.n_embd, a);
                if (threadIdx.x == 0) h[(size_t)e * L.n_ff + r] = (v / (1.0f + expf(-v))) * u;
            }
        }
    }
}
}
void kt_dequant_rows(int type, const void* src, int64_t cols, int64_t rows, void* dst,
                     int dst_type, void* stream, int64_t ld, int interleave, const int32_t* tokens) {
    if (!kt::row_bytes(type, cols) || cols > INT32_MAX || rows < 0 || rows > INT32_MAX || interleave < 1)
        throw std::invalid_argument("invalid KT matrix shape");
    if (!rows) return;
    if (!ld) ld = cols;
    if (ld < cols) throw std::invalid_argument("KT output stride is smaller than a row");
    const unsigned grid = (unsigned)((rows * (cols / 8) + 127) / 128);
    const auto s = (cudaStream_t)stream;
#define DQ_IMPL(K,T) dq<K,T><<<grid,128,0,s>>>((const uint8_t*)src,kt::row_bytes(type,cols),(int)cols,(int)rows,(T*)dst,ld,interleave,tokens)
#define DQ(T) do { if (type == kt::IQ3) { DQ_IMPL(kt::IQ3,T); } else { DQ_IMPL(kt::IQ4,T); } } while (0)
    if (dst_type == 0) { DQ(float); }
    else if (dst_type == 1) { DQ(__half); }
    else if (dst_type == 30) { DQ(Bf16); }
    else throw std::invalid_argument("invalid KT dequantization destination");
#undef DQ
#undef DQ_IMPL
    check();
}
void kt_mmvq(int type, const void* w, const void* x, float* y, int cols, int rows, int tokens, void* stream) {
    if (!kt::row_bytes(type, cols) || rows <= 0 || tokens <= 0) throw std::invalid_argument("invalid KT GEMV shape");
#define MV(T) mv<T><<<(rows + 3) / 4,dim3(32,4),0,(cudaStream_t)stream>>>( \
    (const uint8_t*)w,kt::row_bytes(type,cols),(const Q8*)x,y,cols,rows,tokens)
    if (type == kt::IQ3) { MV(kt::IQ3); }
    else { MV(kt::IQ4); }
#undef MV
    check();
}
void kt_expert_grouped(const NativeExpertLayout& L, const unsigned long long* ptr, const int32_t* start,
                       const int32_t* count, const int32_t* dst, const int32_t* tok, int64_t groups,
                       int64_t entries, const void* x, void* scratch, float* out, void* stream) {
    if (groups <= 0 || entries <= 0) return;
    if (!kt::supported(L.gu_type) || !kt::supported(L.d_type))
        throw std::invalid_argument("invalid KT expert type");
    const size_t bytes = (size_t)entries * L.n_ff * sizeof(float), aligned = (bytes + 255) & ~(size_t)255;
    float* h = (float*)((uint8_t*)scratch + 2 * aligned);
    Q8* hq = (Q8*)((uint8_t*)scratch + 3 * aligned);
    auto s = (cudaStream_t)stream;
    cudaMemsetAsync(h,0,bytes,s);
#define EXPERT(DOWN,T,N) expert<DOWN,T><<<dim3((unsigned)(((N)+3)/4),(unsigned)groups),dim3(32,4),0,s>>>( \
    L,ptr,start,count,dst,tok,(const Q8*)x,h,hq,out)
    if (L.gu_type == kt::IQ3) { EXPERT(false,kt::IQ3,L.n_ff); }
    else { EXPERT(false,kt::IQ4,L.n_ff); }
    quantize_q8_1_rows(h,entries,L.n_ff,hq,stream);
    if (L.d_type == kt::IQ3) { EXPERT(true,kt::IQ3,L.n_embd); }
    else { EXPERT(true,kt::IQ4,L.n_embd); }
#undef EXPERT
    check();
}
}
