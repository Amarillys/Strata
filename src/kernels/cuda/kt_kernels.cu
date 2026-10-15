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
// One 32-value group's eight values at lane j (as dq): the signed codes, four per word, and the group's scale.
template<int TYPE> __device__ __forceinline__ float codes8(const uint8_t* row, int cols, int group, int j,
                                                          uint32_t& w0, uint32_t& w1) {
    const int b = group / 8, ib = group % 8, nt = cols % 256 / 32;
    const bool tail = b == cols / 256;
    const uint8_t* p = row + 4 + b * (TYPE == kt::IQ3 ? 100 : 128);
    const float scale = *reinterpret_cast<const float*>(row);
    if constexpr (TYPE == kt::IQ3) {
        const int ls = tail ? ((p[12 * nt + ib / 2] >> (4 * (ib & 1))) & 15)
                            : ((p[ib % 4] >> (4 * (ib / 4))) & 15);
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
        w0 = __vsub4(v0 ^ s0, s0);
        w1 = __vsub4(v1 ^ s1, s1);
        return scale * ls * 1.01f;
    } else {
        const uint32_t sh = word(p + (tail ? 16 * ib : 4 * ib));
        uint32_t w[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int k = 2 * j + h;
            const uint32_t lo = p[(tail ? 16 * ib + 4 : 32 + 8 * ib) + k];
            const uint32_t hi = tail ? ((p[16 * ib + 12 + k / 2] >> (4 * (k & 1))) & 15)
                                     : ((p[96 + 8 * (ib % 4) + k] >> (4 * (ib / 4))) & 15);
            uint32_t seed = lo + (hi << 8) + (((sh >> (8 + 3 * k)) & 7) << 12) + ((sh & 1) << 15) + 4096;
            w[h] = codes4<false>(seed);
        }
        w0 = w[0];
        w1 = w[1];
        return scale * (int((sh & 255) >> 1) - 64);
    }
}
// kt_to_q8_0_group: a CTA converts kQ8Blocks consecutive Q8_0 blocks of one expert's gate/up or down region (four
// lanes per block, eight values each, as dq) into shared memory, then stores them as 16-byte words: 64 blocks are 136
// of them, so every CTA's output starts aligned.  The 2-byte stores straight to global memory this replaced (a block is
// 34 bytes) were measured 1.3-5.5x slower than the FP16 expansion on the 4080 / 2080 Ti.  After the gate/up and down
// CTAs, two more (expert n - 1 only) zero nz uint4 after each of its slots.
constexpr int kQ8Blocks = 64;
struct KtGroup { const uint8_t* blob[kKtQ8GroupMax]; };
// (__grid_constant__: g.blob[q] is read from parameter space; a by-value struct indexed at run time is otherwise copied
// to every thread's local memory.  An expert's blocks are below 2^31, so the block arithmetic is 32-bit.)
template<int GU, int DT> __global__ void __launch_bounds__(4 * kQ8Blocks) to_q8_0(const __grid_constant__ KtGroup g,
        int first, int n, int n_embd, int n_ff, size_t gu_row, size_t d_row, size_t up_off, size_t down_off,
        uint8_t* gu_dst, size_t gu_stride, uint8_t* d_dst, size_t d_stride, int64_t nz) {
    __shared__ uint4 sm[kQ8Blocks * 34 / 16];
    const int q = first + (int)blockIdx.y;
    const int gu_blocks = 2 * n_ff * (n_embd / 32), d_blocks = n_embd * (n_ff / 32);
    const int gu_ctas = (gu_blocks + kQ8Blocks - 1) / kQ8Blocks, d_ctas = (d_blocks + kQ8Blocks - 1) / kQ8Blocks;
    const int c = (int)blockIdx.x;
    if (c >= gu_ctas + d_ctas) {
        if (q != n - 1) return;
        uint4* z = reinterpret_cast<uint4*>(c > gu_ctas + d_ctas ? d_dst + (size_t)n * d_stride
                                                                 : gu_dst + (size_t)n * gu_stride);
        for (int64_t k = threadIdx.x; k < nz; k += blockDim.x) z[k] = make_uint4(0, 0, 0, 0);
        return;
    }
    const bool down = c >= gu_ctas;
    const int cols = down ? n_ff : n_embd, bpr = cols / 32;
    const int b0 = (down ? c - gu_ctas : c) * kQ8Blocks;
    const int nb = min(kQ8Blocks, (down ? d_blocks : gu_blocks) - b0);
    uint8_t* s = reinterpret_cast<uint8_t*>(sm);
    const int b = threadIdx.x / 4, j = threadIdx.x % 4;
    if (b < nb) {
        const uint8_t* blob = g.blob[q];
        const int blk = b0 + b, r = blk / bpr;
        const int group = blk % bpr;
        uint32_t w0, w1;
        float d;
        if (down) {
            d = codes8<DT>(blob + down_off + (size_t)r * d_row, cols, group, j, w0, w1);
        } else {
            const uint8_t* row = r < n_ff ? blob + (size_t)r * gu_row : blob + up_off + (size_t)(r - n_ff) * gu_row;
            d = codes8<GU>(row, cols, group, j, w0, w1);
        }
        uint16_t* qs = reinterpret_cast<uint16_t*>(s + b * 34 + 2 + 8 * j);
        qs[0] = (uint16_t)w0; qs[1] = (uint16_t)(w0 >> 16); qs[2] = (uint16_t)w1; qs[3] = (uint16_t)(w1 >> 16);
        if (j == 0) *reinterpret_cast<__half*>(s + b * 34) = __float2half(d);
    }
    __syncthreads();
    uint8_t* dst = (down ? d_dst + q * d_stride : gu_dst + q * gu_stride) + (size_t)b0 * 34;
    if (nb == kQ8Blocks) {
        for (int k = threadIdx.x; k < kQ8Blocks * 34 / 16; k += blockDim.x) reinterpret_cast<uint4*>(dst)[k] = sm[k];
    } else {
        for (int k = threadIdx.x; k < nb * 34; k += blockDim.x) dst[k] = s[k];
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
void kt_to_q8_0_group(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, size_t gu_row, size_t d_row,
                      size_t up_off, size_t down_off, const uint8_t* const* blobs, int first, int n,
                      void* gu_dst, size_t gu_stride, void* d_dst, size_t d_stride, size_t zero_bytes, void* stream) {
    if (!kt::supported(gu_type) || !kt::supported(d_type) || n_embd <= 0 || n_ff <= 0 || n_embd > INT32_MAX ||
        n_ff > INT32_MAX || gu_row != kt::row_bytes(gu_type, n_embd) || d_row != kt::row_bytes(d_type, n_ff))
        throw std::invalid_argument("invalid KT expert layout for Q8_0");
    if (first < 0 || n <= first || n > kKtQ8GroupMax) throw std::invalid_argument("invalid KT group");
    if (((uintptr_t)gu_dst | (uintptr_t)d_dst | gu_stride | d_stride | zero_bytes) % 16 != 0)
        throw std::invalid_argument("KT Q8_0 group buffers must be 16-byte aligned");
    if (gu_stride < (size_t)(2 * n_ff) * (size_t)(n_embd / 32) * 34 || d_stride < (size_t)n_embd * (size_t)(n_ff / 32) * 34)
        throw std::invalid_argument("KT Q8_0 group slots are too small");
    KtGroup g{};
    for (int q = first; q < n; ++q) g.blob[q] = blobs[q];
    const int64_t nz = (int64_t)(zero_bytes / 16);
    const int64_t ctas = (2 * n_ff * (n_embd / 32) + kQ8Blocks - 1) / kQ8Blocks +
                         (n_embd * (n_ff / 32) + kQ8Blocks - 1) / kQ8Blocks + (nz > 0 ? 2 : 0);
    const dim3 grid((unsigned)ctas, (unsigned)(n - first));
    const auto s = (cudaStream_t)stream;
#define TOQ8(GU,DT) to_q8_0<GU,DT><<<grid,4 * kQ8Blocks,0,s>>>(g,first,n,(int)n_embd,(int)n_ff,gu_row,d_row,up_off, \
    down_off,(uint8_t*)gu_dst,gu_stride,(uint8_t*)d_dst,d_stride,nz)
    if (gu_type == kt::IQ3) { if (d_type == kt::IQ3) TOQ8(kt::IQ3, kt::IQ3); else TOQ8(kt::IQ3, kt::IQ4); }
    else { if (d_type == kt::IQ3) TOQ8(kt::IQ4, kt::IQ3); else TOQ8(kt::IQ4, kt::IQ4); }
#undef TOQ8
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
