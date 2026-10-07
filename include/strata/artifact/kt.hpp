// IQ3_KT / IQ4_KT row layouts and trellis reconstruction from ik_llama.cpp
// c68d4e3c8 (ggml/src/iqk/iqk_quantize.cpp and CUDA IQKT kernels), MIT.
// Copyright (C) 2024 Iwan Kawrakow; see third_party/ik_kt/LICENSE.
// The on-disk ids are IK extensions, deliberately kept out of upstream ggml's enum.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_KT_HD __host__ __device__
#else
#define STRATA_KT_HD
#endif
namespace strata::kt {
inline constexpr int IQ3 = 154, IQ4 = 155;
STRATA_KT_HD inline bool supported(int t) { return t == IQ3 || t == IQ4; }
STRATA_KT_HD inline size_t row_bytes(int t, int64_t n) {
    if (!supported(t) || n <= 0 || n % 32) return 0;
    const size_t nt = (size_t)(n % 256) / 32;
    return (4 + (size_t)(n / 256) * (t == IQ3 ? 100 : 128) +
            (t == IQ3 ? (nt + 1) / 2 + 12 * nt : 16 * nt) + 3) & ~(size_t)3;
}
STRATA_KT_HD inline uint32_t u16(const uint8_t* p) { return p[0] | (uint32_t(p[1]) << 8); }
STRATA_KT_HD inline uint32_t u32(const uint8_t* p) { return u16(p) | (u16(p + 2) << 16); }
STRATA_KT_HD inline float f32(const uint8_t* p) {
    const uint32_t bits = u32(p);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float(bits);
#else
    float v; std::memcpy(&v, &bits, sizeof(v)); return v;
#endif
}
STRATA_KT_HD inline int next(uint32_t& v) {
    v *= 0xCBAC1FEDu;
    return int(v & 63) + int((v >> 8) & 63) + int((v >> 16) & 63) + int((v >> 24) & 63) - 126;
}
// Reconstruct one 32-element group as signed integer codes and its floating scale.
// Every row has a float scale; a final partial 256-element block uses a different layout.
STRATA_KT_HD inline float decode32(int t, const uint8_t* row, int n, int group, int8_t* q) {
    const int b = group / 8, ib = group % 8, nt = n % 256 / 32;
    const bool tail = b == n / 256;
    const uint8_t* p = row + 4 + b * (t == IQ3 ? 100 : 128);
    int ls;
    if (t == IQ3) {
        ls = tail ? ((p[12 * nt + ib / 2] >> (4 * (ib & 1))) & 15)
                  : ((p[ib % 4] >> (4 * (ib / 4))) & 15);
        for (int j = 0; j < 4; ++j) {
            uint32_t v = u16(p + (tail ? 0 : 4) + 8 * ib + 2 * j) + 4096;
            for (int k = 0; k < 8; ++k) {
                int a = next(v); if (a < 0) a = -a;
                const bool sign = tail ? ((p[8 * nt + 4 * ib + j] >> k) & 1)
                                       : ((p[68 + 8 * j + k] >> ib) & 1);
                q[8 * j + k] = (int8_t)(sign ? -a : a);
            }
        }
        return f32(row) * ls * 1.01f;
    }
    const uint32_t sh = u32(p + (tail ? 16 * ib : 4 * ib));
    ls = int((sh & 255) >> 1) - 64;
    for (int j = 0; j < 8; ++j) {
        const uint32_t lo = p[(tail ? 16 * ib + 4 : 32 + 8 * ib) + j];
        const uint32_t hi = tail ? ((p[16 * ib + 12 + j / 2] >> (4 * (j & 1))) & 15)
                                 : ((p[96 + 8 * (ib % 4) + j] >> (4 * (ib / 4))) & 15);
        uint32_t v = lo + (hi << 8) + (((sh >> (8 + 3 * j)) & 7) << 12) + ((sh & 1) << 15) + 4096;
        for (int k = 0; k < 4; ++k) q[4 * j + k] = (int8_t)next(v);
    }
    return f32(row) * ls;
}
inline void dequantize_row(int t, const uint8_t* row, int n, float* out) {
    for (int b = 0; b < n / 32; ++b) {
        int8_t q[32]; const float d = decode32(t, row, n, b, q);
        for (int j = 0; j < 32; ++j) out[32 * b + j] = d * q[j];
    }
}
} // namespace strata::kt
#undef STRATA_KT_HD
