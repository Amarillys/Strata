// src/prefill/kt_mmq_bench.cpp - the KT prompt experts' two paths timed at the engine's shapes, on one GPU.
//
//     kt_mmq_bench [rows per expert ...]
//
// One group of 16 IQ3_KT experts (gate/up 1280 x 2560, down 2560 x 640, synthetic codes with row scales in the model's
// range), R activation rows each, as the prompt path issues them:
//   FP16  - per expert: the expansion to FP16 (iq_dequant_gu_f16 + kt_dequant_rows), then two cuBLAS products;
//   MMQ   - STRATA_PREFILL_KT_MMQ: the group's Q8_0 conversion (kt_to_q8_0_group, one launch), the activations to q8_1,
//           llama.cpp's Q8_0 MMQ for gate/up and down (16 experts a launch), the SwiGLU and H's q8_1 between them.
// Medians of 25 runs (3 warm-ups), and the gate/up outputs of both paths compared (relative L2): the MMQ path's int8
// activations round differently.  Default R: 15 (a 759-token image prompt: 759 x 10 / 512) and 40 (a 2048 chunk).
#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/kt_kernels.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;
namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
void ckb(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "%s: cuBLAS %d\n", what, (int) s); std::exit(1); }
}
constexpr int E = 16, N = 2560, FF = 640, T8 = 8;   // GGML_TYPE_Q8_0 = 8

float median_ms(const std::function<void()>& f, cudaStream_t s) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<float> ts;
    for (int i = 0; i < 28; ++i) {
        cudaEventRecord(a, s);
        f();
        cudaEventRecord(b, s);
        ck(cudaEventSynchronize(b), "run");
        float ms; cudaEventElapsedTime(&ms, a, b);
        if (i >= 3) ts.push_back(ms);
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<int> Rs;
    for (int i = 1; i < argc; ++i) Rs.push_back(std::atoi(argv[i]));
    if (Rs.empty()) Rs = {15, 40};
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    const int t = strata::kt::IQ3;
    const size_t gr = strata::kt::row_bytes(t, N), dr = strata::kt::row_bytes(t, FF);
    const size_t up_off = FF * gr, down_off = 2 * FF * gr, blob = down_off + (size_t) N * dr;
    std::mt19937 rng(11);
    std::vector<uint8_t> hw((size_t) E * blob);
    { std::uniform_int_distribution<int> byte(0, 255); for (auto& b : hw) b = (uint8_t) byte(rng); }
    std::uniform_real_distribution<float> scale(-0.01f, 0.01f);
    for (int e = 0; e < E; ++e) {
        uint8_t* p = hw.data() + (size_t) e * blob;
        for (int r = 0; r < 2 * FF; ++r) { const float v = scale(rng); std::memcpy(p + (size_t) r * gr, &v, 4); }
        for (int r = 0; r < N; ++r) { const float v = scale(rng); std::memcpy(p + down_off + (size_t) r * dr, &v, 4); }
    }
    uint8_t* dw;
    ck(cudaMalloc(&dw, hw.size()), "w");
    ck(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice), "w");
    const uint8_t* blobs[E];
    for (int e = 0; e < E; ++e) blobs[e] = dw + (size_t) e * blob;
    const size_t gub = mmq::matrix_bytes(T8, 2 * FF, N), db = mmq::matrix_bytes(T8, N, FF);
    uint8_t *qgu, *qd; uint16_t *fgu, *fd;
    ck(cudaMalloc(&qgu, E * gub + 4096), "qgu"); ck(cudaMalloc(&qd, E * db + 4096), "qd");
    ck(cudaMalloc(&fgu, (size_t) 2 * 2 * FF * N * 2), "fgu"); ck(cudaMalloc(&fd, (size_t) 2 * N * FF * 2), "fd");
    cudaStream_t s; ck(cudaStreamCreate(&s), "stream");
    cublasHandle_t h; ckb(cublasCreate(&h), "cublas"); ckb(cublasSetStream(h, s), "stream");
    std::printf("kt_mmq_bench: %s, IQ3_KT, %d experts, medians of 25 (ms)\n", prop.name, E);
    for (int R : Rs) {
        const int T = E * R;
        std::vector<float> x((size_t) T * N);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        std::vector<__half> xh(x.size());
        for (size_t i = 0; i < x.size(); ++i) xh[i] = __float2half(x[i]);
        float *dx, *gu_m, *gu_f, *hm, *dm; __half *dxh, *hh; void *xq, *hq; int32_t *ids, *bounds, *rb;
        ck(cudaMalloc(&dx, x.size() * 4), "x"); ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        ck(cudaMalloc(&dxh, xh.size() * 2), "xh"); ck(cudaMemcpy(dxh, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice), "xh");
        ck(cudaMalloc(&gu_m, (size_t) T * 2 * FF * 4), "gu"); ck(cudaMalloc(&gu_f, (size_t) T * 2 * FF * 4), "gu");
        ck(cudaMalloc(&hm, (size_t) T * FF * 4), "h"); ck(cudaMalloc(&hh, (size_t) T * FF * 2), "hh");
        ck(cudaMalloc(&dm, (size_t) T * N * 4), "d");
        ck(cudaMalloc(&xq, mmq::q8_bytes(T, N)), "xq"); ck(cudaMalloc(&hq, mmq::q8_bytes(T, FF)), "hq");
        ck(cudaMalloc(&ids, (size_t) T * 4), "ids"); ck(cudaMalloc(&bounds, (E + 1) * 4), "b"); ck(cudaMalloc(&rb, (E + 1) * 4), "rb");
        std::vector<int32_t> hb(E + 1);
        for (int e = 0; e <= E; ++e) hb[e] = e * R;
        ck(cudaMemcpy(bounds, hb.data(), (E + 1) * 4, cudaMemcpyHostToDevice), "b");
        ck(cudaMemcpy(rb, hb.data(), (E + 1) * 4, cudaMemcpyHostToDevice), "rb");
        mmq::iota(ids, T, s);
        ck(cudaMemset(hh, 0, (size_t) T * FF * 2), "hh");
        mmq::Context ctx;
        mmq::Product pg, pd;
        pg.w = qgu; pg.type = T8; pg.w_rows = 2 * FF; pg.w_cols = N; pg.expert_bytes = gub; pg.n = E; pg.xq = xq;
        pg.bounds = bounds; pg.ids = ids; pg.total_rows = T; pg.max_rows = R; pg.dst = gu_m; pg.ld_dst = 2 * FF;
        pd.w = qd; pd.type = T8; pd.w_rows = N; pd.w_cols = FF; pd.expert_bytes = db; pd.n = E; pd.xq = hq;
        pd.bounds = rb; pd.ids = ids; pd.total_rows = T; pd.max_rows = R; pd.dst = dm; pd.ld_dst = N;
        const float alpha = 1.f, beta = 0.f;
        auto convert = [&] { k::kt_to_q8_0_group(t, t, N, FF, gr, dr, up_off, down_off, blobs, 0, E, qgu, gub, qd, db, 4096, s); };
        auto quant_x = [&] { mmq::quantize(dx, nullptr, xq, T8, N, N, T, s); };
        auto mmq_gu = [&] { ctx.run(pg, s); };
        auto mmq_mid = [&] { mmq::swiglu(gu_m, hm, T, FF, false, s); mmq::quantize(hm, nullptr, hq, T8, FF, FF, T, s); };
        auto mmq_d = [&] { ctx.run(pd, s); };
        auto fp16_dq = [&] {
            for (int e = 0; e < E; ++e) {
                const int q = e % 2;
                k::iq_dequant_gu_f16(t, blobs[e], blobs[e] + up_off, FF, N, fgu + (size_t) q * 2 * FF * N, s);
                k::kt_dequant_rows(t, blobs[e] + down_off, FF, N, fd + (size_t) q * N * FF, 1, s);
            }
        };
        auto fp16_gemm = [&] {
            for (int e = 0; e < E; ++e) {
                const int q = e % 2;
                ckb(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 2 * FF, R, N, &alpha, fgu + (size_t) q * 2 * FF * N,
                                 CUDA_R_16F, N, dxh + (size_t) e * R * N, CUDA_R_16F, N, &beta, gu_f + (size_t) e * R * 2 * FF,
                                 CUDA_R_32F, 2 * FF, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "gemm gu");
                ckb(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, R, FF, &alpha, fd + (size_t) q * N * FF, CUDA_R_16F, FF,
                                 hh + (size_t) e * R * FF, CUDA_R_16F, FF, &beta, dm + (size_t) e * R * N, CUDA_R_32F, N,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "gemm d");
            }
        };
        const float t_conv = median_ms(convert, s), t_qx = median_ms(quant_x, s), t_gu = median_ms(mmq_gu, s),
                    t_mid = median_ms(mmq_mid, s), t_d = median_ms(mmq_d, s);
        const float t_dq = median_ms(fp16_dq, s), t_gemm = median_ms(fp16_gemm, s);
        // gate/up of both paths: MMQ rows are [gate | up], the FP16 expansion interleaves gate and up rows
        convert(); quant_x(); mmq_gu(); fp16_dq();
        ck(cudaStreamSynchronize(s), "sync");
        // the last expert's FP16 weights sit in buffer (E - 1) % 2; recompute its product alone
        const int e = E - 1, q = e % 2;
        ckb(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 2 * FF, R, N, &alpha, fgu + (size_t) q * 2 * FF * N, CUDA_R_16F, N,
                         dxh + (size_t) e * R * N, CUDA_R_16F, N, &beta, gu_f + (size_t) e * R * 2 * FF, CUDA_R_32F, 2 * FF,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "gemm check");
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> a((size_t) R * 2 * FF), b(a.size());
        ck(cudaMemcpy(a.data(), gu_m + (size_t) e * R * 2 * FF, a.size() * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b.data(), gu_f + (size_t) e * R * 2 * FF, b.size() * 4, cudaMemcpyDeviceToHost), "b");
        double num = 0, den = 0;
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < 2 * FF; ++c) {   // MMQ column c: gate (c < FF) or up row; FP16: interleaved 2*j / 2*j+1
                const float m = a[(size_t) r * 2 * FF + c];
                const float f = b[(size_t) r * 2 * FF + (c < FF ? 2 * c : 2 * (c - FF) + 1)];
                num += (double) (m - f) * (m - f); den += (double) f * f;
            }
        std::printf("  R %3d (%4d rows): MMQ path %7.3f = convert %.3f + x q8_1 %.3f + gate/up %.3f + swiglu/h q8_1 %.3f"
                    " + down %.3f | FP16 path %7.3f = expand %.3f + cuBLAS %.3f | gate/up rel L2 %.2e\n",
                    R, T, t_conv + t_qx + t_gu + t_mid + t_d, t_conv, t_qx, t_gu, t_mid, t_d, t_dq + t_gemm, t_dq, t_gemm,
                    std::sqrt(num / std::max(den, 1e-30)));
        cudaFree(dx); cudaFree(dxh); cudaFree(gu_m); cudaFree(gu_f); cudaFree(hm); cudaFree(hh); cudaFree(dm);
        cudaFree(xq); cudaFree(hq); cudaFree(ids); cudaFree(bounds); cudaFree(rb);
    }
    return 0;
}
