// KT trellis lookup + AVX2 signed integer dots, reusing each group across tokens.
// The 16-bit trellis seeds have only 65536 possible reconstructions: 512 KiB
// of IQ3 magnitudes and 256 KiB of IQ4 signed codes, shared by all pool workers.
#include "strata/kernels/cpu/kt_avx2.hpp"
#include "strata/artifact/kt.hpp"
#include <immintrin.h>
#include <cmath>
#include <cstring>

namespace strata::kernels::cpu {
namespace {
struct Tables {
    alignas(64) uint64_t iq3[65536];
    alignas(64) uint32_t iq4[65536];
    Tables() {
        for (uint32_t seed = 0; seed < 65536; ++seed) {
            uint32_t v = seed + 4096;
            uint64_t a = 0;
            uint32_t b = 0;
            for (int j = 0; j < 8; ++j) {
                const int q = kt::next(v);
                a |= uint64_t(q < 0 ? -q : q) << (8*j);
                if (j < 4) b |= uint32_t(uint8_t(q)) << (8*j);
            }
            iq3[seed] = a;
            iq4[seed] = b;
        }
    }
};
const Tables& tables() { static const Tables t; return t; }
uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v,p,8); return v; }

__m256i signs(uint32_t m) {
    const __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int)m),
        _mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3));
    const __m256i bits = _mm256_setr_epi8(1,2,4,8,16,32,64,(char)128,1,2,4,8,16,32,64,(char)128,
                                          1,2,4,8,16,32,64,(char)128,1,2,4,8,16,32,64,(char)128);
    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(bytes,bits),bits),_mm256_set1_epi8(1));
}

template<int TYPE> __m256i decode(const Tables& table, const uint8_t* row, int n, int group, float& scale) {
    const int b=group/8, ib=group%8, nt=n%256/32;
    const bool tail=b==n/256;
    const uint8_t* p=row+4+b*(TYPE==kt::IQ3 ? 100 : 128);
    if constexpr (TYPE==kt::IQ3) {
        const int ls=tail ? (p[12*nt+ib/2]>>(4*(ib&1)))&15 : (p[ib%4]>>(4*(ib/4)))&15;
        const auto* seeds=p+(tail?0:4)+8*ib;
        const __m256i codes=_mm256_set_epi64x((long long)table.iq3[kt::u16(seeds+6)],
            (long long)table.iq3[kt::u16(seeds+4)],(long long)table.iq3[kt::u16(seeds+2)],
            (long long)table.iq3[kt::u16(seeds)]);
        uint32_t mask=tail ? kt::u32(p+8*nt+4*ib) : 0;
        if (!tail) for(int j=0;j<4;++j) {
            const uint64_t column=(u64(p+68+8*j)>>ib)&0x0101010101010101ull;
            mask |= uint32_t((column*0x0102040810204080ull)>>56)<<(8*j);
        }
        scale=kt::f32(row)*ls*1.01f;
        return _mm256_sign_epi8(codes,signs(mask));
    } else {
        const uint32_t sh=kt::u32(p+(tail?16*ib:4*ib));
        alignas(32) uint32_t codes[8];
        for(int j=0;j<8;++j) {
            const uint32_t lo=p[(tail?16*ib+4:32+8*ib)+j];
            const uint32_t hi=tail ? (p[16*ib+12+j/2]>>(4*(j&1)))&15 : (p[96+8*(ib%4)+j]>>(4*(ib/4)))&15;
            const uint32_t seed=lo+(hi<<8)+(((sh>>(8+3*j))&7)<<12)+((sh&1)<<15);
            codes[j]=table.iq4[seed];
        }
        scale=kt::f32(row)*(int((sh&255)>>1)-64);
        return _mm256_load_si256((const __m256i*)codes);
    }
}

// Widen both signed operands: exact even if a caller supplies -128 activations.
// Each 32-value integer sum fits int32; no saturating maddubs intermediate.
int dot(__m256i w, const uint8_t* a) {
    const __m256i x=_mm256_loadu_si256((const __m256i*)(a+2));
    const __m256i lo=_mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(w)),
                                     _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x)));
    const __m256i hi=_mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(w,1)),
                                     _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x,1)));
    const __m256i s=_mm256_add_epi32(lo,hi);
    __m128i h=_mm_add_epi32(_mm256_castsi256_si128(s),_mm256_extracti128_si256(s,1));
    h=_mm_hadd_epi32(h,h);h=_mm_hadd_epi32(h,h);
    return _mm_cvtsi128_si32(h);
}
float h2f(const uint8_t* a) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)kt::u16(a)))); }

template<int TYPE,int NT,bool GU> void rows(const NativeFmt& f,const uint8_t* blob,const void* const* act,
                                            float* const* out,int r0,int r1) {
    const auto& table=tables();
    const int n=(int)(GU?f.n_embd:f.n_ff);
    const size_t rb=GU?f.gu_row:f.d_row;
    for(int r=r0;r<r1;++r) {
        const uint8_t* w=blob+(GU?0:f.down_off)+r*rb;
        float sums[NT]={},ups[NT]={};
        for(int b=0;b<n/32;++b) {
            float scale=0,uscale=0;
            const __m256i q=decode<TYPE>(table,w,n,b,scale);
            __m256i u=_mm256_setzero_si256();
            if constexpr(GU) u=decode<TYPE>(table,w+f.up_off,n,b,uscale);
            for(int t=0;t<NT;++t) {
                const auto* a=(const uint8_t*)act[t]+34*b;
                const float ad=h2f(a);
                sums[t]+=scale*ad*dot(q,a);
                if constexpr(GU) ups[t]+=uscale*ad*dot(u,a);
            }
        }
        for(int t=0;t<NT;++t) {
            if constexpr(GU) out[t][r]=(sums[t]/(1.f+std::exp(-sums[t])))*ups[t];
            else out[t][r]=sums[t];
        }
    }
}
template<int TYPE,bool GU> void dispatch(const NativeFmt& f,const uint8_t* blob,const void* const* act,
                                         int nt,float* const* out,int r0,int r1) {
    while(nt>8) { rows<TYPE,8,GU>(f,blob,act,out,r0,r1);act+=8;out+=8;nt-=8; }
#define RUN(N) case N: rows<TYPE,N,GU>(f,blob,act,out,r0,r1);break
    switch(nt) { RUN(1);RUN(2);RUN(3);RUN(4);RUN(5);RUN(6);RUN(7);RUN(8); }
#undef RUN
}
}
void kt256_gu_rows(const NativeFmt& f,const uint8_t* blob,const void* const* act,int nt,float* const* ff,int r0,int r1) {
    if(f.gu_type==kt::IQ3) dispatch<kt::IQ3,true>(f,blob,act,nt,ff,r0,r1);
    else dispatch<kt::IQ4,true>(f,blob,act,nt,ff,r0,r1);
}
void kt256_down_rows(const NativeFmt& f,const uint8_t* blob,const void* const* act,int nt,float* const* out,int r0,int r1) {
    if(f.d_type==kt::IQ3) dispatch<kt::IQ3,false>(f,blob,act,nt,out,r0,r1);
    else dispatch<kt::IQ4,false>(f,blob,act,nt,out,r0,r1);
}
}
