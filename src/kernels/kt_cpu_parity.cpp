// AVX2 KT vs the original scalar arithmetic, plus an optional real-expert benchmark.
#include "strata/kernels/cpu/kt_avx2.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/artifact/dequant.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace cpu = strata::kernels::cpu;
namespace kt = strata::kt;
namespace {
void read(std::ifstream& f,void* p,size_t n) { if(!f.read((char*)p,n)) throw std::runtime_error("truncated fixture"); }
float scalar(int type,const uint8_t* w,const uint8_t* a,int n) {
    float sum=0;
    for(int b=0;b<n/32;++b) {
        int8_t q[32];const float d=kt::decode32(type,w,n,b,q);
        int v=0;for(int j=0;j<32;++j)v+=int(q[j])*int((int8_t)a[34*b+2+j]);
        sum+=d*strata::fp16_to_fp32((uint16_t)kt::u16(a+34*b))*v;
    }
    return sum;
}
void reference(const cpu::NativeFmt& f,const uint8_t* blob,const void* const* act,int nt,
               float* const* out,int r0,int r1,bool gu) {
    for(int r=r0;r<r1;++r) for(int t=0;t<nt;++t) {
        const uint8_t* a=(const uint8_t*)act[t];
        if(gu) {
            const float g=scalar(f.gu_type,blob+r*f.gu_row,a,(int)f.n_embd);
            const float u=scalar(f.gu_type,blob+f.up_off+r*f.gu_row,a,(int)f.n_embd);
            out[t][r]=(g/(1.f+std::exp(-g)))*u;
        } else out[t][r]=scalar(f.d_type,blob+f.down_off+r*f.d_row,a,(int)f.n_ff);
    }
}
void exact(const std::vector<float>& a,const std::vector<float>& b,int type,int n,int nt) {
    if(a.size()!=b.size())throw std::runtime_error("size mismatch");
    for(size_t i=0;i<a.size();++i) if(!std::isfinite(a[i]) || a[i]!=b[i]) {
        std::fprintf(stderr,"KT CPU type=%d width=%d tokens=%d i=%zu got=%g ref=%g\n",type,n,nt,i,a[i],b[i]);
        throw std::runtime_error("CPU arithmetic changed");
    }
}
void fixture(const cpu::NativeFmt& f,const std::vector<uint8_t>& raw,int nr) {
    const int n=(int)f.n_embd;
    constexpr int MAXT=9; // 9 also exercises the public wrapper's token tiling.
    const size_t ab=(size_t)n/32*34;
    std::vector<uint8_t> acts(MAXT*ab);
    const void* ap[MAXT];
    for(int t=0;t<MAXT;++t) {
        ap[t]=acts.data()+t*ab;
        for(int b=0;b<n/32;++b) {
            auto* p=acts.data()+t*ab+34*b;p[0]=0;p[1]=0x20; // 1/128 as FP16
            for(int j=0;j<32;++j)p[2+j]=(uint8_t)(t*43+b*17+j*7); // includes -128 and +127
        }
    }
    for(int nt=1;nt<=MAXT;++nt) {
        std::vector<float> got(nt*nr,123.f),want(got);
        float* gp[MAXT];float* rp[MAXT];
        for(int t=0;t<nt;++t) { gp[t]=got.data()+t*nr;rp[t]=want.data()+t*nr; }
        cpu::kt256_down_rows(f,raw.data(),ap,nt,gp,1,nr);
        reference(f,raw.data(),ap,nt,rp,1,nr,false);exact(got,want,f.gu_type,n,nt);
        std::fill(got.begin(),got.end(),123.f);std::fill(want.begin(),want.end(),123.f);
        cpu::kt256_gu_rows(f,raw.data(),ap,nt,gp,0,nr/2);
        reference(f,raw.data(),ap,nt,rp,0,nr/2,true);exact(got,want,f.gu_type,n,nt);
    }
}
void bench(const char* path,int layer) {
    strata::GgufFile model(path);
    const strata::TensorInfo* tensors[3]={};
    const char* roles[]={"gate","up","down"};
    for(const auto& t:model.tensors())for(int i=0;i<3;++i)
        if(t.name=="blk."+std::to_string(layer)+".ffn_"+roles[i]+"_exps.weight")tensors[i]=&t;
    if(!tensors[0]||!tensors[1]||!tensors[2])throw std::runtime_error("expert roles missing in shard");
    cpu::NativeFmt f;std::string err;
    if(!cpu::native_fmt(tensors[0]->type,tensors[2]->type,2560,640,f,err))throw std::runtime_error(err);
    constexpr int NE=16,NT=8;
    std::vector<uint8_t> blobs(NE*f.bytes),acts(NT*f.act_bytes),hs(NT*f.h_bytes);
    const size_t sizes[]={f.up_off,f.up_off,f.bytes-f.down_off},offsets[]={0,f.up_off,f.down_off};
    for(int e=0;e<NE;++e)for(int role=0;role<3;++role)
        std::memcpy(blobs.data()+e*f.bytes+offsets[role],model.tensor_data(*tensors[role])+e*sizes[role],sizes[role]);
    std::vector<float> x(2560),h(640),got(NT*2560),want(got);
    const void* ap[NT];const void* hp[NT];float* gp[NT];float* rp[NT];
    for(int t=0;t<NT;++t) {
        for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i+t)*.173f);
        for(size_t i=0;i<h.size();++i)h[i]=std::cos(float(i+t)*.137f);
        cpu::native_quant_act(f,x.data(),acts.data()+t*f.act_bytes);
        cpu::native_quant_h(f,h.data(),hs.data()+t*f.h_bytes);
        ap[t]=acts.data()+t*f.act_bytes;hp[t]=hs.data()+t*f.h_bytes;
        gp[t]=got.data()+t*2560;rp[t]=want.data()+t*2560;
    }
    for(int nt:{1,3,8}) {
        double ms[2]={};
        for(int fast=0;fast<2;++fast) {
            const auto start=std::chrono::steady_clock::now();
            for(int e=0;e<NE;++e) {
                const auto* b=blobs.data()+e*f.bytes;
                if(fast) {
                    cpu::kt256_gu_rows(f,b,ap,nt,gp,0,640);
                    cpu::kt256_down_rows(f,b,hp,nt,gp,0,2560);
                } else {
                    reference(f,b,ap,nt,rp,0,640,true);
                    reference(f,b,hp,nt,rp,0,2560,false);
                }
            }
            ms[fast]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/NE;
        }
        exact(got,want,f.gu_type,2560,nt);
        std::printf("real expert layer=%d tokens=%d GU+down scalar %.3f ms, AVX2 %.3f ms, %.2fx\n",
                    layer,nt,ms[0],ms[1],ms[0]/ms[1]);
    }
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=2 && argc!=4) { std::fprintf(stderr,"usage: kt_cpu_parity fixture.bin [model.gguf layer]\n");return 2; }
        if(!cpu::cpu_avx2_ok())throw std::runtime_error("AVX2/F16C unavailable");
        std::ifstream f(argv[1],std::ios::binary);uint32_t header[2];read(f,header,8);
        if(header[0]!=0x4B545331)throw std::runtime_error("invalid fixture");
        for(uint32_t i=0;i<header[1];++i) {
            uint32_t s[3];read(f,s,12);const int n=s[1],nr=s[2];
            cpu::NativeFmt fmt;fmt.gu_type=fmt.d_type=s[0];fmt.n_embd=fmt.n_ff=n;
            fmt.gu_row=fmt.d_row=kt::row_bytes(s[0],n);fmt.up_off=(nr/2)*fmt.gu_row;
            std::vector<uint8_t> raw(nr*fmt.gu_row);read(f,raw.data(),raw.size());
            f.seekg((int64_t)n*nr*4,std::ios::cur);
            fixture(fmt,raw,nr);
        }
        std::printf("KT AVX2: %u fixtures x 1..9 tokens, gate/up/down and row ranges exactly match scalar\n",header[1]);
        if(argc==4)bench(argv[2],std::atoi(argv[3]));
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"kt_cpu_parity: %s\n",e.what());return 1; }
}
