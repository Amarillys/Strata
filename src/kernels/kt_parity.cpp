// Oracle fixtures are produced by tools/test_kt.py through IK's existing DLL.
#include "strata/kernels/kt_kernels.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/ngram.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>

namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p,n)); }
    ~Dev() { cudaFree(p); }
};
void read(std::ifstream& f, void* p, size_t n) { if (!f.read((char*)p,n)) throw std::runtime_error("truncated fixture"); }
float error(const std::vector<float>& a,const std::vector<float>& b) {
    double diff=0,norm=0;
    for(size_t i=0;i<a.size();++i) {
        if(!std::isfinite(a[i])) throw std::runtime_error("non-finite result");
        diff=std::max(diff,std::abs(double(a[i])-b[i])); norm=std::max(norm,std::abs(double(b[i])));
    }
    return (float)(diff/std::max(norm,1e-20));
}
void require(float e,float limit,const char* what,int type,int cols) {
    if(e>limit) { std::fprintf(stderr,"%s type=%d width=%d rel=%g\n",what,type,cols,e); throw std::runtime_error("parity failed"); }
}
float ref_dot(const float* w,const uint8_t* q,int n,int block_bytes,int offset) {
    double sum=0;
    for(int c=0;c<n;++c) {
        const auto* a=q+(c/32)*block_bytes;
        sum+=double(w[c])*strata::fp16_to_fp32(strata::kt::u16(a))*int((int8_t)a[offset+c%32]);
    }
    return (float)sum;
}
void grouped(int type,const std::vector<uint8_t>& gu,const std::vector<float>& gu_ref,
             const std::vector<uint8_t>& dn,const std::vector<float>& dn_ref,cudaStream_t stream) {
    namespace k=strata::kernels; namespace cpu=k::cpu;
    constexpr int N=64,FF=32,NT=4,NE=3;
    auto L=k::native_expert_layout(type,type,N,FF);
    std::vector<uint8_t> blob=gu;blob.insert(blob.end(),dn.begin(),dn.end());
    if(blob.size()!=L.bytes) throw std::runtime_error("fixture layout mismatch");
    std::vector<float> x(N*NT);for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.17f);
    Dev w(blob.size()),dx(x.size()*4),qx(N/32*36*NT),scratch(k::native_expert_scratch_bytes(NE,FF)),out(N*NT*4);
    ck(cudaMemcpy(w.p,blob.data(),blob.size(),cudaMemcpyHostToDevice));ck(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    const unsigned long long ptrs[]={ (unsigned long long)w.p,(unsigned long long)w.p,(unsigned long long)w.p };
    // Repeated tokens, permuted destinations, two active groups and an empty final group.
    const int starts[]={0,2,3,3},toks[]={3,0,3},dsts[]={2,0,1},count=3;
    Dev dp(sizeof(ptrs)),ds(sizeof(starts)),dt(sizeof(toks)),dd(sizeof(dsts)),dc(4);
    ck(cudaMemcpy(dp.p,ptrs,sizeof(ptrs),cudaMemcpyHostToDevice));ck(cudaMemcpy(ds.p,starts,sizeof(starts),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dt.p,toks,sizeof(toks),cudaMemcpyHostToDevice));ck(cudaMemcpy(dd.p,dsts,sizeof(dsts),cudaMemcpyHostToDevice));
    ck(cudaMemcpy(dc.p,&count,4,cudaMemcpyHostToDevice));ck(cudaMemset(out.p,0,N*NT*4));
    k::quantize_q8_1_rows((float*)dx.p,NT,N,qx.p,stream);
    k::native_expert_grouped(L,(const unsigned long long*)dp.p,(const int32_t*)ds.p,(const int32_t*)dc.p,
        (const int32_t*)dd.p,(const int32_t*)dt.p,3,NE,qx.p,scratch.p,(float*)out.p,stream,1);
    ck(cudaStreamSynchronize(stream));
    std::vector<uint8_t> q(N/32*36*NT),hq(NE*FF/32*36);
    std::vector<float> h(NE*FF),want_h(h.size()),got(N*NT),want(got.size(),0);
    const size_t fa=(NE*FF*4+255)&~size_t(255);
    ck(cudaMemcpy(q.data(),qx.p,q.size(),cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(h.data(),(uint8_t*)scratch.p+2*fa,h.size()*4,cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(hq.data(),(uint8_t*)scratch.p+3*fa,hq.size(),cudaMemcpyDeviceToHost));
    for(int e=0;e<NE;++e) {
        const auto* a=q.data()+toks[e]*(N/32)*36;
        for(int r=0;r<FF;++r) {
            float g=ref_dot(gu_ref.data()+r*N,a,N,36,4),u=ref_dot(gu_ref.data()+(r+FF)*N,a,N,36,4);
            want_h[e*FF+r]=(g/(1+std::exp(-g)))*u;
        }
        for(int r=0;r<N;++r) want[dsts[e]*N+r]=ref_dot(dn_ref.data()+r*FF,hq.data()+e*(FF/32)*36,FF,36,4);
    }
    ck(cudaMemcpy(got.data(),out.p,got.size()*4,cudaMemcpyDeviceToHost));
    require(error(h,want_h),8e-6f,"grouped gate/up",type,N);
    require(error(got,want),5e-6f,"grouped down",type,FF);
    cpu::NativeFmt cf;std::string err;
    if(!cpu::native_fmt(type,type,N,FF,cf,err)) throw std::runtime_error(err);
    std::vector<uint8_t> cq(cf.act_bytes),ch(cf.h_bytes);std::vector<float> cpu_h(FF),cpu_out(N),cpu_ref(FF);
    cpu::native_quant_act(cf,x.data(),cq.data());const void* acts[]={cq.data()};float* ffs[]={cpu_h.data()};
    cpu::native_gu_rows(cf,blob.data(),acts,1,ffs,0,FF);
    for(int r=0;r<FF;++r) {
        float g=ref_dot(gu_ref.data()+r*N,cq.data(),N,34,2),u=ref_dot(gu_ref.data()+(r+FF)*N,cq.data(),N,34,2);
        cpu_ref[r]=(g/(1+std::exp(-g)))*u;
    }
    require(error(cpu_h,cpu_ref),8e-6f,"CPU expert gate/up",type,N);
    cpu::native_quant_h(cf,cpu_h.data(),ch.data());const void* hs[]={ch.data()};float* outs[]={cpu_out.data()};
    cpu::native_down_rows(cf,blob.data(),hs,1,outs,0,N);cpu_ref.resize(N);
    for(int r=0;r<N;++r)cpu_ref[r]=ref_dot(dn_ref.data()+r*FF,ch.data(),FF,34,2);
    require(error(cpu_out,cpu_ref),5e-6f,"CPU expert down",type,FF);
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=3) { std::fprintf(stderr,"usage: kt_parity fixture.bin device\n"); return 2; }
        ck(cudaSetDevice(std::atoi(argv[2])));
        cudaStream_t stream; ck(cudaStreamCreate(&stream));
        std::ifstream f(argv[1],std::ios::binary);
        uint32_t header[2];read(f,header,sizeof(header));
        if(header[0]!=0x4B545331) throw std::runtime_error("invalid fixture");
        float worst_dq=0,worst_mv=0;
        std::map<int,std::vector<uint8_t>> down_raw;
        std::map<int,std::vector<float>> down_ref;
        for(uint32_t test=0;test<header[1];++test) {
            uint32_t shape[3];read(f,shape,sizeof(shape));
            const int type=shape[0],n=shape[1],nr=shape[2];const size_t rb=strata::kt::row_bytes(type,n);
            std::vector<uint8_t> raw(rb*nr); std::vector<float> ref((size_t)n*nr),got(ref.size());
            read(f,raw.data(),raw.size());read(f,ref.data(),ref.size()*4);
            if(n==32 && nr==64) { down_raw[type]=raw;down_ref[type]=ref; }
            if(n==64 && nr==64) grouped(type,raw,ref,down_raw.at(type),down_ref.at(type),stream);
            for(int r=0;r<nr;++r) strata::kt::dequantize_row(type,raw.data()+r*rb,n,got.data()+(size_t)r*n);
            require(error(got,ref),4e-7f,"CPU decode",type,n);
            Dev w(raw.size()),out(ref.size()*4);
            ck(cudaMemcpy(w.p,raw.data(),raw.size(),cudaMemcpyHostToDevice));
            strata::kernels::kt_dequant_rows(type,w.p,n,nr,out.p,0,stream);
            ck(cudaStreamSynchronize(stream));ck(cudaMemcpy(got.data(),out.p,got.size()*4,cudaMemcpyDeviceToHost));
            float e=error(got,ref);require(e,5e-7f,"GPU decode",type,n);worst_dq=std::max(e,worst_dq);
            // Vector stores must handle offset destinations, odd leading dimensions
            // and interleaved rows without touching their padding. Check exact bits
            // against the original scalar decoder with the destination's rounding.
            std::vector<float> scalar(ref.size());
            for(int r=0;r<nr;++r) strata::kt::dequantize_row(type,raw.data()+r*rb,n,scalar.data()+(size_t)r*n);
            for(int dt : {0,1,30}) for(int pad : {0,3}) {
                const size_t es=dt==0?4:2, stride=2*(n+pad), prefix=pad?1:0;
                const size_t size=(prefix+nr*stride+1)*es;
                Dev checked(size);
                ck(cudaMemset(checked.p,0x5a,size));
                strata::kernels::kt_dequant_rows(type,w.p,n,nr,(uint8_t*)checked.p+prefix*es,
                                                dt,stream,n+pad,2);
                ck(cudaStreamSynchronize(stream));
                std::vector<uint8_t> actual(size),expected(size,0x5a);
                ck(cudaMemcpy(actual.data(),checked.p,size,cudaMemcpyDeviceToHost));
                for(int r=0;r<nr;++r) for(int c=0;c<n;++c) {
                    const float v=scalar[(size_t)r*n+c];
                    auto* p=expected.data()+(prefix+r*stride+c)*es;
                    if(dt==0) std::memcpy(p,&v,4);
                    else {
                        const uint16_t bits=dt==1?__half_as_ushort(__float2half_rn(v))
                                                  :__bfloat16_as_ushort(__float2bfloat16_rn(v));
                        std::memcpy(p,&bits,2);
                    }
                }
                if(actual!=expected) throw std::runtime_error("KT exact decode/stride/padding mismatch");
            }
            // The actual PLE dispatch, including its 160-wide tail and stored row stride.
            if (n == 160) {
                const auto* fmt = strata::kernels::ple_format_for_type(type == strata::kt::IQ3 ? "IQ3_KT" : "IQ4_KT");
                if (!fmt || fmt->row_bytes != rb) throw std::runtime_error("PLE KT row size mismatch");
                for (int r=0;r<nr;++r) fmt->dequant(raw.data()+r*rb,1.f,got.data()+r*n);
                require(error(got,ref),5e-7f,"PLE decode",type,n);
            }
            // Token embedding lookup must preserve repeated and permuted row indices.
            const int32_t ids[] = {nr-1,0,nr/2,nr-1};
            Dev indices(sizeof(ids)),gather(4*n*sizeof(float));
            ck(cudaMemcpy(indices.p,ids,sizeof(ids),cudaMemcpyHostToDevice));
            strata::kernels::iq_embed_rows(type,w.p,rb,(int32_t*)indices.p,4,n,(float*)gather.p,stream);
            ck(cudaStreamSynchronize(stream));
            std::vector<float> gathered(4*n),gather_ref(4*n);
            ck(cudaMemcpy(gathered.data(),gather.p,gathered.size()*4,cudaMemcpyDeviceToHost));
            for(int r=0;r<4;++r) std::copy_n(ref.data()+ids[r]*n,n,gather_ref.data()+r*n);
            require(error(gathered,gather_ref),5e-7f,"embedding gather",type,n);
            // Prefill's FP16 output uses a padded leading dimension; padding stays untouched.
            const int ld=n+16;
            Dev half_out((size_t)nr*ld*2);
            ck(cudaMemset(half_out.p,0x5a,(size_t)nr*ld*2));
            if(!strata::kernels::dequant_f16_ld(type,w.p,0,nr,n,ld,(uint16_t*)half_out.p,stream))
                throw std::runtime_error("KT FP16 adapter unavailable");
            ck(cudaStreamSynchronize(stream));
            std::vector<uint16_t> half((size_t)nr*ld);
            ck(cudaMemcpy(half.data(),half_out.p,half.size()*2,cudaMemcpyDeviceToHost));
            for(int r=0;r<nr;++r) {
                for(int c=0;c<n;++c) got[r*n+c]=strata::fp16_to_fp32(half[r*ld+c]);
                for(int c=n;c<ld;++c) if(half[r*ld+c]!=0x5a5a) throw std::runtime_error("FP16 padding overwritten");
            }
            require(error(got,ref),6e-4f,"prefill FP16 stride",type,n);
            strata::kernels::dequant_bf16(type,w.p,0,nr,n,(uint16_t*)half_out.p,stream);
            ck(cudaStreamSynchronize(stream));
            ck(cudaMemcpy(half.data(),half_out.p,ref.size()*2,cudaMemcpyDeviceToHost));
            for(size_t i=0;i<ref.size();++i) got[i]=strata::bf16_to_fp32(half[i]);
            require(error(got,ref),4e-3f,"prefill BF16",type,n);
            const int ff=nr/2;
            strata::kernels::iq_dequant_gu_f16(type,w.p,(uint8_t*)w.p+ff*rb,ff,n,(uint16_t*)half_out.p,stream);
            ck(cudaStreamSynchronize(stream));
            ck(cudaMemcpy(half.data(),half_out.p,(size_t)2*ff*n*2,cudaMemcpyDeviceToHost));
            std::vector<float> interleaved(2*ff*n),interleaved_ref(2*ff*n);
            for(int r=0;r<2*ff;++r) for(int c=0;c<n;++c) {
                interleaved[r*n+c]=strata::fp16_to_fp32(half[r*n+c]);
                interleaved_ref[r*n+c]=ref[(r/2+(r%2)*ff)*n+c];
            }
            require(error(interleaved,interleaved_ref),6e-4f,"prefill gate/up interleave",type,n);
            // The row-aware prompt adapter, including a nonzero starting row.
            if(nr>1) {
                strata::kernels::dequant_f32(type,w.p,1,nr-1,n,(float*)out.p,stream);
                ck(cudaStreamSynchronize(stream));
                std::vector<float> tail((size_t)(nr-1)*n),want(ref.begin()+n,ref.end());
                ck(cudaMemcpy(tail.data(),out.p,tail.size()*4,cudaMemcpyDeviceToHost));
                require(error(tail,want),5e-7f,"prompt decode",type,n);
            }
            for(int nt : {1,4}) {
                std::vector<float> x((size_t)nt*n),expected((size_t)nt*nr),y(expected.size());
                for(size_t i=0;i<x.size();++i) x[i]=std::sin(float(i)*.13f);
                Dev dx(x.size()*4),qx((size_t)nt*(n/32)*36),dy(y.size()*4);
                ck(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));
                strata::kernels::quantize_q8_1_rows((float*)dx.p,nt,n,qx.p,stream);
                strata::kernels::kt_mmvq(type,w.p,qx.p,(float*)dy.p,n,nr,nt,stream);
                ck(cudaStreamSynchronize(stream));
                std::vector<uint8_t> q((size_t)nt*(n/32)*36);ck(cudaMemcpy(q.data(),qx.p,q.size(),cudaMemcpyDeviceToHost));
                for(int t=0;t<nt;++t) for(int r=0;r<nr;++r) {
                    double sum=0;
                    for(int c=0;c<n;++c) {
                        const auto* a=q.data()+((size_t)t*(n/32)+c/32)*36;
                        sum+=double(ref[(size_t)r*n+c])*strata::fp16_to_fp32(strata::kt::u16(a))*int((int8_t)a[4+c%32]);
                    }
                    expected[(size_t)t*nr+r]=(float)sum;
                }
                ck(cudaMemcpy(y.data(),dy.p,y.size()*4,cudaMemcpyDeviceToHost));
                e=error(y,expected);require(e,4e-6f,"GEMV",type,n);worst_mv=std::max(e,worst_mv);
            }
        }
        ck(cudaStreamDestroy(stream));
        std::printf("device %s: %u KT cases passed, GPU decode max rel %.3g, GEMV %.3g\n",argv[2],header[1],worst_dq,worst_mv);
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"kt_parity: %s\n",e.what());return 1; }
}
