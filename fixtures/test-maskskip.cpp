#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
static uint32_t rng_step(uint32_t & x) { x ^= x<<13; x ^= x>>17; x ^= x<<5; return x; }
static void fill_quant(ggml_tensor * t, uint32_t seed) {
    std::vector<float> f(ggml_nelements(t));
    for (auto & x:f) x=(int(rng_step(seed)&1023)-512)*0.002f;
    std::vector<uint8_t> q(ggml_nbytes(t));
    auto n=ggml_quantize_chunk(t->type,f.data(),q.data(),0,ggml_nrows(t),t->ne[0],nullptr);
    require(n==q.size(),"quantized size mismatch");
    ggml_backend_tensor_set(t,q.data(),0,q.size());
}

int main(int argc,char ** argv) {
    try {
        require(argc==3,"usage: test-maskskip backend-directory output-directory");
        const std::filesystem::path outdir(argv[2]);
        std::filesystem::create_directories(outdir);
        ggml_backend_load_all_from_path(argv[1]);
        auto dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        require(dev!=nullptr,"no GPU backend");
        auto backend=ggml_backend_dev_init(dev,nullptr);
        require(backend!=nullptr,"GPU initialization failed");
        std::ifstream maps("/proc/self/maps");std::ofstream saved(outdir/"maps.txt");saved<<maps.rdbuf();
        for (int kv:{2048,4096,21504,83200}) {
            for (const std::string mode:{"blocks","scattered","dense","empty","tail"}) {
                ggml_init_params ip={32*1024*1024,nullptr,true};
                auto ctx=ggml_init(ip);require(ctx!=nullptr,"context allocation failed");
                auto q=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,256,1,24,1);
                auto k=ggml_new_tensor_4d(ctx,GGML_TYPE_Q8_0,256,kv,2,1);
                auto v=ggml_new_tensor_4d(ctx,GGML_TYPE_Q8_0,256,kv,2,1);
                auto mask=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,kv,64,1,1);
                auto result=ggml_flash_attn_ext(ctx,q,k,v,mask,1.0f/16.0f,0.0f,0.0f);
                ggml_flash_attn_ext_set_prec(result,GGML_PREC_F32);
                ggml_set_output(result);
                auto graph=ggml_new_graph_custom(ctx,256,false);ggml_build_forward_expand(graph,result);
                auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer!=nullptr,"GPU buffer allocation failed");
                fill_quant(k,1234567);fill_quant(v,89101112);
                std::vector<float> query(ggml_nelements(q));uint32_t seed=31519;
                for (auto & x:query)x=(int(rng_step(seed)&1023)-512)*0.001f;
                ggml_backend_tensor_set(q,query.data(),0,query.size()*sizeof(float));
                std::vector<ggml_fp16_t> masks(ggml_nelements(mask),ggml_fp32_to_fp16(-INFINITY));
                std::mt19937 gen(380914);
                if(mode=="dense")std::fill(masks.begin(),masks.begin()+kv,ggml_fp32_to_fp16(0.0f));
                if(mode=="tail")masks[kv-1]=ggml_fp32_to_fp16(0.0f);
                if(mode=="blocks") {
                    std::vector<int> blocks(kv/4);std::iota(blocks.begin(),blocks.end(),0);std::shuffle(blocks.begin(),blocks.end(),gen);
                    for(int b=0;b<std::min(512,kv/4);++b)for(int j=0;j<4;++j)masks[blocks[b]*4+j]=ggml_fp32_to_fp16(0.0f);
                }
                if(mode=="scattered") {
                    std::vector<int> positions(kv);std::iota(positions.begin(),positions.end(),0);std::shuffle(positions.begin(),positions.end(),gen);
                    for(int j=0;j<std::min(2051,kv);++j)masks[positions[j]]=ggml_fp32_to_fp16(0.0f);
                }
                ggml_backend_tensor_set(mask,masks.data(),0,masks.size()*sizeof(ggml_fp16_t));
                std::vector<double> times;
                for(int rep=-3;rep<25;++rep) {
                    auto t=std::chrono::steady_clock::now();
                    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"graph compute failed");
                    ggml_backend_synchronize(backend);
                    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count();
                    if(rep>=0)times.push_back(ms);
                }
                std::sort(times.begin(),times.end());
                std::vector<float> values(ggml_nelements(result));ggml_backend_tensor_get(result,values.data(),0,values.size()*sizeof(float));
                const auto label=std::to_string(kv)+"-"+mode;
                std::ofstream bytes(outdir/(label+".f32"),std::ios::binary);bytes.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(float));
                int finite=std::count_if(values.begin(),values.end(),[](float f){return std::isfinite(f);});
                printf("{\"case\":\"%s\",\"median_ms\":%.9f,\"min_ms\":%.9f,\"max_ms\":%.9f,\"elements\":%zu,\"finite\":%d}\n",label.c_str(),times[12],times.front(),times.back(),values.size(),finite);fflush(stdout);
                ggml_backend_buffer_free(buffer);ggml_free(ctx);
            }
        }
        ggml_backend_free(backend);ggml_quantize_free();return 0;
    } catch(const std::exception & e) {fprintf(stderr,"%s\n",e.what());return 1;}
}
