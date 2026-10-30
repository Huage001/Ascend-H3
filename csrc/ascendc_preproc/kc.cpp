#include "kernel_operator.h"
using namespace AscendC;

// Explicit BF16 round after each add matches upstream's register reduction.
extern "C" __global__ __aicore__ void spark_kc_v4(GM_ADDR input,GM_ADDR output,uint32_t n,uint32_t m,uint32_t total) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 TPipe pipe;TBuf<TPosition::VECCALC> kb,ab,tb,rb;
 pipe.InitBuffer(kb,64*128*2);pipe.InitBuffer(ab,8*128*4);pipe.InitBuffer(tb,8*128*4);pipe.InitBuffer(rb,8*128*2);
 auto k=kb.Get<bfloat16_t>();auto a=ab.Get<float>();auto t=tb.Get<float>();auto r=rb.Get<bfloat16_t>();
 GlobalTensor<bfloat16_t> ig,og;ig.SetGlobalBuffer((__gm__ bfloat16_t*)input);og.SetGlobalBuffer((__gm__ bfloat16_t*)output);
 for(uint32_t task=GetBlockIdx();task<total;task+=GetBlockNum()) {
  const uint32_t h=task/m,b=task%m;DataCopy(k,ig[(static_cast<uint64_t>(h)*n+b*64)*128],64*128);PipeBarrier<PIPE_ALL>();
  Cast(a,k,RoundMode::CAST_NONE,1024);PipeBarrier<PIPE_V>();
  for(uint32_t j=1;j<8;++j) {
   Cast(t,k[j*1024],RoundMode::CAST_NONE,1024);PipeBarrier<PIPE_V>();Add(a,a,t,1024);PipeBarrier<PIPE_V>();
   Cast(r,a,RoundMode::CAST_RINT,1024);PipeBarrier<PIPE_V>();Cast(a,r,RoundMode::CAST_NONE,1024);PipeBarrier<PIPE_V>();
  }
  for(uint32_t j=0;j<4;++j) Add(t[j*128],a[j*256],a[j*256+128],128);
  PipeBarrier<PIPE_V>();Cast(r,t,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();Cast(a,r,RoundMode::CAST_NONE,512);PipeBarrier<PIPE_V>();
  Add(t,a,a[256],256);PipeBarrier<PIPE_V>();Cast(r,t,RoundMode::CAST_RINT,256);PipeBarrier<PIPE_V>();Cast(a,r,RoundMode::CAST_NONE,256);PipeBarrier<PIPE_V>();
  Add(t,a,a[128],128);PipeBarrier<PIPE_V>();Cast(r,t,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();Cast(a,r,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
  Muls(a,a,1.f/64.f,128);PipeBarrier<PIPE_V>();Cast(r,a,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_ALL>();DataCopy(og[static_cast<uint64_t>(task)*128],r,128);PipeBarrier<PIPE_ALL>();
 }
}

