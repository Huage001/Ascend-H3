#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void spark_inverse_copy_v1(GM_ADDR input,GM_ADDR ids,GM_ADDR output,uint32_t H,uint32_t N,uint32_t G,uint32_t P,uint32_t B){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 TPipe pipe;TBuf<TPosition::VECCALC> xb,ib;
 pipe.InitBuffer(xb,128*128*2);pipe.InitBuffer(ib,128*8);
 auto x=xb.Get<uint16_t>();auto ix=ib.Get<int64_t>();
 GlobalTensor<uint16_t> src,dst;GlobalTensor<int64_t> map;
 src.SetGlobalBuffer((__gm__ uint16_t*)input);dst.SetGlobalBuffer((__gm__ uint16_t*)output);map.SetGlobalBuffer((__gm__ int64_t*)ids);
 uint32_t T=N+G,tiles=(T+B-1)/B,total=H*tiles;
 for(uint32_t task=GetBlockIdx();task<total;task+=GetBlockNum()){
  uint32_t h=task/tiles,start=(task%tiles)*B,count=(B<T-start?B:T-start),nc=start<N?(count<N-start?count:N-start):0;
  DataCopy(x,src[(uint64_t(h)*P+start)*128],count*128);
  if(nc){DataCopyExtParams cp{1,nc*8,0,0,0};DataCopyPadExtParams<int64_t> pad{false,0,0,0};DataCopyPad(ix,map[uint64_t(h)*N+start],cp,pad);}
  PipeBarrier<PIPE_ALL>();
  for(uint32_t r=0;r<count;++r){uint32_t dest=(start+r<N)?G+uint32_t(ix.GetValue(r)):start+r-N;DataCopy(dst[(uint64_t(dest)*H+h)*128],x[r*128],128);}
  PipeBarrier<PIPE_ALL>();
 }
}
