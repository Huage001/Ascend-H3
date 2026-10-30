#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void spark_permute_tail_kv_v1(GM_ADDR key,GM_ADDR value,GM_ADDR ids,GM_ADDR outk,GM_ADDR outv,uint32_t H,uint32_t N,uint32_t G,uint32_t P){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 TPipe pipe;TBuf<TPosition::VECCALC> kb,vb,ib;
 pipe.InitBuffer(kb,128*128*2);pipe.InitBuffer(vb,128*128*2);pipe.InitBuffer(ib,128*8);
 auto k=kb.Get<uint16_t>();auto v=vb.Get<uint16_t>();auto ix=ib.Get<int64_t>();
 GlobalTensor<uint16_t> kg,vg,ko,vo;GlobalTensor<int64_t> map;
 kg.SetGlobalBuffer((__gm__ uint16_t*)key);vg.SetGlobalBuffer((__gm__ uint16_t*)value);ko.SetGlobalBuffer((__gm__ uint16_t*)outk);vo.SetGlobalBuffer((__gm__ uint16_t*)outv);map.SetGlobalBuffer((__gm__ int64_t*)ids);
 uint32_t T=N+G,tiles=(P+127)/128,total=H*tiles;
 for(uint32_t task=GetBlockIdx();task<total;task+=GetBlockNum()){
  uint32_t h=task/tiles,start=(task%tiles)*128,count=(128<P-start?128:P-start),nr=start<T?(count<T-start?count:T-start):0,nc=start<N?(nr<N-start?nr:N-start):0;
  if(nr<count){Duplicate(k,uint16_t(0),count*128);Duplicate(v,uint16_t(0),count*128);PipeBarrier<PIPE_ALL>();}
  if(nc){DataCopyExtParams cp{1,nc*8,0,0,0};DataCopyPadExtParams<int64_t> pad{false,0,0,0};DataCopyPad(ix,map[uint64_t(h)*N+start],cp,pad);PipeBarrier<PIPE_ALL>();}
  for(uint32_t r=0;r<nr;++r){uint32_t src=(start+r<N)?G+uint32_t(ix.GetValue(r)):start+r-N;DataCopy(k[r*128],kg[(uint64_t(src)*H+h)*128],128);DataCopy(v[r*128],vg[(uint64_t(src)*H+h)*128],128);}
  PipeBarrier<PIPE_ALL>();DataCopy(ko[(uint64_t(h)*P+start)*128],k,count*128);DataCopy(vo[(uint64_t(h)*P+start)*128],v,count*128);PipeBarrier<PIPE_ALL>();
 }
}
