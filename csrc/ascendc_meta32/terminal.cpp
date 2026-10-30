#include "kernel_operator.h"
using namespace AscendC;
__aicore__ inline uint32_t Pop(uint32_t x){x-=((x>>1)&0x55555555U);x=(x&0x33333333U)+((x>>2)&0x33333333U);x=(x+(x>>4))&0x0f0f0f0fU;return (x*0x01010101U)>>24;}
extern "C" __global__ __aicore__ void spark_terminal_i32(GM_ADDR scores,GM_ADDR ids,GM_ADDR heads,GM_ADDR starts,GM_ADDR output,GM_ADDR status,uint32_t rows,uint32_t stride,uint32_t outN){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 TPipe pipe;TBuf<TPosition::VECCALC> sb,ib,db,tb,lb,eb,idb,mb,ob,stb;
 pipe.InitBuffer(sb,512);pipe.InitBuffer(ib,512);pipe.InitBuffer(db,1024);pipe.InitBuffer(tb,1024);pipe.InitBuffer(lb,32);pipe.InitBuffer(eb,32);pipe.InitBuffer(idb,512);pipe.InitBuffer(mb,128);pipe.InitBuffer(ob,512);pipe.InitBuffer(stb,32);
 auto s=sb.Get<float>();auto index=ib.Get<int32_t>();auto sorted=db.Get<float>();auto temp=tb.Get<float>();auto lt=lb.Get<uint8_t>();auto eq=eb.Get<uint8_t>();auto id=idb.Get<uint32_t>();auto mask=mb.Get<uint32_t>();auto out=ob.Get<uint32_t>();auto stat=stb.Get<uint32_t>();
 ArithProgression(index,0,1,128);PipeBarrier<PIPE_ALL>();
 GlobalTensor<float> sg;GlobalTensor<uint32_t> ig,og,st;GlobalTensor<int32_t> hg,bg;
 sg.SetGlobalBuffer((__gm__ float*)scores);ig.SetGlobalBuffer((__gm__ uint32_t*)ids);og.SetGlobalBuffer((__gm__ uint32_t*)output);st.SetGlobalBuffer((__gm__ uint32_t*)status);hg.SetGlobalBuffer((__gm__ int32_t*)heads);bg.SetGlobalBuffer((__gm__ int32_t*)starts);
 for(uint32_t row=GetBlockIdx();row<rows;row+=GetBlockNum()){
  DataCopy(s,sg[uint64_t(row)*128],128);PipeBarrier<PIPE_ALL>();
  Sort<float,true>(sorted,s,index.ReinterpretCast<uint32_t>(),temp,4);PipeBarrier<PIPE_ALL>();SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  float cut=sorted.GetValue(64*2);
  DataCopy(s,sg[uint64_t(row)*128],128);DataCopy(id,ig[uint64_t(row)*stride],128);PipeBarrier<PIPE_ALL>();
  CompareScalar(lt,s,cut,CMPMODE::LT,128);CompareScalar(eq,s,cut,CMPMODE::EQ,128);PipeBarrier<PIPE_ALL>();
  auto lower=lt.ReinterpretCast<uint32_t>();auto equal=eq.ReinterpretCast<uint32_t>();uint32_t count=0;
  for(uint32_t j=0;j<4;++j)count+=Pop(lower.GetValue(j));uint32_t need=64-count;
  for(uint32_t j=0;j<4;++j){uint32_t e=equal.GetValue(j),ec=Pop(e),selected=0;if(ec<=need){selected=e;need-=ec;}else{for(uint32_t k=0;k<need;++k){uint32_t bit=e&(~e+1);selected|=bit;e-=bit;}need=0;}mask.SetValue((j/2)*8+j%2,lower.GetValue(j)|selected);}
  PipeBarrier<PIPE_ALL>();uint64_t offset=(uint64_t(hg.GetValue(row))*outN+uint64_t(bg.GetValue(row))*64);GatherMaskParams params{1,2,8,1};uint64_t selectedWords=0;
  GatherMask(out,id,mask,false,0,params,selectedWords);PipeBarrier<PIPE_ALL>();uint32_t left=selectedWords;DataCopy(og[offset],out,64);PipeBarrier<PIPE_ALL>();
  for(uint32_t j=0;j<4;++j){mask.SetValue((j/2)*8+j%2,~mask.GetValue((j/2)*8+j%2));}PipeBarrier<PIPE_ALL>();
  GatherMask(out,id,mask,false,0,params,selectedWords);PipeBarrier<PIPE_ALL>();uint32_t right=selectedWords;DataCopy(og[offset+64],out,64);PipeBarrier<PIPE_ALL>();
  Duplicate(stat,uint32_t(0),8);PipeBarrier<PIPE_ALL>();stat.SetValue(0,left);stat.SetValue(1,right);stat.SetValue(2,need);PipeBarrier<PIPE_ALL>();DataCopy(st[uint64_t(row)*8],stat,8);PipeBarrier<PIPE_ALL>();
 }
}
