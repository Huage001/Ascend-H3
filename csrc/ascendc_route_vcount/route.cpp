#include "kernel_operator.h"
using namespace AscendC;
__aicore__ inline uint32_t Pop(uint32_t x){x-=((x>>1)&0x55555555U);x=(x&0x33333333U)+((x>>2)&0x33333333U);x=(x+(x>>4))&0x0f0f0f0fU;return (x*0x01010101U)>>24;}
extern "C" __global__ __aicore__ void spark_route_vcount_v1(GM_ADDR input,GM_ADDR output,uint32_t rows,uint32_t m,uint32_t w) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 uint32_t p=(m+127)/128*128;TPipe pipe;TBuf<TPosition::VECCALC> sb,ib,db,tb,gb,eb,hb,ob,ub,mb,rb,fb,nb;
 pipe.InitBuffer(fb,p*4);pipe.InitBuffer(nb,32);pipe.InitBuffer(rb,p*4);pipe.InitBuffer(sb,p*4);pipe.InitBuffer(ib,p*4);pipe.InitBuffer(db,p*8);pipe.InitBuffer(tb,p*8);pipe.InitBuffer(gb,p/8);pipe.InitBuffer(eb,p/8);pipe.InitBuffer(hb,p*2);pipe.InitBuffer(ob,p*2);pipe.InitBuffer(ub,p);pipe.InitBuffer(mb,p/4);
 auto countFloat=fb.Get<float>();auto number=nb.Get<float>();auto s=sb.Get<float>();auto raw=rb.Get<float>();auto idx=ib.Get<int32_t>();auto sorted=db.Get<float>();auto temp=tb.Get<float>();auto gt=gb.Get<uint8_t>();auto eq=eb.Get<uint8_t>();auto ones=hb.Get<half>();auto selected=ob.Get<half>();auto out=ub.Get<uint8_t>();auto selectMask=mb.Get<uint32_t>();
 ArithProgression(idx,0,1,p);Duplicate(ones,half(1),p);PipeBarrier<PIPE_ALL>();
 GlobalTensor<float> sg;GlobalTensor<uint8_t> og;sg.SetGlobalBuffer((__gm__ float*)input);og.SetGlobalBuffer((__gm__ uint8_t*)output);
 for(uint32_t row=GetBlockIdx();row<rows;row+=GetBlockNum()) {
  Duplicate(s,-3.402823466e38f,p);PipeBarrier<PIPE_ALL>();DataCopyPad(s,sg[static_cast<uint64_t>(row)*m],DataCopyExtParams{1,m*4,0,0,0},DataCopyPadExtParams<float>{true,0,static_cast<uint8_t>((8-m%8)%8),-3.402823466e38f});PipeBarrier<PIPE_ALL>();
  DataCopy(raw,s,p);PipeBarrier<PIPE_V>();
  Sort<float,true>(sorted,s,idx.ReinterpretCast<uint32_t>(),temp,p/32);PipeBarrier<PIPE_ALL>();SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);float cut=sorted.GetValue((w-1)*2);
  CompareScalar(gt,raw,cut,CMPMODE::GT,p);CompareScalar(eq,raw,cut,CMPMODE::EQ,p);PipeBarrier<PIPE_ALL>();SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  float after=(w<m)?sorted.GetValue(w*2):float(-3.402823466e38F);
  if(w<m && cut!=after){CompareScalar(gt,raw,cut,CMPMODE::GE,p);PipeBarrier<PIPE_V>();}
  else {
   auto gbits=gt.ReinterpretCast<uint32_t>();auto ebits=eq.ReinterpretCast<uint32_t>();
   if(m%32){uint32_t valid=(uint32_t(1)<<(m%32))-1;gbits.SetValue(m/32,gbits.GetValue(m/32)&valid);}
   for(uint32_t j=(m+31)/32;j<p/32;++j)gbits.SetValue(j,0);
   PipeBarrier<PIPE_ALL>();Select(selected,gt,ones,half(0),SELMODE::VSEL_TENSOR_SCALAR_MODE,p);PipeBarrier<PIPE_V>();Cast(countFloat,selected,RoundMode::CAST_NONE,p);PipeBarrier<PIPE_V>();ReduceSum(number,countFloat,temp,p);PipeBarrier<PIPE_ALL>();uint32_t need=w-uint32_t(int32_t(number.GetValue(0)));
   for(uint32_t j=0;j<p/32 && need;++j){uint32_t remain=(j*32<m?m-j*32:0);uint32_t valid=(remain>=32?0xffffffffU:(remain?(1U<<remain)-1:0));uint32_t e=ebits.GetValue(j)&valid;if(!e)continue;uint32_t ec=Pop(e),take=0;if(ec<=need){take=e;need-=ec;}else{for(uint32_t t=0;t<need;++t){uint32_t bit=e&(~e+1);take|=bit;e-=bit;}need=0;}gbits.SetValue(j,gbits.GetValue(j)|take);}
   PipeBarrier<PIPE_ALL>();
  }
  Select(selected,gt,ones,half(0),SELMODE::VSEL_TENSOR_SCALAR_MODE,p);PipeBarrier<PIPE_V>();Cast(out,selected,RoundMode::CAST_NONE,p);PipeBarrier<PIPE_V>();
  PipeBarrier<PIPE_ALL>();
  DataCopy(og[static_cast<uint64_t>(row)*p],out,p);PipeBarrier<PIPE_ALL>();
 }
}
