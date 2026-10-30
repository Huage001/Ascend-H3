#include "kernel_operator.h"
using namespace AscendC;
__aicore__ inline uint32_t CountBits(uint32_t x) {
 x-=((x>>1)&0x55555555U);x=(x&0x33333333U)+((x>>2)&0x33333333U);x=(x+(x>>4))&0x0f0f0f0fU;return (x*0x01010101U)>>24;
}
extern "C" __global__ __aicore__ void spark_partition_fast_v1(GM_ADDR ids,GM_ADDR positions,GM_ADDR scores,GM_ADDR cuts,GM_ADDR oi,GM_ADDR op,GM_ADDR status,
 uint32_t b,uint32_t n,uint32_t take,uint32_t is0,uint32_t ps0,uint32_t same) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 TPipe pipe;TBuf<TPosition::VECCALC> sb,lb,eb,ib,pb,mb,ob;
 pipe.InitBuffer(sb,512*4);pipe.InitBuffer(lb,64);pipe.InitBuffer(eb,64);pipe.InitBuffer(ib,512*4);pipe.InitBuffer(pb,512*4);pipe.InitBuffer(mb,8*32);pipe.InitBuffer(ob,512*4);
 auto s=sb.Get<float>();auto lt=lb.Get<uint8_t>();auto eq=eb.Get<uint8_t>();auto ii=ib.Get<uint32_t>();auto pp=pb.Get<uint32_t>();auto mask=mb.Get<uint32_t>();auto out=ob.Get<uint32_t>();
 GlobalTensor<uint32_t> cgBits;cgBits.SetGlobalBuffer((__gm__ uint32_t*)cuts);
 GlobalTensor<float> sg,cg;sg.SetGlobalBuffer((__gm__ float*)scores);cg.SetGlobalBuffer((__gm__ float*)cuts);
 GlobalTensor<uint32_t> ig,pg,og,pog,st;ig.SetGlobalBuffer((__gm__ uint32_t*)ids);pg.SetGlobalBuffer((__gm__ uint32_t*)positions);og.SetGlobalBuffer((__gm__ uint32_t*)oi);pog.SetGlobalBuffer((__gm__ uint32_t*)op);st.SetGlobalBuffer((__gm__ uint32_t*)status);
 for(uint32_t row=GetBlockIdx();row<b;row+=GetBlockNum()) {
  float cut=cg.GetValue(row*2);uint32_t lo=cgBits.GetValue(row*2),hi=cgBits.GetValue(row*2+1);lo=(lo&0x7fffffffU)?lo:0;hi=(hi&0x7fffffffU)?hi:0;bool tied=(lo==hi);uint32_t totalLT=0;
  if(tied)for(uint32_t begin=0;begin<n;begin+=512) {
   uint32_t count=(n-begin<512?n-begin:512);DataCopy(s,sg[row*n+begin],count);PipeBarrier<PIPE_ALL>();CompareScalar(lt,s,cut,CMPMODE::LT,count);PipeBarrier<PIPE_ALL>();
   auto bits=lt.ReinterpretCast<uint32_t>();for(uint32_t j=0;j<count/32;++j)totalLT+=CountBits(bits.GetValue(j));
  }
  uint32_t need=tied?take-totalLT:0,left=0,right=0,bad=0;
  for(uint32_t begin=0;begin<n;begin+=512) {
   uint32_t count=(n-begin<512?n-begin:512);DataCopy(s,sg[row*n+begin],count);DataCopy(ii,ig[(static_cast<uint64_t>(row)*is0+begin)],count);if(!same)DataCopy(pp,pg[(static_cast<uint64_t>(row)*ps0+begin)],count);PipeBarrier<PIPE_ALL>();
   CompareScalar(lt,s,cut,CMPMODE::LT,count);CompareScalar(eq,s,cut,CMPMODE::EQ,count);PipeBarrier<PIPE_ALL>();
   auto lower=lt.ReinterpretCast<uint32_t>();auto equal=eq.ReinterpretCast<uint32_t>();uint32_t expectedL=0;
   for(uint32_t j=0;j<count/32;++j) {
    uint32_t e=equal.GetValue(j),ec=CountBits(e),selected=e;
    if(tied){selected=0;
    if(ec<=need){selected=e;need-=ec;}else{for(uint32_t t=0;t<need;++t){uint32_t bit=e&(~e+1);selected|=bit;e-=bit;}need=0;}
    }
    uint32_t bits=lower.GetValue(j)|selected;expectedL+=CountBits(bits);mask.SetValue((j/2)*8+j%2,bits);
   }
   PipeBarrier<PIPE_ALL>();GatherMaskParams params{1,static_cast<uint16_t>(count/64),8,1};uint64_t selectedWords=0;
   GatherMask(out,ii,mask,false,0,params,selectedWords);PipeBarrier<PIPE_ALL>();bad+=(selectedWords!=expectedL);
   if(expectedL)DataCopyPad(og[(static_cast<uint64_t>(row)*n+left)],out,DataCopyExtParams{1,expectedL*4,0,0,0});PipeBarrier<PIPE_ALL>();
   if(!same){GatherMask(out,pp,mask,false,0,params,selectedWords);PipeBarrier<PIPE_ALL>();
   if(expectedL)DataCopyPad(pog[(static_cast<uint64_t>(row)*n+left)],out,DataCopyExtParams{1,expectedL*4,0,0,0});PipeBarrier<PIPE_ALL>();}left+=expectedL;
   for(uint32_t j=0;j<count/32;++j){mask.SetValue((j/2)*8+j%2,~mask.GetValue((j/2)*8+j%2));}
   PipeBarrier<PIPE_ALL>();uint32_t expectedR=count-expectedL;
   GatherMask(out,ii,mask,false,0,params,selectedWords);PipeBarrier<PIPE_ALL>();bad+=(selectedWords!=expectedR);
   if(expectedR)DataCopyPad(og[(static_cast<uint64_t>(row)*n+take+right)],out,DataCopyExtParams{1,expectedR*4,0,0,0});PipeBarrier<PIPE_ALL>();
   if(!same){GatherMask(out,pp,mask,false,0,params,selectedWords);PipeBarrier<PIPE_ALL>();
   if(expectedR)DataCopyPad(pog[(static_cast<uint64_t>(row)*n+take+right)],out,DataCopyExtParams{1,expectedR*4,0,0,0});PipeBarrier<PIPE_ALL>();}right+=expectedR;
  }
  // Full cache-line DMA avoids unflushed scalar GM stores and false sharing.
  Duplicate(mask,uint32_t(0),8);PipeBarrier<PIPE_ALL>();mask.SetValue(0,left);mask.SetValue(1,right);mask.SetValue(2,bad);mask.SetValue(3,need);PipeBarrier<PIPE_ALL>();DataCopy(st[row*8],mask,8);PipeBarrier<PIPE_ALL>();
 }
}
