#include "kernel_operator.h"
using namespace AscendC;

// Explicit BF16 round after each add matches upstream's register reduction.
extern "C" __global__ __aicore__ void spark_kc(GM_ADDR input,GM_ADDR output,uint32_t n,uint32_t m,uint32_t total) {
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

__aicore__ inline uint32_t CountBits(uint32_t x) {
 x-=((x>>1)&0x55555555U);x=(x&0x33333333U)+((x>>2)&0x33333333U);x=(x+(x>>4))&0x0f0f0f0fU;return (x*0x01010101U)>>24;
}
__aicore__ inline uint64_t DoubleBits(uint32_t value) {
 uint64_t x=value;x=(x|(x<<16))&0x0000ffff0000ffffULL;x=(x|(x<<8))&0x00ff00ff00ff00ffULL;
 x=(x|(x<<4))&0x0f0f0f0f0f0f0f0fULL;x=(x|(x<<2))&0x3333333333333333ULL;x=(x|(x<<1))&0x5555555555555555ULL;return x|(x<<1);
}

// GatherMask compacts on-chip; all GM output writes are contiguous. Int64
// indices are compacted as pairs of uint32 words, with a duplicated predicate.
extern "C" __global__ __aicore__ void spark_partition(GM_ADDR ids,GM_ADDR positions,GM_ADDR scores,GM_ADDR cuts,GM_ADDR oi,GM_ADDR op,GM_ADDR status,
 uint32_t b,uint32_t n,uint32_t take,uint32_t is0,uint32_t ps0) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 TPipe pipe;TBuf<TPosition::VECCALC> sb,lb,eb,ib,pb,mb,ob;
 pipe.InitBuffer(sb,512*4);pipe.InitBuffer(lb,64);pipe.InitBuffer(eb,64);pipe.InitBuffer(ib,512*8);pipe.InitBuffer(pb,512*8);pipe.InitBuffer(mb,16*32);pipe.InitBuffer(ob,512*8);
 auto s=sb.Get<float>();auto lt=lb.Get<uint8_t>();auto eq=eb.Get<uint8_t>();auto ii=ib.Get<uint32_t>();auto pp=pb.Get<uint32_t>();auto mask=mb.Get<uint32_t>();auto out=ob.Get<uint32_t>();
 GlobalTensor<float> sg,cg;sg.SetGlobalBuffer((__gm__ float*)scores);cg.SetGlobalBuffer((__gm__ float*)cuts);
 GlobalTensor<uint32_t> ig,pg,og,pog,st;ig.SetGlobalBuffer((__gm__ uint32_t*)ids);pg.SetGlobalBuffer((__gm__ uint32_t*)positions);og.SetGlobalBuffer((__gm__ uint32_t*)oi);pog.SetGlobalBuffer((__gm__ uint32_t*)op);st.SetGlobalBuffer((__gm__ uint32_t*)status);
 for(uint32_t row=GetBlockIdx();row<b;row+=GetBlockNum()) {
  float cut=cg.GetValue(row);uint32_t totalLT=0;
  for(uint32_t begin=0;begin<n;begin+=512) {
   uint32_t count=(n-begin<512?n-begin:512);DataCopy(s,sg[row*n+begin],count);PipeBarrier<PIPE_ALL>();CompareScalar(lt,s,cut,CMPMODE::LT,count);PipeBarrier<PIPE_ALL>();
   auto bits=lt.ReinterpretCast<uint32_t>();for(uint32_t j=0;j<count/32;++j)totalLT+=CountBits(bits.GetValue(j));
  }
  uint32_t need=take-totalLT,left=0,right=0,bad=0;
  for(uint32_t begin=0;begin<n;begin+=512) {
   uint32_t count=(n-begin<512?n-begin:512);DataCopy(s,sg[row*n+begin],count);DataCopy(ii,ig[(static_cast<uint64_t>(row)*is0+begin)*2],count*2);DataCopy(pp,pg[(static_cast<uint64_t>(row)*ps0+begin)*2],count*2);PipeBarrier<PIPE_ALL>();
   CompareScalar(lt,s,cut,CMPMODE::LT,count);CompareScalar(eq,s,cut,CMPMODE::EQ,count);PipeBarrier<PIPE_ALL>();
   auto lower=lt.ReinterpretCast<uint32_t>();auto equal=eq.ReinterpretCast<uint32_t>();uint32_t expectedL=0;
   for(uint32_t j=0;j<count/32;++j) {
    uint32_t e=equal.GetValue(j),ec=CountBits(e),selected=0;
    if(ec<=need){selected=e;need-=ec;}else{for(uint32_t t=0;t<need;++t){uint32_t bit=e&(~e+1);selected|=bit;e-=bit;}need=0;}
    uint32_t bits=lower.GetValue(j)|selected;expectedL+=CountBits(bits);uint64_t expanded=DoubleBits(bits);mask.SetValue(j*8,static_cast<uint32_t>(expanded));mask.SetValue(j*8+1,static_cast<uint32_t>(expanded>>32));
   }
   PipeBarrier<PIPE_ALL>();GatherMaskParams params{1,static_cast<uint16_t>(count/32),8,1};uint64_t selectedWords=0;
   GatherMask(out,ii,mask,false,0,params,selectedWords);PipeBarrier<PIPE_ALL>();bad+=(selectedWords!=expectedL*2);
   if(expectedL)DataCopyPad(og[(static_cast<uint64_t>(row)*n+left)*2],out,DataCopyExtParams{1,expectedL*8,0,0,0});PipeBarrier<PIPE_ALL>();
   GatherMask(out,pp,mask,false,0,params,selectedWords);PipeBarrier<PIPE_ALL>();
   if(expectedL)DataCopyPad(pog[(static_cast<uint64_t>(row)*n+left)*2],out,DataCopyExtParams{1,expectedL*8,0,0,0});PipeBarrier<PIPE_ALL>();left+=expectedL;
   for(uint32_t j=0;j<count/32;++j){mask.SetValue(j*8,~mask.GetValue(j*8));mask.SetValue(j*8+1,~mask.GetValue(j*8+1));}
   PipeBarrier<PIPE_ALL>();uint32_t expectedR=count-expectedL;
   GatherMask(out,ii,mask,false,0,params,selectedWords);PipeBarrier<PIPE_ALL>();bad+=(selectedWords!=expectedR*2);
   if(expectedR)DataCopyPad(og[(static_cast<uint64_t>(row)*n+take+right)*2],out,DataCopyExtParams{1,expectedR*8,0,0,0});PipeBarrier<PIPE_ALL>();
   GatherMask(out,pp,mask,false,0,params,selectedWords);PipeBarrier<PIPE_ALL>();
   if(expectedR)DataCopyPad(pog[(static_cast<uint64_t>(row)*n+take+right)*2],out,DataCopyExtParams{1,expectedR*8,0,0,0});PipeBarrier<PIPE_ALL>();right+=expectedR;
  }
  // Full cache-line DMA avoids unflushed scalar GM stores and false sharing.
  Duplicate(mask,uint32_t(0),8);PipeBarrier<PIPE_ALL>();mask.SetValue(0,left);mask.SetValue(1,right);mask.SetValue(2,bad);mask.SetValue(3,need);PipeBarrier<PIPE_ALL>();DataCopy(st[row*8],mask,8);PipeBarrier<PIPE_ALL>();
 }
}
