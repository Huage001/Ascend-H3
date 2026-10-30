#include "kernel_operator.h"
#include "utils/std/cmath.h"
using namespace AscendC;
__aicore__ inline uint32_t Pop(uint32_t x){x-=((x>>1)&0x55555555U);x=(x&0x33333333U)+((x>>2)&0x33333333U);x=(x+(x>>4))&0x0f0f0f0fU;return (x*0x01010101U)>>24;}
__aicore__ inline void RowSum128(LocalTensor<float> x,LocalTensor<float> sums,LocalTensor<float> red,uint32_t rows){
 Add(red,x,x[64],uint64_t(64),uint8_t(rows),BinaryRepeatParams{1,1,1,8,16,16});PipeBarrier<PIPE_V>();WholeReduceSum(sums,red,uint64_t(64),rows,1,1,8);PipeBarrier<PIPE_V>();
}
__aicore__ inline void Normalize(LocalTensor<float> x,LocalTensor<float> y,uint32_t rows,LocalTensor<float> prod,LocalTensor<float> red,LocalTensor<float> sums,LocalTensor<float> scale,uint32_t normMode){
 Mul(prod,x,x,rows*128);PipeBarrier<PIPE_V>();if(rows==2 && normMode==1){ReduceSum(sums,prod,red,128);PipeBarrier<PIPE_V>();ReduceSum(sums[8],prod[128],red,128);PipeBarrier<PIPE_ALL>();sums.SetValue(1,sums.GetValue(8));PipeBarrier<PIPE_ALL>();}
 else if(rows==2 && normMode==2){Add(red,prod,prod[64],uint64_t(64),uint8_t(2),BinaryRepeatParams{1,1,1,8,16,16});PipeBarrier<PIPE_V>();for(uint32_t r=0;r<2;++r){auto cur=red[r*64];for(uint32_t k=32;k;k/=2){if(k>=8)Add(cur,cur,cur[k],k);else{PipeBarrier<PIPE_ALL>();for(uint32_t j=0;j<k;++j)scale.SetValue(j,cur.GetValue(k+j));PipeBarrier<PIPE_ALL>();Add(cur,cur,scale,k);}PipeBarrier<PIPE_V>();}PipeBarrier<PIPE_ALL>();sums.SetValue(r,cur.GetValue(0));}PipeBarrier<PIPE_ALL>();}
 else RowSum128(prod,sums,red,rows);
 if(rows==2 && normMode==5){PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<2;++r){float v=sums.GetValue(r);sums.SetValue(r,AscendC::Std::sqrt(v));}PipeBarrier<PIPE_ALL>();}
 else Sqrt(sums,sums,rows);PipeBarrier<PIPE_V>();Maxs(sums,sums,1.e-12f,rows);PipeBarrier<PIPE_V>();Brcb(scale,sums,uint8_t((rows+7)/8),BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();
 Div(y,x,scale,uint64_t(64),uint8_t(rows),BinaryRepeatParams{1,1,0,16,16,1});Div(y[64],x[64],scale,uint64_t(64),uint8_t(rows),BinaryRepeatParams{1,1,0,16,16,1});PipeBarrier<PIPE_V>();
}
__aicore__ inline void Centroid(LocalTensor<float> center,LocalTensor<float> weight,LocalTensor<float> rep,LocalTensor<float> prod,LocalTensor<float> red,LocalTensor<float> output,uint32_t cap){
 Brcb(rep,weight,4,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();Mul(prod,center,rep,uint64_t(64),uint8_t(32),BinaryRepeatParams{1,1,0,16,16,1});Mul(prod[64],center[64],rep,uint64_t(64),uint8_t(32),BinaryRepeatParams{1,1,0,16,16,1});PipeBarrier<PIPE_V>();
 Add(red,prod,prod[2048],2048);PipeBarrier<PIPE_V>();for(uint32_t k=8;k;k/=2){Add(red,red,red[k*128],k*128);PipeBarrier<PIPE_V>();}Duplicate(rep,float(int32_t(cap)),128);PipeBarrier<PIPE_V>();Div(output,red,rep,128);PipeBarrier<PIPE_V>();
}
extern "C" __global__ __aicore__ void spark_proxy_four_v1(GM_ADDR centers,GM_ADDR distances,GM_ADDR upperMask,GM_ADDR output,uint32_t parents,uint32_t c0,uint32_t c1,uint32_t c2,uint32_t c3){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 TPipe p;TBuf<TPosition::VECCALC> cb,fb,ub,pb,rb,db,dwb,lrb,nlrb,dirb,sumb,scaleb,wlb,wrb,wrpb,sb,socb,sortb,sorttmpb,indexb,upperb,maskb,eqb,ltb,maxb,rootb,ch0b,ch1b,numberb;
 p.InitBuffer(cb,8192);p.InitBuffer(fb,16384);p.InitBuffer(ub,16384);p.InitBuffer(pb,16384);p.InitBuffer(rb,8192);p.InitBuffer(db,4096);p.InitBuffer(dwb,4096);p.InitBuffer(lrb,1024);p.InitBuffer(nlrb,1024);p.InitBuffer(dirb,512);p.InitBuffer(sumb,256);p.InitBuffer(scaleb,1024);p.InitBuffer(wlb,256);p.InitBuffer(wrb,128);p.InitBuffer(wrpb,1024);p.InitBuffer(sb,256);p.InitBuffer(socb,256);p.InitBuffer(sortb,512);p.InitBuffer(sorttmpb,512);p.InitBuffer(indexb,256);p.InitBuffer(upperb,128);p.InitBuffer(maskb,128);p.InitBuffer(eqb,128);p.InitBuffer(ltb,32);p.InitBuffer(maxb,32);p.InitBuffer(rootb,128);p.InitBuffer(ch0b,128);p.InitBuffer(ch1b,128);p.InitBuffer(numberb,32);
 auto cbf=cb.Get<bfloat16_t>();auto c=fb.Get<float>();auto u=ub.Get<float>();auto prod=pb.Get<float>();auto red=rb.Get<float>();auto dist=db.Get<float>();auto distwork=dwb.Get<float>();auto lr=lrb.Get<float>();auto nlr=nlrb.Get<float>();auto dir=dirb.Get<float>();auto sums=sumb.Get<float>();auto scale=scaleb.Get<float>();auto wl=wlb.Get<float>();auto wr=wrb.Get<float>();auto wrep=wrpb.Get<float>();auto scores=sb.Get<float>();auto scorecopy=socb.Get<float>();auto sorted=sortb.Get<float>();auto sorttmp=sorttmpb.Get<float>();auto index=indexb.Get<int32_t>();auto upper=upperb.Get<uint32_t>();auto mask=maskb.Get<uint32_t>();auto eq=eqb.Get<uint8_t>();auto lt=ltb.Get<uint8_t>();auto maximum=maxb.Get<float>();auto rootw=rootb.Get<float>();auto child0=ch0b.Get<float>();auto child1=ch1b.Get<float>();auto number=numberb.Get<float>();
 GlobalTensor<bfloat16_t> cg;GlobalTensor<float> dg,og;GlobalTensor<uint32_t> mg;cg.SetGlobalBuffer((__gm__ bfloat16_t*)centers);dg.SetGlobalBuffer((__gm__ float*)distances);og.SetGlobalBuffer((__gm__ float*)output);mg.SetGlobalBuffer((__gm__ uint32_t*)upperMask);
 DataCopy(upper,mg,32);ArithProgression(index,0,1,64);Duplicate(rootw,float(int32_t(c0+c1+c2+c3))/32.f,32);Duplicate(sums,0.f,64);PipeBarrier<PIPE_ALL>();
 for(uint32_t row=GetBlockIdx();row<parents;row+=GetBlockNum()){
  DataCopy(cbf,cg[uint64_t(row)*4096],4096);DataCopy(dist,dg[uint64_t(row)*1024],1024);PipeBarrier<PIPE_ALL>();Cast(c,cbf,RoundMode::CAST_NONE,4096);PipeBarrier<PIPE_V>();Normalize(c,u,32,prod,red,sums,scale,5);PipeBarrier<PIPE_ALL>();
  for(uint32_t node=0;node<3;++node){
   auto weight=node==0?rootw:(node==1?child0:child1);uint32_t lc=node==0?c0+c1:(node==1?c0:c2),rc=node==0?c2+c3:(node==1?c1:c3);uint32_t active=0,only=0;
   PipeBarrier<PIPE_ALL>();for(uint32_t j=0;j<32;++j)if(weight.GetValue(j)>0.f){active|=uint32_t(1)<<j;only=j;}
   uint32_t pair=only*32+only;
   if(Pop(active)>1){
    for(uint32_t j=0;j<32;++j)mask.SetValue(j,(active&(uint32_t(1)<<j))?(upper.GetValue(j)&active):0);PipeBarrier<PIPE_ALL>();Select(distwork,mask.ReinterpretCast<uint8_t>(),dist,float(-3.402823466e38F),SELMODE::VSEL_TENSOR_SCALAR_MODE,1024);PipeBarrier<PIPE_V>();ReduceMax(maximum,distwork,red,1024,false);PipeBarrier<PIPE_ALL>();float md=maximum.GetValue(0);CompareScalar(eq,distwork,md,CMPMODE::EQ,1024);PipeBarrier<PIPE_ALL>();auto words=eq.ReinterpretCast<uint32_t>();bool found=false;
    for(uint32_t j=0;j<32&&!found;++j){uint32_t bits=words.GetValue(j);if(bits){for(uint32_t k=0;k<32;++k)if(bits&(uint32_t(1)<<k)){pair=j*32+k;found=true;break;}}}
   }
   DataCopy(lr,c[(pair/32)*128],128);DataCopy(lr[128],c[(pair%32)*128],128);PipeBarrier<PIPE_V>();Adds(lr,lr,0.f,256);PipeBarrier<PIPE_V>();
   for(uint32_t it=0;it<2;++it){
    Normalize(lr,nlr,2,prod,red,sums,scale,5);Sub(dir,nlr[128],nlr,128);PipeBarrier<PIPE_V>();Mul(prod,u,dir,uint64_t(64),uint8_t(32),BinaryRepeatParams{1,1,1,16,16,0});Mul(prod[64],u[64],dir[64],uint64_t(64),uint8_t(32),BinaryRepeatParams{1,1,1,16,16,0});PipeBarrier<PIPE_V>();RowSum128(prod,scores,red,32);Duplicate(scores[32],float(3.402823466e38F),32);Muls(scorecopy,scores,-1.f,32);Duplicate(scorecopy[32],float(-3.402823466e38F),32);PipeBarrier<PIPE_V>();Sort<float,true>(sorted,scorecopy,index.ReinterpretCast<uint32_t>(),sorttmp,2);PipeBarrier<PIPE_ALL>();
    float total=0.f,cut=0.f;auto sortedbits=sorted.ReinterpretCast<uint32_t>();for(uint32_t rank=0;rank<32;++rank){uint32_t ix=sortedbits.GetValue(rank*2+1);if(ix<32)total+=weight.GetValue(ix);if(total>=float(int32_t(lc))){cut=-sorted.GetValue(rank*2);break;}}
    CompareScalar(lt,scores,cut,CMPMODE::LT,64);CompareScalar(eq,scores,cut,CMPMODE::EQ,64);PipeBarrier<PIPE_V>();Select(wl,lt,weight,0.f,SELMODE::VSEL_TENSOR_SCALAR_MODE,32);PipeBarrier<PIPE_V>();Duplicate(wl[32],0.f,32);PipeBarrier<PIPE_V>();ReduceSum(number,wl,sorttmp,64);PipeBarrier<PIPE_ALL>();uint32_t need=lc-uint32_t(int32_t(number.GetValue(0))),equal=eq.ReinterpretCast<uint32_t>().GetValue(0);
    while(need&&equal){uint32_t bit=equal&(~equal+1),ix=0;while(!(bit&(uint32_t(1)<<ix)))++ix;uint32_t available=uint32_t(int32_t(weight.GetValue(ix))),take=available<need?available:need;wl.SetValue(ix,float(int32_t(take)));need-=take;equal-=bit;}
    PipeBarrier<PIPE_ALL>();Sub(wr,weight,wl,32);PipeBarrier<PIPE_V>();Centroid(c,wl,wrep,prod,red,lr,lc);Centroid(c,wr,wrep,prod,red,lr[128],rc);
   }
   Normalize(lr,nlr,2,prod,red,sums,scale,5);Sub(dir,nlr[128],nlr,128);PipeBarrier<PIPE_ALL>();DataCopy(og[(uint64_t(row)*3+node)*128],dir,128);if(node==0){DataCopy(child0,wl,32);DataCopy(child1,wr,32);}PipeBarrier<PIPE_ALL>();
  }
 }
}
