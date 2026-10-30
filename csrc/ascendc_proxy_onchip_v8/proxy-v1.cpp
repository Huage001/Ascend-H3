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
 else Sqrt(sums,sums,rows);PipeBarrier<PIPE_V>();Maxs(sums,sums,1.e-12f,rows);PipeBarrier<PIPE_V>();if(rows==2){Brcb(scale,sums,1,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();Div(y,x,scale,uint64_t(64),uint8_t(rows),BinaryRepeatParams{1,1,0,16,16,1});Div(y[64],x[64],scale,uint64_t(64),uint8_t(rows),BinaryRepeatParams{1,1,0,16,16,1});}
 else{Brcb(scale,sums,uint8_t((rows+7)/8),BrcbRepeatParams{2,16});Brcb(scale[8],sums,uint8_t((rows+7)/8),BrcbRepeatParams{2,16});PipeBarrier<PIPE_V>();Div(y,x,scale,uint64_t(64),uint8_t(rows*2),BinaryRepeatParams{1,1,0,8,8,1});}PipeBarrier<PIPE_V>();
}
__aicore__ inline void Centroid(LocalTensor<float> center,LocalTensor<float> weight,LocalTensor<float> rep,LocalTensor<float> prod,LocalTensor<float> red,LocalTensor<float> output){
 Brcb(rep,weight,4,BrcbRepeatParams{1,8});PipeBarrier<PIPE_V>();Mul(prod,center,rep,uint64_t(64),uint8_t(32),BinaryRepeatParams{1,1,0,16,16,1});Mul(prod[64],center[64],rep,uint64_t(64),uint8_t(32),BinaryRepeatParams{1,1,0,16,16,1});PipeBarrier<PIPE_V>();
 Add(red,prod,prod[2048],2048);PipeBarrier<PIPE_V>();for(uint32_t k=8;k;k/=2){Add(red,red,red[k*128],k*128);PipeBarrier<PIPE_V>();}Muls(output,red,1.f/64.f,128);PipeBarrier<PIPE_V>();
}
extern "C" __global__ __aicore__ void spark_proxy_terminal_scalar_v8(GM_ADDR centers,GM_ADDR distances,GM_ADDR upperMask,GM_ADDR output,GM_ADDR debugUnits,uint32_t parents,uint32_t debug,uint32_t normMode){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
 TPipe p;TBuf<TPosition::VECCALC> cb,fb,ub,pb,rb,db,lrb,nlrb,dirb,sumb,scaleb,wb,w4b,wrb,sb,socb,sortb,sorttmpb,indexb,upperb,eqb,ltb,maxb;
 p.InitBuffer(cb,8192);p.InitBuffer(fb,16384);p.InitBuffer(ub,16384);p.InitBuffer(pb,16384);p.InitBuffer(rb,8192);p.InitBuffer(db,4096);p.InitBuffer(lrb,1024);p.InitBuffer(nlrb,1024);p.InitBuffer(dirb,512);p.InitBuffer(sumb,256);p.InitBuffer(scaleb,2048);p.InitBuffer(wb,128);p.InitBuffer(w4b,128);p.InitBuffer(wrb,1024);p.InitBuffer(sb,256);p.InitBuffer(socb,256);p.InitBuffer(sortb,512);p.InitBuffer(sorttmpb,512);p.InitBuffer(indexb,256);p.InitBuffer(upperb,128);p.InitBuffer(eqb,128);p.InitBuffer(ltb,32);p.InitBuffer(maxb,32);
 auto cbf=cb.Get<bfloat16_t>();auto c=fb.Get<float>();auto u=ub.Get<float>();auto prod=pb.Get<float>();auto red=rb.Get<float>();auto dist=db.Get<float>();auto lr=lrb.Get<float>();auto nlr=nlrb.Get<float>();auto dir=dirb.Get<float>();auto sums=sumb.Get<float>();auto scale=scaleb.Get<float>();auto w=wb.Get<float>();auto w4=w4b.Get<float>();auto wrep=wrb.Get<float>();auto scores=sb.Get<float>();auto scorecopy=socb.Get<float>();auto sorted=sortb.Get<float>();auto sorttmp=sorttmpb.Get<float>();auto index=indexb.Get<int32_t>();auto upper=upperb.Get<uint32_t>();auto eq=eqb.Get<uint8_t>();auto lt=ltb.Get<uint8_t>();auto maximum=maxb.Get<float>();
 GlobalTensor<bfloat16_t> cg;GlobalTensor<float> dg,og,ug;GlobalTensor<uint32_t> mg;cg.SetGlobalBuffer((__gm__ bfloat16_t*)centers);dg.SetGlobalBuffer((__gm__ float*)distances);og.SetGlobalBuffer((__gm__ float*)output);ug.SetGlobalBuffer((__gm__ float*)debugUnits);mg.SetGlobalBuffer((__gm__ uint32_t*)upperMask);
 DataCopy(upper,mg,32);ArithProgression(index,0,1,64);Duplicate(w4,4.f,32);Duplicate(sums,0.f,64);PipeBarrier<PIPE_ALL>();
 for(uint32_t row=GetBlockIdx();row<parents;row+=GetBlockNum()){
  uint32_t next=row+GetBlockNum();if(row==GetBlockIdx())DataCopy(cbf,cg[uint64_t(row)*4096],4096);DataCopy(dist,dg[uint64_t(row)*1024],1024);PipeBarrier<PIPE_ALL>();Cast(c,cbf,RoundMode::CAST_NONE,4096);PipeBarrier<PIPE_V>();if(next<parents){SetFlag<HardEvent::V_MTE2>(EVENT_ID6);WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);DataCopy(cbf,cg[uint64_t(next)*4096],4096);}Normalize(c,u,32,prod,red,sums,scale,normMode);PipeBarrier<PIPE_ALL>();if(debug){DataCopy(ug[uint64_t(row)*4096],u,4096);PipeBarrier<PIPE_ALL>();}
  Select(dist,upper.ReinterpretCast<uint8_t>(),dist,float(-3.402823466e38F),SELMODE::VSEL_TENSOR_SCALAR_MODE,1024);PipeBarrier<PIPE_V>();ReduceMax(maximum,dist,red,1024,false);PipeBarrier<PIPE_ALL>();float md=maximum.GetValue(0);CompareScalar(eq,dist,md,CMPMODE::EQ,1024);PipeBarrier<PIPE_ALL>();uint32_t pair=1;auto words=eq.ReinterpretCast<uint32_t>();bool found=false;
  for(uint32_t j=0;j<32 && !found;++j){uint32_t bits=words.GetValue(j);if(bits){for(uint32_t k=0;k<32;++k){if(bits&(uint32_t(1)<<k)){pair=j*32+k;found=true;break;}}}}
  DataCopy(lr,c[(pair/32)*128],128);DataCopy(lr[128],c[(pair%32)*128],128);PipeBarrier<PIPE_V>();Adds(lr,lr,0.f,256);PipeBarrier<PIPE_V>();
  for(uint32_t it=0;it<2;++it){
   Normalize(lr,nlr,2,prod,red,sums,scale,normMode);Sub(dir,nlr[128],nlr,128);PipeBarrier<PIPE_V>();Mul(prod,u,dir,uint64_t(64),uint8_t(32),BinaryRepeatParams{1,1,1,16,16,0});Mul(prod[64],u[64],dir[64],uint64_t(64),uint8_t(32),BinaryRepeatParams{1,1,1,16,16,0});PipeBarrier<PIPE_V>();RowSum128(prod,scores,red,32);Duplicate(scores[32],float(3.402823466e38F),32);PipeBarrier<PIPE_V>();DataCopy(scorecopy,scores,64);PipeBarrier<PIPE_V>();Sort<float,true>(sorted,scorecopy,index.ReinterpretCast<uint32_t>(),sorttmp,2);PipeBarrier<PIPE_ALL>();float cut=sorted.GetValue(96);
   CompareScalar(lt,scores,cut,CMPMODE::LT,64);CompareScalar(eq,scores,cut,CMPMODE::EQ,64);PipeBarrier<PIPE_ALL>();uint32_t lower=lt.ReinterpretCast<uint32_t>().GetValue(0),equal=eq.ReinterpretCast<uint32_t>().GetValue(0),need=16-Pop(lower),chosen=0;
   while(need && equal){uint32_t bit=equal&(~equal+1);chosen|=bit;equal-=bit;--need;}chosen|=lower;
   lt.ReinterpretCast<uint32_t>().SetValue(0,chosen);PipeBarrier<PIPE_ALL>();Select(w,lt,w4,0.f,SELMODE::VSEL_TENSOR_SCALAR_MODE,32);PipeBarrier<PIPE_V>();Centroid(c,w,wrep,prod,red,lr);Sub(w,w4,w,32);PipeBarrier<PIPE_V>();Centroid(c,w,wrep,prod,red,lr[128]);
  }
  Normalize(lr,nlr,2,prod,red,sums,scale,normMode);Sub(dir,nlr[128],nlr,128);PipeBarrier<PIPE_ALL>();DataCopy(og[uint64_t(row)*128],dir,128);PipeBarrier<PIPE_ALL>();
 }
}
