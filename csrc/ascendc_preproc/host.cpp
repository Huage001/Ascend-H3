#include <ATen/ATen.h>
#include <torch/library.h>
#include <tiling/platform/platform_ascendc.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "aclrtlaunch_spark_kc_v4.h"
#include "aclrtlaunch_spark_partition_v4.h"
#include "aclrtlaunch_spark_route_v5.h"

at::Tensor centroids(const at::Tensor &k,int64_t blocks) {
 TORCH_CHECK(k.device().type()==c10::DeviceType::PrivateUse1 && k.scalar_type()==at::kBFloat16 && k.dim()==3 && k.size(2)==128 && k.is_contiguous(),"K requires contiguous HND BF16 on NPU");
 TORCH_CHECK(blocks>0 && blocks*64<=k.size(1),"invalid active block count");
 uint32_t h=k.size(0),n=k.size(1),m=blocks,total=h*m;
 auto out=at::empty({h,m,128},k.options());
 auto platform=platform_ascendc::PlatformAscendCManager::GetInstance();
 uint32_t grid=std::min(total,platform->GetCoreNumAiv());
 auto stream=c10_npu::getCurrentNPUStream().stream(false);void *kp=k.data_ptr(),*op=out.data_ptr();
 auto call=[=]() -> int {auto ret=ACLRT_LAUNCH_KERNEL(spark_kc_v4)(grid,stream,kp,op,n,m,total);TORCH_CHECK(ret==0,"spark_kc launch failed: ",ret);return 0;};
 at_npu::native::OpCommand::RunOpApi("spark_kc",call);return out;
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> partition(const at::Tensor &ids,const at::Tensor &pos,const at::Tensor &scores,const at::Tensor &cuts,int64_t take) {
 TORCH_CHECK(ids.device().type()==c10::DeviceType::PrivateUse1 && ids.dim()==2 && ids.scalar_type()==at::kLong && pos.scalar_type()==at::kLong && pos.sizes()==ids.sizes() && ids.stride(1)==1 && pos.stride(1)==1,"expected row-strided Int64 IDs/positions");
 TORCH_CHECK(scores.scalar_type()==at::kFloat && scores.is_contiguous() && scores.sizes()==ids.sizes() && cuts.scalar_type()==at::kFloat && cuts.is_contiguous() && cuts.numel()==ids.size(0),"invalid scores/cuts");
 TORCH_CHECK(ids.device()==pos.device() && ids.device()==scores.device() && ids.device()==cuts.device(),"device mismatch");
 uint32_t b=ids.size(0),n=ids.size(1),tk=take,is0=ids.stride(0),ps0=pos.stride(0);
 TORCH_CHECK(n%64==0 && take>0 && take<n,"invalid partition capacity");
 auto oi=at::empty(ids.sizes(),ids.options()),op=at::empty(ids.sizes(),pos.options()),status=at::empty({b,8},ids.options().dtype(at::kInt));
 uint32_t grid=std::min(b,platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
 auto stream=c10_npu::getCurrentNPUStream().stream(false);void *ip=ids.data_ptr(),*pp=pos.data_ptr(),*sp=scores.data_ptr(),*cp=cuts.data_ptr(),*oip=oi.data_ptr(),*opp=op.data_ptr(),*st=status.data_ptr();
 auto call=[=]() -> int {auto ret=ACLRT_LAUNCH_KERNEL(spark_partition_v4)(grid,stream,ip,pp,sp,cp,oip,opp,st,b,n,tk,is0,ps0);TORCH_CHECK(ret==0,"spark_partition launch failed: ",ret);return 0;};at_npu::native::OpCommand::RunOpApi("spark_partition",call);return {oi,op,status};
}
at::Tensor route(const at::Tensor &scores,int64_t width) {
 TORCH_CHECK(scores.device().type()==c10::DeviceType::PrivateUse1 && scores.scalar_type()==at::kFloat && scores.dim()==3 && scores.size(1)==scores.size(2) && scores.is_contiguous(),"expected contiguous HMM FP32 scores");
 uint32_t m=scores.size(2),w=width,rows=scores.size(0)*m;TORCH_CHECK(m>0 && m<=4096 && w>0 && w<=m,"unsupported route size");
 auto out=at::empty({scores.size(0),m,((m+127)/128)*128},scores.options().dtype(at::kChar));uint32_t grid=std::min(rows,platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());auto stream=c10_npu::getCurrentNPUStream().stream(false);void *sp=scores.data_ptr(),*op=out.data_ptr();
 auto call=[=]() -> int {auto ret=ACLRT_LAUNCH_KERNEL(spark_route_v5)(grid,stream,sp,op,rows,m,w);TORCH_CHECK(ret==0,"spark_route launch failed: ",ret);return 0;};at_npu::native::OpCommand::RunOpApi("spark_route",call);return out;
}
TORCH_LIBRARY_FRAGMENT(spark_preproc_v4,m){m.def("centroids(Tensor k, int blocks) -> Tensor");m.def("partition(Tensor ids, Tensor positions, Tensor scores, Tensor cuts, int take) -> (Tensor, Tensor, Tensor)");m.def("route(Tensor scores, int width) -> Tensor");}
TORCH_LIBRARY_IMPL(spark_preproc_v4,PrivateUse1,m){m.impl("centroids",TORCH_FN(centroids));m.impl("partition",TORCH_FN(partition));m.impl("route",TORCH_FN(route));}
