#include <ATen/ATen.h>
#include <torch/library.h>
#include <tiling/platform/platform_ascendc.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "aclrtlaunch_spark_partition_fast_v1.h"
std::tuple<at::Tensor,at::Tensor,at::Tensor> partition(const at::Tensor &ids,const at::Tensor &pos,const at::Tensor &scores,const at::Tensor &cuts,int64_t take) {
 TORCH_CHECK(ids.device().type()==c10::DeviceType::PrivateUse1 && ids.dim()==2 && ids.scalar_type()==at::kInt && pos.scalar_type()==at::kInt && pos.sizes()==ids.sizes() && ids.stride(1)==1 && pos.stride(1)==1,"expected row-strided Int32 IDs/positions");
 TORCH_CHECK(scores.scalar_type()==at::kFloat && scores.is_contiguous() && scores.sizes()==ids.sizes() && cuts.scalar_type()==at::kFloat && cuts.is_contiguous() && cuts.numel()==ids.size(0)*2,"invalid scores/cuts");
 TORCH_CHECK(ids.device()==pos.device() && ids.device()==scores.device() && ids.device()==cuts.device(),"device mismatch");
 uint32_t b=ids.size(0),n=ids.size(1),tk=take,is0=ids.stride(0),ps0=pos.stride(0);
 TORCH_CHECK(n%64==0 && take>0 && take<n,"invalid partition capacity");
 uint32_t same=(ids.data_ptr()==pos.data_ptr() && ids.strides()==pos.strides());auto oi=at::empty(ids.sizes(),ids.options()),op=same?oi:at::empty(ids.sizes(),pos.options()),status=at::empty({b,8},ids.options().dtype(at::kInt));
 uint32_t grid=std::min(b,platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
 auto stream=c10_npu::getCurrentNPUStream().stream(false);void *ip=ids.data_ptr(),*pp=pos.data_ptr(),*sp=scores.data_ptr(),*cp=cuts.data_ptr(),*oip=oi.data_ptr(),*opp=op.data_ptr(),*st=status.data_ptr();
 auto call=[=]() -> int {auto ret=ACLRT_LAUNCH_KERNEL(spark_partition_fast_v1)(grid,stream,ip,pp,sp,cp,oip,opp,st,b,n,tk,is0,ps0,same);TORCH_CHECK(ret==0,"spark_partition launch failed: ",ret);return 0;};at_npu::native::OpCommand::RunOpApi("spark_partition",call);return {oi,op,status};
}
TORCH_LIBRARY_FRAGMENT(spark_partition_fast,m){m.def("partition(Tensor ids, Tensor positions, Tensor scores, Tensor cuts, int take) -> (Tensor, Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(spark_partition_fast,PrivateUse1,m){m.impl("partition",TORCH_FN(partition));}
