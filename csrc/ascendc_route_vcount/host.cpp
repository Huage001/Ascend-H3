#include <ATen/ATen.h>
#include <torch/library.h>
#include <tiling/platform/platform_ascendc.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "aclrtlaunch_spark_route_vcount_v1.h"
at::Tensor route_vcount(const at::Tensor &scores,int64_t width) {
 TORCH_CHECK(scores.device().type()==c10::DeviceType::PrivateUse1 && scores.scalar_type()==at::kFloat && scores.dim()==3 && scores.size(1)==scores.size(2) && scores.is_contiguous(),"expected contiguous HMM FP32 scores");
 uint32_t m=scores.size(2),w=width,rows=scores.size(0)*m;TORCH_CHECK(m>0 && m<=4096 && w>0 && w<=m,"unsupported route size");
 auto out=at::empty({scores.size(0),m,((m+127)/128)*128},scores.options().dtype(at::kChar));uint32_t grid=std::min(rows,platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());auto stream=c10_npu::getCurrentNPUStream().stream(false);void *sp=scores.data_ptr(),*op=out.data_ptr();
 auto call=[=]() -> int {auto ret=ACLRT_LAUNCH_KERNEL(spark_route_vcount_v1)(grid,stream,sp,op,rows,m,w);TORCH_CHECK(ret==0,"spark_route launch failed: ",ret);return 0;};at_npu::native::OpCommand::RunOpApi("spark_route",call);return out;
}
TORCH_LIBRARY_FRAGMENT(spark_route_vcount,m){m.def("route(Tensor scores, int width) -> Tensor");}
TORCH_LIBRARY_IMPL(spark_route_vcount,PrivateUse1,m){m.impl("route",TORCH_FN(route_vcount));}
