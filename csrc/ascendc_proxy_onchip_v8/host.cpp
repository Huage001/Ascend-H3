#include <ATen/ATen.h>
#include <torch/library.h>
#include <tiling/platform/platform_ascendc.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "aclrtlaunch_spark_proxy_terminal_scalar_v8.h"
std::tuple<at::Tensor,at::Tensor> directions_v8(const at::Tensor &c,const at::Tensor &d,const at::Tensor &m,bool debug,int64_t normMode){
 TORCH_CHECK(c.device().type()==c10::DeviceType::PrivateUse1 && c.scalar_type()==at::kBFloat16 && c.dim()==3 && c.size(1)==32 && c.size(2)==128 && c.is_contiguous(),"centers must be B32128 BF16");TORCH_CHECK(d.scalar_type()==at::kFloat && d.is_contiguous() && d.dim()==3 && d.size(0)==c.size(0) && d.size(1)==32 && d.size(2)==32,"distances must be B3232 FP32");TORCH_CHECK(m.scalar_type()==at::kInt && m.numel()==32 && m.is_contiguous() && m.device()==c.device() && d.device()==c.device(),"invalid upper mask");uint32_t b=c.size(0),dbg=debug,mode=normMode;
 auto out=at::empty({b,1,128},c.options().dtype(at::kFloat));auto unit=debug?at::empty(c.sizes(),c.options().dtype(at::kFloat)):at::empty({1},c.options().dtype(at::kFloat));uint32_t grid=std::min(b,platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());auto stream=c10_npu::getCurrentNPUStream().stream(false);void *cp=c.data_ptr(),*dp=d.data_ptr(),*mp=m.data_ptr(),*op=out.data_ptr(),*up=unit.data_ptr();auto call=[=]()->int{auto rc=ACLRT_LAUNCH_KERNEL(spark_proxy_terminal_scalar_v8)(grid,stream,cp,dp,mp,op,up,b,dbg,mode);TORCH_CHECK(rc==0,"proxy onchip launch ",rc);return 0;};at_npu::native::OpCommand::RunOpApi("spark_proxy_onchip_v8",call);return {out,unit};
}
TORCH_LIBRARY_FRAGMENT(spark_proxy_onchip_v8,m){m.def("directions(Tensor centers, Tensor distances, Tensor upper_mask, bool debug=False, int norm_mode=0) -> (Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(spark_proxy_onchip_v8,PrivateUse1,m){m.impl("directions",TORCH_FN(directions_v8));}
