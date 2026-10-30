#include <ATen/ATen.h>
#include <torch/library.h>
#include <tiling/platform/platform_ascendc.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "aclrtlaunch_spark_proxy_four_v1.h"
at::Tensor proxy_four_directions(const at::Tensor& c,const at::Tensor& d,const at::Tensor& mask,int64_t c0,int64_t c1,int64_t c2,int64_t c3){
 TORCH_CHECK(c.device().type()==c10::DeviceType::PrivateUse1 && c.scalar_type()==at::kBFloat16 && c.dim()==3 && c.size(1)==32 && c.size(2)==128 && c.is_contiguous(),"centers B32128 BF16");TORCH_CHECK(d.device()==c.device() && d.scalar_type()==at::kFloat && d.is_contiguous() && d.numel()==c.size(0)*1024,"invalid distances");TORCH_CHECK(mask.device()==c.device() && mask.scalar_type()==at::kInt && mask.is_contiguous() && mask.numel()==32,"invalid mask");TORCH_CHECK(c0>0&&c1>0&&c2>0&&c3>0 && (c0+c1+c2+c3)%32==0,"invalid capacities");uint32_t b=c.size(0),a0=c0,a1=c1,a2=c2,a3=c3;
 auto out=at::empty({b,3,128},c.options().dtype(at::kFloat));auto stream=c10_npu::getCurrentNPUStream().stream(false);uint32_t grid=std::min(b,platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());void* cp=c.data_ptr(),*dp=d.data_ptr(),*mp=mask.data_ptr(),*op=out.data_ptr();auto call=[=]()->int{auto rc=ACLRT_LAUNCH_KERNEL(spark_proxy_four_v1)(grid,stream,cp,dp,mp,op,b,a0,a1,a2,a3);TORCH_CHECK(rc==0,"four proxy launch ",rc);return 0;};at_npu::native::OpCommand::RunOpApi("spark_proxy_four",call);return out;
}
TORCH_LIBRARY_FRAGMENT(spark_proxy_four,m){m.def("directions(Tensor centers, Tensor distances, Tensor mask, int c0, int c1, int c2, int c3) -> Tensor");}
TORCH_LIBRARY_IMPL(spark_proxy_four,PrivateUse1,m){m.impl("directions",TORCH_FN(proxy_four_directions));}
