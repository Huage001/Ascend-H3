#include <ATen/ATen.h>
#include <torch/library.h>
#include <tiling/platform/platform_ascendc.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "aclrtlaunch_spark_inverse_copy_v1.h"
at::Tensor inverse(const at::Tensor &x,const at::Tensor &ids,int64_t g,int64_t b){
 TORCH_CHECK(x.scalar_type()==at::kBFloat16 && x.is_contiguous() && x.dim()==3 && x.size(2)==128,"expected contiguous H P128 BF16");
 TORCH_CHECK(ids.scalar_type()==at::kLong && ids.is_contiguous() && ids.dim()==2 && ids.size(0)==x.size(0),"expected contiguous H N int64");
 TORCH_CHECK((b==32||b==64||b==128)&&g>=0&&ids.size(1)+g<=x.size(1),"invalid dimensions");
 uint32_t H=x.size(0),N=ids.size(1),G=g,P=x.size(1),B=b;auto out=at::empty({1,int64_t(N+G),H,128},x.options());uint32_t grid=platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();auto stream=c10_npu::getCurrentNPUStream().stream(false);void *xp=x.data_ptr(),*ip=ids.data_ptr(),*op=out.data_ptr();
 auto call=[=]()->int{auto rc=ACLRT_LAUNCH_KERNEL(spark_inverse_copy_v1)(grid,stream,xp,ip,op,H,N,G,P,B);TORCH_CHECK(rc==0,"inverse launch failed",rc);return 0;};at_npu::native::OpCommand::RunOpApi("spark_inverse_copy",call);return out;
}
TORCH_LIBRARY_FRAGMENT(spark_inverse,m){m.def("inverse(Tensor x, Tensor ids, int g, int b) -> Tensor");}
TORCH_LIBRARY_IMPL(spark_inverse,PrivateUse1,m){m.impl("inverse",TORCH_FN(inverse));}
