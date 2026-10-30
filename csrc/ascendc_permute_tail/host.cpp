#include <ATen/ATen.h>
#include <torch/library.h>
#include <tiling/platform/platform_ascendc.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "aclrtlaunch_spark_permute_tail_kv_v1.h"
std::tuple<at::Tensor,at::Tensor> permute(const at::Tensor &k,const at::Tensor &v,const at::Tensor &ids,int64_t g,int64_t tail){
 TORCH_CHECK(k.scalar_type()==at::kBFloat16 && k.is_contiguous() && v.sizes()==k.sizes() && v.is_contiguous() && v.scalar_type()==k.scalar_type() && k.dim()==4 && k.size(0)==1 && k.size(3)==128,"expected BSND BF16");
 TORCH_CHECK(ids.scalar_type()==at::kLong && ids.dim()==2 && ids.is_contiguous() && ids.size(0)==k.size(2) && ids.size(1)+g==k.size(1) && g>=0 && tail>=0 && tail%64==0,"invalid ids or padding");
 uint32_t H=ids.size(0),N=ids.size(1),G=g,P=(N+G+63)/64*64+tail;auto ko=at::empty({H,P,128},k.options()),vo=at::empty_like(ko);uint32_t grid=platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();auto stream=c10_npu::getCurrentNPUStream().stream(false);void *kp=k.data_ptr(),*vp=v.data_ptr(),*ip=ids.data_ptr(),*kop=ko.data_ptr(),*vop=vo.data_ptr();
 auto call=[=]()->int{auto rc=ACLRT_LAUNCH_KERNEL(spark_permute_tail_kv_v1)(grid,stream,kp,vp,ip,kop,vop,H,N,G,P);TORCH_CHECK(rc==0,"KV permutation launch failed",rc);return 0;};at_npu::native::OpCommand::RunOpApi("spark_permute_tail_kv",call);return std::make_tuple(ko,vo);
}
TORCH_LIBRARY_FRAGMENT(spark_permute_tail,m){m.def("permute(Tensor k, Tensor v, Tensor ids, int g, int tail) -> (Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(spark_permute_tail,PrivateUse1,m){m.impl("permute",TORCH_FN(permute));}
