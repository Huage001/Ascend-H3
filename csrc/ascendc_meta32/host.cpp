#include <ATen/ATen.h>
#include <torch/library.h>
#include <tiling/platform/platform_ascendc.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "aclrtlaunch_spark_terminal_i32.h"
#include "aclrtlaunch_spark_partition_i32.h"
std::tuple<at::Tensor,at::Tensor,at::Tensor> partition(const at::Tensor &ids,const at::Tensor &pos,const at::Tensor &scores,const at::Tensor &cuts,int64_t take) {
 TORCH_CHECK(ids.device().type()==c10::DeviceType::PrivateUse1 && ids.dim()==2 && ids.scalar_type()==at::kInt && pos.scalar_type()==at::kInt && pos.sizes()==ids.sizes() && ids.stride(1)==1 && pos.stride(1)==1,"expected row-strided Int32 IDs/positions");
 TORCH_CHECK(scores.scalar_type()==at::kFloat && scores.is_contiguous() && scores.sizes()==ids.sizes() && cuts.scalar_type()==at::kFloat && cuts.is_contiguous() && cuts.numel()==ids.size(0),"invalid scores/cuts");
 TORCH_CHECK(ids.device()==pos.device() && ids.device()==scores.device() && ids.device()==cuts.device(),"device mismatch");
 uint32_t b=ids.size(0),n=ids.size(1),tk=take,is0=ids.stride(0),ps0=pos.stride(0);
 TORCH_CHECK(n%64==0 && take>0 && take<n,"invalid partition capacity");
 auto oi=at::empty(ids.sizes(),ids.options()),op=at::empty(ids.sizes(),pos.options()),status=at::empty({b,8},ids.options().dtype(at::kInt));
 uint32_t grid=std::min(b,platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
 auto stream=c10_npu::getCurrentNPUStream().stream(false);void *ip=ids.data_ptr(),*pp=pos.data_ptr(),*sp=scores.data_ptr(),*cp=cuts.data_ptr(),*oip=oi.data_ptr(),*opp=op.data_ptr(),*st=status.data_ptr();
 auto call=[=]() -> int {auto ret=ACLRT_LAUNCH_KERNEL(spark_partition_i32)(grid,stream,ip,pp,sp,cp,oip,opp,st,b,n,tk,is0,ps0);TORCH_CHECK(ret==0,"spark_partition launch failed: ",ret);return 0;};at_npu::native::OpCommand::RunOpApi("spark_partition",call);return {oi,op,status};
}
at::Tensor partition_write(const at::Tensor &scores,const at::Tensor &ids,const at::Tensor &heads,const at::Tensor &starts,at::Tensor output){
 TORCH_CHECK(scores.device().type()==c10::DeviceType::PrivateUse1 && scores.scalar_type()==at::kFloat && scores.dim()==2 && scores.size(1)==128 && scores.is_contiguous(),"scores must be contiguous B128 float32");
 TORCH_CHECK(ids.scalar_type()==at::kInt && ids.sizes()==scores.sizes() && ids.stride(1)==1,"ids must be row-strided B128 int32");
 TORCH_CHECK(heads.scalar_type()==at::kInt && starts.scalar_type()==at::kInt && heads.is_contiguous() && starts.is_contiguous() && heads.numel()==ids.size(0) && starts.numel()==ids.size(0),"invalid output row descriptors");
 TORCH_CHECK(output.scalar_type()==at::kInt && output.dim()==2 && output.is_contiguous(),"output must be contiguous HN int32");
 TORCH_CHECK(scores.device()==ids.device() && scores.device()==heads.device() && scores.device()==starts.device() && scores.device()==output.device(),"devices differ");
 uint32_t rows=ids.size(0),stride=ids.stride(0),n=output.size(1);auto status=at::empty({rows,8},ids.options().dtype(at::kInt));uint32_t grid=std::min(rows,platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());auto stream=c10_npu::getCurrentNPUStream().stream(false);
 void *sp=scores.data_ptr(),*ip=ids.data_ptr(),*hp=heads.data_ptr(),*bp=starts.data_ptr(),*op=output.data_ptr(),*st=status.data_ptr();
 auto call=[=]()->int{auto rc=ACLRT_LAUNCH_KERNEL(spark_terminal_i32)(grid,stream,sp,ip,hp,bp,op,st,rows,stride,n);TORCH_CHECK(rc==0,"terminal partition launch failed: ",rc);return 0;};at_npu::native::OpCommand::RunOpApi("spark_terminal_partition",call);return status;
}
TORCH_LIBRARY_FRAGMENT(spark_meta32,m){m.def("partition(Tensor ids, Tensor positions, Tensor scores, Tensor cuts, int take) -> (Tensor, Tensor, Tensor)");m.def("partition_write(Tensor scores, Tensor ids, Tensor heads, Tensor starts, Tensor(a!) output) -> Tensor");}
TORCH_LIBRARY_IMPL(spark_meta32,PrivateUse1,m){m.impl("partition",TORCH_FN(partition));m.impl("partition_write",TORCH_FN(partition_write));}
