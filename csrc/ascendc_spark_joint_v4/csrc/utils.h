// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef UTILS_H
#define UTILS_H
#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace spark_joint_v4_ext {

#define DEVICE_TYPE c10::DeviceType::PrivateUse1

inline at::Tensor CopyTensorHostToDevice(const at::Tensor& cpu_tensor)
{
    at::Tensor cpuPinMemTensor = cpu_tensor.pin_memory();
    int deviceIndex = 0;
    c10_npu::GetDevice(&deviceIndex);
    return cpuPinMemTensor.to(c10::Device(DEVICE_TYPE, deviceIndex), cpuPinMemTensor.scalar_type(), true, true);
}

inline at::Tensor CopyScalarToDevice(const c10::Scalar& cpu_scalar, at::ScalarType scalar_data_type)
{
    return CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

inline void *ConvertType(const at::Tensor &at_tensor)
{
    // data_ptr() includes a tensor view's storage_offset.  Using the storage
    // base address silently aliases every narrow/view passed to a custom
    // kernel to byte zero of the backing allocation.
    return const_cast<void *>(at_tensor.data_ptr());
}

template <typename T> T ConvertType(T value)
{
    return value;
}

template <typename... Ts> constexpr auto ConvertTypes(Ts &...args)
{
    return std::make_tuple(ConvertType(args)...);
}


#define EXEC_KERNEL_CMD(kernel_name, blockdim, ...)                                          \
    do {                                                                                     \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                      \
        auto converted_params = ConvertTypes(__VA_ARGS__);                                   \
        auto acl_call = [acl_stream, blockdim, converted_params]() -> int {                  \
            std::apply([&](auto&&... params) {                                               \
                auto ret = ACLRT_LAUNCH_KERNEL(kernel_name)(blockdim, acl_stream, params...); \
                TORCH_CHECK(ret == 0, #kernel_name " launch failed: ", ret);                 \
            }, converted_params);                                                            \
            return 0;                                                                         \
        };                                                                                   \
        at_npu::native::OpCommand::RunOpApi(#kernel_name, acl_call);                         \
    } while (false)
} // namespace spark_joint_v4_ext
#endif
