// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the BSD 3-Clause License.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include <tiling/platform/platform_ascendc.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

#include "aclrtlaunch_miowtion128pipe3_attention.h"
#include "../op_kernel/miowtion128pipe3_attention_tiling_data.h"
#include "utils.h"

namespace miowtion128pipe3_ext {
namespace {

// The compute pipeline still gathers KV tokens into 128-token tiles. Sparse
// mask blocks may be smaller because the gather path can join adjacent selected
// blocks before QK/PV computation.
constexpr uint32_t kBasicBlock = 128;
constexpr uint32_t kKvSparseBlockAlignment = 64;
constexpr uint32_t kWorkspaceBlockElements = 131072;
constexpr uint32_t kPipelineSlots = 4;
constexpr uint32_t kPackedCacheBytes = 512;
constexpr uint32_t kPackedCacheBits = kPackedCacheBytes * 8;

uint32_t CeilDiv(uint64_t x, uint64_t y)
{
    return x == 0 ? 0 : static_cast<uint32_t>((x + y - 1) / y);
}

uint32_t GetQBlocks(uint32_t seq, uint32_t blockShapeX)
{
    const uint32_t blocksPerShape = CeilDiv(blockShapeX, kBasicBlock);
    const uint32_t fullShapes = seq / blockShapeX;
    return fullShapes * blocksPerShape + CeilDiv(seq - fullShapes * blockShapeX, kBasicBlock);
}

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == DEVICE_TYPE, name, " must be on NPU");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    // Contiguous views are valid: ConvertType passes data_ptr(), including
    // storage_offset. The kernel uses logical shapes and linear strides.
}

at::Tensor MakeInt64DeviceTensor(const std::vector<int64_t> &values, const at::Tensor &like)
{
    auto cpu = at::tensor(values, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    return CopyTensorHostToDevice(cpu).to(like.device(), at::kLong, true, true);
}

uint64_t MakeTilingKey(at::ScalarType dtype, uint32_t layout, uint32_t innerPrecise, bool lse)
{
    uint64_t key = 9000000000000000ULL;
    key += layout == 0 ? 30000000ULL : 50000000ULL;
    key += layout == 0 ? 2ULL : 3ULL;
    if (dtype == at::kBFloat16) {
        key += 22220ULL;
    }
    if (innerPrecise == 1) {
        key += 100000ULL;
    }
    if (lse) {
        key += 100000000ULL;
    }
    return key;
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> run_miowtion128pipe3_attention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &blockSparseMask,
    const at::Tensor &validSizes,
    at::IntArrayRef blockShape,
    const c10::optional<std::vector<int64_t>> &actualSeqLengths,
    const c10::optional<std::vector<int64_t>> &actualSeqLengthsKv,
    c10::string_view inputLayout,
    int64_t numKeyValueHeads,
    double scaleValue,
    int64_t innerPrecise,
    bool returnLse, int64_t computeTile, int64_t stackTiles)
{
    CheckNpuTensor(query, "query");
    TORCH_CHECK(returnLse, "return_lse required for task kernel");
    CheckNpuTensor(validSizes, "valid_sizes");
    TORCH_CHECK(validSizes.scalar_type()==at::kInt && validSizes.device()==query.device(), "valid_sizes must be int32 on same device");
    TORCH_CHECK(query.scalar_type()==at::kBFloat16, "task kernel currently BF16 only");
    TORCH_CHECK(computeTile==128, "compute_tile must be 64 or 128");
    TORCH_CHECK(stackTiles>=1 && stackTiles*computeTile<=512, "invalid stack_tiles");
    CheckNpuTensor(key, "key");
    CheckNpuTensor(value, "value");
    TORCH_CHECK(query.device() == key.device() && query.device() == value.device(),
        "query, key and value must be on the same NPU");
    TORCH_CHECK(c10_npu::current_device() == query.get_device(),
        "inputs must be on the current NPU device");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
        "query dtype must be float16 or bfloat16");
    TORCH_CHECK(key.scalar_type() == query.scalar_type() && value.scalar_type() == query.scalar_type(),
        "query, key and value must have the same dtype");
    CheckNpuTensor(blockSparseMask, "block_sparse_mask");
    TORCH_CHECK(blockSparseMask.device() == query.device(),
        "block_sparse_mask must be on the same NPU as query");
    TORCH_CHECK(blockSparseMask.scalar_type() == at::kByte ||
        blockSparseMask.scalar_type() == at::kChar ||
        blockSparseMask.scalar_type() == at::kBool,
        "block_sparse_mask dtype must be uint8, int8 or bool");
    TORCH_CHECK(blockShape.size() == 2 && blockShape[0] > 0 && blockShape[1] > 0,
        "block_shape must contain two positive integers");
    TORCH_CHECK(blockShape[1] % kKvSparseBlockAlignment == 0,
        "block_shape[1] must be a multiple of 64");
    TORCH_CHECK(numKeyValueHeads > 0, "num_key_value_heads must be positive");
    TORCH_CHECK(innerPrecise == 0 || innerPrecise == 1, "inner_precise must be 0 or 1");
    TORCH_CHECK(!(query.scalar_type() == at::kBFloat16 && innerPrecise == 1),
        "bfloat16 only supports inner_precise=0");

    const bool isTnd = inputLayout == "TND";
    TORCH_CHECK(isTnd || inputLayout == "BNSD", "input_layout must be TND or BNSD");
    TORCH_CHECK(query.dim() == (isTnd ? 3 : 4) && key.dim() == query.dim() && value.dim() == query.dim(),
        "query, key and value rank does not match input_layout");

    uint32_t batch;
    uint32_t numHeads;
    uint32_t kvHeads;
    uint32_t headDim;
    uint32_t maxQSeq;
    uint32_t maxKvSeq;
    std::vector<int64_t> qSeq;
    std::vector<int64_t> kvSeq;
    if (isTnd) {
        TORCH_CHECK(actualSeqLengths && actualSeqLengthsKv,
            "TND layout requires actual_seq_lengths and actual_seq_lengths_kv");
        qSeq = *actualSeqLengths;
        kvSeq = *actualSeqLengthsKv;
        TORCH_CHECK(!qSeq.empty() && qSeq.size() == kvSeq.size(), "sequence length lists must be non-empty and equal");
        batch = qSeq.size();
        numHeads = query.size(1);
        kvHeads = key.size(1);
        headDim = query.size(2);
        TORCH_CHECK(std::accumulate(qSeq.begin(), qSeq.end(), int64_t{0}) == query.size(0),
            "sum(actual_seq_lengths) must equal query.size(0)");
        TORCH_CHECK(std::accumulate(kvSeq.begin(), kvSeq.end(), int64_t{0}) == key.size(0),
            "sum(actual_seq_lengths_kv) must equal key.size(0)");
        maxQSeq = *std::max_element(qSeq.begin(), qSeq.end());
        maxKvSeq = *std::max_element(kvSeq.begin(), kvSeq.end());
    } else {
        batch = query.size(0);
        numHeads = query.size(1);
        maxQSeq = query.size(2);
        headDim = query.size(3);
        kvHeads = key.size(1);
        maxKvSeq = key.size(2);
        TORCH_CHECK(!actualSeqLengths && !actualSeqLengthsKv,
            "BNSD currently accepts neither sequence list; use padded uniform sequences");
        qSeq.assign(batch, maxQSeq);
        kvSeq.assign(batch, maxKvSeq);
    }
    TORCH_CHECK(static_cast<int64_t>(kvHeads) == numKeyValueHeads,
        "num_key_value_heads must equal key's head dimension");
    TORCH_CHECK(numHeads % kvHeads == 0, "query heads must be divisible by KV heads");
    TORCH_CHECK(headDim == 64 || headDim == 128, "head dimension must be 64 or 128");
    TORCH_CHECK(key.sizes().back() == headDim && value.sizes() == key.sizes(),
        "key/value shapes and Q/K/V head dimensions must match");

    TORCH_CHECK(blockShape[1]==128 && batch==1 && validSizes.numel()==CeilDiv(maxKvSeq,128), "valid_sizes expects B1 logical KV128");
    const uint32_t maxQBlocks = CeilDiv(maxQSeq, blockShape[0]);
    const uint32_t maxKvBlocks = CeilDiv(maxKvSeq, blockShape[1]);
    TORCH_CHECK(maxKvBlocks <= kPackedCacheBits,
        "ceil(max_kv_seq/block_shape[1]) must not exceed 4096 because the kernel "
        "uses a 512-byte task-local packed bitmap cache");
    TORCH_CHECK(blockSparseMask.dim() == 4 &&
        blockSparseMask.size(0) == batch && blockSparseMask.size(1) == numHeads &&
        blockSparseMask.size(2) == maxQBlocks && blockSparseMask.size(3) == maxKvBlocks,
        "block_sparse_mask must have shape "
        "[batch, num_heads, ceil(max_q/block_x), ceil(max_kv/block_y)]");

    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "failed to query Ascend platform information");
    const uint32_t aicNum = platform->GetCoreNumAic();

    BlockSparseAttentionTilingData tiling{};
    const char *qGroupEnv = std::getenv("MIOWTION_TASK_QGROUP");
    const int qGroup = qGroupEnv ? std::atoi(qGroupEnv) : 0;
    TORCH_CHECK(qGroup >= 0 && qGroup <= 4096, "task Q group must be in [0,4096]");
    tiling.taskQGroup = static_cast<uint32_t>(qGroup);
    const char *orderEnv = std::getenv("MIOWTION_TASK_QORDER");
    const int qOrder = orderEnv ? std::atoi(orderEnv) : 0;
    TORCH_CHECK(qOrder >= 0 && qOrder <= 2, "task query order must be 0..2");
    tiling.taskQOrder = static_cast<uint32_t>(qOrder);

    tiling.batch = batch;
    tiling.numHeads = numHeads;
    tiling.kvHeads = kvHeads;
    tiling.embeddingSize = headDim;
    // This is the kernel's internal KV processing block, not the user sparse
    // block shape. The source tiling implementation defaults it to 128 even
    // when the (unsupported) paged-cache blockSize attribute is zero.
    tiling.blockSize = computeTile;
    tiling.maskType = stackTiles;
    tiling.scaleValue = static_cast<float>(scaleValue);
    tiling.blockShapeX = blockShape[0];
    tiling.blockShapeY = blockShape[1];
    tiling.maxQBlockNum = maxQBlocks;
    tiling.maxKvBlockNum = maxKvBlocks;
    tiling.queryLayout = isTnd ? 0 : 1;
    tiling.kvCacheLayout = tiling.queryLayout;
    tiling.maxQSeqlen = maxQSeq;
    tiling.maxKvSeqlen = maxKvSeq;
    tiling.useUniformQSeqlen = isTnd ? 0 : 1;
    tiling.useUniformKvSeqlen = isTnd ? 0 : 1;

    TORCH_CHECK(numHeads == kvHeads && blockShape[0] == 128, "pipe3 supports Q128 non-GQA only");
    const uint32_t groupSize = numHeads / kvHeads;
    for (uint32_t i = 0; i < batch; ++i) {
        const uint32_t tasks = GetQBlocks(qSeq[i], blockShape[0]) * groupSize * kvHeads;
        const uint32_t qBlocksForMask = CeilDiv(qSeq[i], blockShape[0]) * numHeads;
        if (i == 0) {
            tiling.firstBatchTaskNum = tasks;
            tiling.firstQBlockNum = qBlocksForMask;
        }
        tiling.totalTaskNum += tasks;
        tiling.totalQBlocks += qBlocksForMask;
    }
    const uint32_t blockDim = std::min(aicNum, tiling.totalTaskNum);
    TORCH_CHECK(blockDim > 0, "empty attention is not supported");

    tiling.mm1OutSize = static_cast<uint64_t>(blockDim) *
        kWorkspaceBlockElements * sizeof(float) * kPipelineSlots;
    tiling.smOnlineOutSize = static_cast<uint64_t>(blockDim) *
        kWorkspaceBlockElements * sizeof(uint16_t) * kPipelineSlots;
    tiling.mm2OutSize = static_cast<uint64_t>(blockDim) *
        kWorkspaceBlockElements * sizeof(float) * kPipelineSlots;
    tiling.updateSize = static_cast<uint64_t>(blockDim) *
        kWorkspaceBlockElements * sizeof(float) * kPipelineSlots;
    tiling.selectIdxSize = static_cast<uint64_t>(CeilDiv(blockShape[0], kBasicBlock)) *
        CeilDiv(maxKvBlocks, 32) * 32 * sizeof(uint32_t) * batch * numHeads * maxQBlocks;
    tiling.selectNumIdxSize = static_cast<uint64_t>(CeilDiv(blockShape[0], kBasicBlock)) *
        sizeof(uint32_t) * 32 * batch * numHeads * maxQBlocks;
    constexpr uint64_t syncSize = sizeof(uint32_t) * 256;
    const uint32_t maskRows = batch * numHeads * maxQBlocks;
    tiling.avgRowNumPerSubCore = CeilDiv(maskRows, blockDim * 2);
    tiling.preActivateSubCoreNum = CeilDiv(maskRows, tiling.avgRowNumPerSubCore);
    tiling.tilingKey = MakeTilingKey(query.scalar_type(), tiling.queryLayout, innerPrecise, returnLse);
    tiling.workSpaceSize = platform->GetLibApiWorkSpaceSize() + tiling.mm1OutSize +
        tiling.smOnlineOutSize + tiling.mm2OutSize + tiling.updateSize +
        tiling.selectNumIdxSize + tiling.selectIdxSize + syncSize;
    auto output = at::empty_like(query);
    auto lse = returnLse
        ? at::empty(isTnd ? at::IntArrayRef{query.size(0), numHeads, 1}
                          : at::IntArrayRef{batch, numHeads, maxQSeq, 1},
              query.options().dtype(at::kFloat))
        : at::empty({1}, query.options().dtype(at::kFloat));
    auto workspace = at::empty({static_cast<int64_t>(tiling.workSpaceSize)}, query.options().dtype(at::kByte));
    auto tilingCpu = at::from_blob(&tiling, {static_cast<int64_t>(sizeof(tiling))},
        at::TensorOptions().dtype(at::kByte)).clone();
    auto dummy = at::empty({1}, query.options().dtype(at::kByte));
    auto tilingDevice = CopyTensorHostToDevice(tilingCpu);
    // Uniform BNSD kernels use maxQSeqlen/maxKvSeqlen from tiling and never
    // dereference these pointers. Avoid constructing two unused device
    // tensors on every invocation.
    auto qSeqDevice = isTnd ? MakeInt64DeviceTensor(qSeq, query) : dummy;
    auto kvSeqDevice = isTnd ? MakeInt64DeviceTensor(kvSeq, query) : dummy;
    auto blockShapeDevice = MakeInt64DeviceTensor({blockShape[0], blockShape[1]}, query);

    EXEC_KERNEL_CMD(miowtion128pipe3_attention, blockDim,
        query, key, value, blockSparseMask, validSizes, blockShapeDevice,
        qSeqDevice, kvSeqDevice, dummy, dummy, dummy, dummy,
        output, lse, workspace, tilingDevice);
    const int64_t sparseMetadataOffset = static_cast<int64_t>(platform->GetLibApiWorkSpaceSize() +
        tiling.mm1OutSize + tiling.smOnlineOutSize + tiling.mm2OutSize + tiling.updateSize);
    auto sparseMetadata = workspace.narrow(0, sparseMetadataOffset,
        static_cast<int64_t>(tiling.selectNumIdxSize + tiling.selectIdxSize));
    return {output, lse, sparseMetadata};
}

} // namespace miowtion128pipe3_ext

namespace {
TORCH_LIBRARY_FRAGMENT(miowtion128pipe3_ext, m)
{
    m.def("forward(Tensor query, Tensor key, Tensor value, "
          "Tensor block_sparse_mask, Tensor valid_sizes, int[] block_shape=[128, 128], "
          "int[]? actual_seq_lengths=None, int[]? actual_seq_lengths_kv=None, "
          "str input_layout=\"BNSD\", int num_key_value_heads=1, float scale_value=0., "
          "int inner_precise=0, bool return_lse=False, int compute_tile=128, int stack_tiles=1) "
          "-> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(miowtion128pipe3_ext, PrivateUse1, m)
{
    m.impl("forward", TORCH_FN(miowtion128pipe3_ext::run_miowtion128pipe3_attention));
}
} // namespace
