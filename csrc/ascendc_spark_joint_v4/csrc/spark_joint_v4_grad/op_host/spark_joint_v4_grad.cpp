// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the CANN Open Software License Agreement Version 2.0.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include <graph/tensor.h>
#include <tiling/platform/platform_ascendc.h>
#include <tiling/tiling_api.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

#include "aclrtlaunch_spark_joint_v4_grad.h"
#include "../op_kernel/spark_joint_v4_grad_tiling_data.h"
#include "utils.h"

namespace spark_joint_v4_ext {
namespace {

constexpr uint32_t kBasicBlock = 128;
constexpr uint32_t kMaxCores = 64;
constexpr uint32_t kWorkspaceBlockElements = 128 * 128 * 2 * 2;
constexpr uint32_t kGroupedWorkspacePlaneElements = 128 * 128 * 16 * 2;
// Logical aggregation spans many Q tasks so a selected KV column can reduce
// several dK/dV contributions before its atomic Fixpipe store.  Physical
// packets remain small to keep the two-stage raw/P/dS workspace resident in
// cache.  Dense columns are split and resumed generically by the kernel.
constexpr uint32_t kWideQTasksPerWorkItem = BSA_WIDE_Q_TASK_CAPACITY;
constexpr uint32_t kWidePacketEdgeCapacity =
    BSA_WIDE_PACKET_EDGE_CAPACITY;
constexpr uint32_t kWideWorkspaceStages = 2;
constexpr uint64_t kGroupedTileElements =
    static_cast<uint64_t>(kBasicBlock) * kBasicBlock;
constexpr uint32_t kFloatBlockElements = 32 / sizeof(float);

uint32_t CeilDiv(uint64_t x, uint64_t y)
{
    return x == 0 ? 0 : static_cast<uint32_t>((x + y - 1) / y);
}

uint32_t GetQBlocks(uint32_t seq, uint32_t blockX)
{
    const uint32_t blocksPerShape = CeilDiv(blockX, kBasicBlock);
    const uint32_t fullShapes = seq / blockX;
    return fullShapes * blocksPerShape + CeilDiv(seq - fullShapes * blockX, kBasicBlock);
}

// Translate an internal <=128-row Q task back to its token offset without
// crossing a user block boundary.  This is also valid for a short final user
// block, so grouping consecutive task indices preserves arbitrary blockX.
uint32_t GetQBlockOffset(uint32_t block, uint32_t blockX)
{
    const uint32_t blocksPerShape = CeilDiv(blockX, kBasicBlock);
    return (block / blocksPerShape) * blockX +
        (block % blocksPerShape) * kBasicBlock;
}

void CheckNpuTensor(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device().type() == DEVICE_TYPE, name, " must be on NPU");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.storage_offset() == 0, name, " must have zero storage offset");
}

at::Tensor MakeInt64DeviceTensor(const std::vector<int64_t> &values, const at::Tensor &like)
{
    auto cpu = at::tensor(values, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    return CopyTensorHostToDevice(cpu).to(like.device(), at::kLong, true, true);
}

template <typename T>
at::Tensor MakeTilingTensor(const T &tiling)
{
    auto cpu = at::from_blob(const_cast<T *>(&tiling), {static_cast<int64_t>(sizeof(T))},
        at::TensorOptions().dtype(at::kByte)).clone();
    return CopyTensorHostToDevice(cpu);
}

void FillSoftmaxGradTiling(uint32_t headDim, uint64_t ubSize,
    BlockSparseAttentionGradTilingData &tiling)
{
    constexpr uint64_t inputBufferLen = 24 * 1024;
    constexpr uint64_t castBufferLen = 48 * 1024;
    const uint64_t outputBufferLen = (castBufferLen + headDim - 1) / headDim * kFloatBlockElements;
    const uint64_t tempBufferLen = 40 * 1024 - outputBufferLen;
    const int64_t rows = inputBufferLen / sizeof(float) / headDim;
    AscendC::SoftMaxGradTilingFunc(ge::Shape({rows, static_cast<int64_t>(headDim)}),
        sizeof(float), tempBufferLen, tiling.softmaxGradTilingData, true);
    tiling.postUbBaseSize = (ubSize - sizeof(tiling) - 2 * 1024) / 3 / 256 * 256;
    tiling.ubSize = ubSize - sizeof(tiling) - 2 * 1024;
}

BlockSparseAttentionGradTilingData MakeArch22Tiling(
    const std::vector<int64_t> &qSeq, const std::vector<int64_t> &kvSeq,
    uint32_t numHeads, uint32_t kvHeads, uint32_t headDim,
    uint32_t maxQSeq, uint32_t maxKvSeq, uint32_t blockX, uint32_t blockY,
    uint32_t layout, uint32_t maskType, float scale, at::ScalarType dtype,
    uint32_t aicNum, uint64_t ubSize)
{
    BlockSparseAttentionGradTilingData tiling{};
    tiling.batch = qSeq.size();
    tiling.numHeads = numHeads;
    tiling.kvHeads = kvHeads;
    tiling.headDim = headDim;
    tiling.maskType = maskType;
    tiling.scaleValue = scale;
    tiling.blockShapeX = blockX;
    tiling.blockShapeY = blockY;
    tiling.inputLayout = layout;
    tiling.maxQSeqlen = maxQSeq;
    tiling.maxKvSeqlen = maxKvSeq;
    tiling.basicQBlockSize = kBasicBlock;
    tiling.basicKVBlockSize = std::min(blockY, kBasicBlock);

    // BNSD/128-KV and BNSD/64x64 can use the same packet pipeline with
    // runtime tile extents. D=128 defaults to cyclic two-Q work items and
    // Q/dO residency regardless of KV head count or sequence length.
    const bool wideEligible = layout == 1 &&
        tiling.basicQBlockSize <= kBasicBlock &&
        tiling.basicKVBlockSize == blockY &&
        (blockY == kBasicBlock || (blockX == 64 && blockY == 64));
    const char *optimizedOverride = std::getenv("BSA_BACKWARD_OPT");
    TORCH_CHECK(optimizedOverride == nullptr ||
        std::strcmp(optimizedOverride, "0") == 0 || std::strcmp(optimizedOverride, "1") == 0,
        "BSA_BACKWARD_OPT must be 0 (legacy) or 1 (optimized)");
    const bool optimizedWide = wideEligible && (optimizedOverride != nullptr ?
        std::strcmp(optimizedOverride, "1") == 0 : headDim == 128);
    // Keep the original KV64 grouped implementation for the legacy override
    // and the default D=64 fallback, allowing same-binary A/B comparisons.
    const bool wideSchedule = wideEligible && (blockY == kBasicBlock || optimizedWide);
    if (wideSchedule) {
        tiling.wideQTasksPerWorkItem = optimizedWide ? 2 : kWideQTasksPerWorkItem;
        tiling.widePacketEdgeCapacity = kWidePacketEdgeCapacity;
        tiling.wideQOperandCache = optimizedWide;
        tiling.wideFastPipeline = optimizedWide;
        tiling.wideWorkspaceStageElements = kGroupedTileElements *
            static_cast<uint64_t>(tiling.widePacketEdgeCapacity);
        tiling.wideWorkspaceCoreElements = tiling.wideWorkspaceStageElements *
            kWideWorkspaceStages;
    }

    // Non-wide KV64 inputs retain their original task mapping and workspace.
    const bool legacyGroupedSchedule = !wideSchedule && layout == 1 &&
        tiling.basicQBlockSize <= kBasicBlock &&
        tiling.basicKVBlockSize == blockY;

    std::vector<uint32_t> tasks(qSeq.size());
    std::vector<uint32_t> qBlocksPerBatch(qSeq.size());
    std::vector<uint32_t> qWorkItemsPerHead(qSeq.size());
    std::vector<uint64_t> qPrefix(qSeq.size() + 1, 0);
    std::vector<uint64_t> kvPrefix(kvSeq.size() + 1, 0);
    for (size_t b = 0; b < qSeq.size(); ++b) {
        const uint32_t qBlocks = GetQBlocks(qSeq[b], blockX);
        qBlocksPerBatch[b] = qBlocks;
        qWorkItemsPerHead[b] = wideSchedule ?
            CeilDiv(qBlocks, tiling.wideQTasksPerWorkItem) : qBlocks;
        tasks[b] = qWorkItemsPerHead[b] * numHeads;
        tiling.totalTaskNum += tasks[b];
        tiling.totalQBlocks += CeilDiv(qSeq[b], blockX) * numHeads;
        qPrefix[b + 1] = qPrefix[b] + qSeq[b];
        kvPrefix[b + 1] = kvPrefix[b] + kvSeq[b];
    }
    const uint32_t blockDim = std::min(aicNum, tiling.totalTaskNum);
    TORCH_CHECK(blockDim > 0 && blockDim <= kMaxCores, "unsupported core/task count for grad tiling");
    tiling.taskNumPerCore = tiling.totalTaskNum / blockDim;
    tiling.tailTaskNum = tiling.totalTaskNum % blockDim;
    if (optimizedWide) {
        tiling.wideTaskStride = blockDim;
    }

    uint32_t globalTask = 0;
    for (uint32_t core = 0; core < blockDim; ++core) {
        uint32_t temp = tiling.wideTaskStride != 0 ? core : globalTask;
        uint32_t batch = 0;
        uint32_t head = 0;
        uint32_t qOffset = 0;
        for (; batch < tasks.size(); ++batch) {
            if (temp < tasks[batch]) {
                break;
            }
            temp -= tasks[batch];
        }
        TORCH_CHECK(batch < tasks.size(), "invalid grad task mapping");

        if (layout == 0) {
            // TND retains its legacy q-major/head-minor order.
            const uint32_t block = temp / numHeads;
            head = temp % numHeads;
            qOffset = GetQBlockOffset(block, blockX);
            tiling.preQSeqLengths[core] = qPrefix[batch] + qOffset;
            tiling.preKVSeqLengths[core] = kvPrefix[batch];
        } else {
            // BNSD is head-major.  On the wide path the local index identifies
            // a wide work item; otherwise it is the original single Q task.
            const uint32_t itemsPerHead = qWorkItemsPerHead[batch];
            head = temp / itemsPerHead;
            const uint32_t itemWithinHead = temp % itemsPerHead;
            const uint32_t firstQTask = wideSchedule ?
                itemWithinHead * tiling.wideQTasksPerWorkItem : itemWithinHead;
            TORCH_CHECK(firstQTask < qBlocksPerBatch[batch],
                "invalid first Q task for wide grad work item");
            qOffset = GetQBlockOffset(firstQTask, blockX);
            tiling.preQSeqLengths[core] = batch * numHeads * maxQSeq + head * maxQSeq + qOffset;
            tiling.preKVSeqLengths[core] = batch * kvHeads * maxKvSeq +
                (head / (numHeads / kvHeads)) * maxKvSeq;
        }
        tiling.beginBatch[core] = batch;
        tiling.beginHead[core] = head;
        tiling.beginQSeqOffset[core] = qOffset;
        globalTask += tiling.taskNumPerCore + (core < tiling.tailTaskNum ? 1 : 0);
    }

    const uint64_t qElements = qPrefix.back() * numHeads * headDim;
    const uint64_t kvElements = kvPrefix.back() * kvHeads * headDim;
    if (wideSchedule) {
        // Each logical plane is [all-core FP32 raw][all-core BF16 result].
        // S/P and dP/dS remain separate, and each section has two stages.
        tiling.wideRawPlaneBytes = static_cast<uint64_t>(blockDim) *
            tiling.wideWorkspaceCoreElements * sizeof(float);
        const uint64_t wideBf16PlaneBytes = static_cast<uint64_t>(blockDim) *
            tiling.wideWorkspaceCoreElements * sizeof(uint16_t);
        tiling.sOutSize = tiling.wideRawPlaneBytes + wideBf16PlaneBytes;
    } else {
        const uint64_t workspaceBytesPerCore = legacyGroupedSchedule ?
            static_cast<uint64_t>(kGroupedWorkspacePlaneElements) *
                (sizeof(float) + sizeof(uint16_t)) :
            static_cast<uint64_t>(kWorkspaceBlockElements) * sizeof(float);
        tiling.sOutSize = static_cast<uint64_t>(blockDim) * workspaceBytesPerCore;
    }
    tiling.dPOutSize = tiling.sOutSize;
    tiling.dQOutSize = qElements * sizeof(float);
    tiling.dKOutSize = kvElements * sizeof(float);
    tiling.dVOutSize = tiling.dKOutSize;
    tiling.gradSize = qPrefix.back() * numHeads * kFloatBlockElements * sizeof(float);
    tiling.usedVecCoreNum = blockDim * 2;
    tiling.qTotalSeqlen = qPrefix.back();
    tiling.kvTotalSeqlen = kvPrefix.back();
    tiling.dqSize = qElements;
    tiling.dkvSize = kvElements;
    tiling.tilingKey = (dtype == at::kBFloat16 ? 10 : 0) + (layout == 1 ? 1 : 0);
    FillSoftmaxGradTiling(headDim, ubSize, tiling);
    return tiling;
}

BlockSparseAttentionGradTilingDataArch35 MakeArch35Tiling(
    uint32_t batch, uint32_t qSeq, uint32_t kvSeq, uint32_t numHeads,
    uint32_t kvHeads, uint32_t headDim, uint32_t blockX, uint32_t blockY,
    uint32_t layout, float scale, uint32_t aicNum, at::ScalarType dtype)
{
    BlockSparseAttentionGradTilingDataArch35 tiling{};
    tiling.cubeCoreNum = aicNum;
    tiling.batchNum = batch;
    tiling.qSeqLen = qSeq;
    tiling.kvSeqLen = kvSeq;
    tiling.qGroup = numHeads / kvHeads;
    tiling.qHeadNum = numHeads;
    tiling.kvHeadNum = kvHeads;
    tiling.headDim = headDim;
    tiling.baseM = std::min(blockX, 128U);
    tiling.baseN = std::min(blockY, 128U);
    tiling.singleM = CeilDiv(2048, tiling.baseM) * tiling.baseM;
    tiling.BlockX = blockX;
    tiling.BlockY = blockY;
    tiling.softmaxScale = scale;
    tiling.tilingKey = 1000 + (dtype == at::kHalf ? 1 : 0) + (layout == 1 ? 2 : (layout == 2 ? 4 : 0));

    const uint64_t qElements = (layout == 2 ? qSeq : static_cast<uint64_t>(batch) * qSeq) * numHeads * headDim;
    const uint64_t kvElements = (layout == 2 ? kvSeq : static_cast<uint64_t>(batch) * kvSeq) * kvHeads * headDim;
    const uint64_t sftgElements = (layout == 2 ? qSeq : static_cast<uint64_t>(batch) * qSeq) * numHeads * 8;
    tiling.dqSize = qElements;
    tiling.dkSize = kvElements;
    tiling.sftgWorkspaceOffset = 0;
    tiling.dqWorkspaceOffset = sftgElements * sizeof(float);
    tiling.dkWorkspaceOffset = tiling.dqWorkspaceOffset + qElements * sizeof(float);
    tiling.dvWorkspaceOffset = tiling.dkWorkspaceOffset + kvElements * sizeof(float);

    constexpr uint32_t elementNum = 8 * 1024;
    const uint32_t rows = elementNum / headDim;
    const ge::Shape shape({rows, static_cast<int64_t>(headDim)});
    tiling.sftgTmpSpaceSize = AscendC::GetSoftMaxGradMaxTmpSize(shape, sizeof(float), true, true);
    AscendC::SoftMaxGradTilingFunc(shape, sizeof(float), tiling.sftgTmpSpaceSize,
        tiling.softmaxGradFrontTilingData, true);
    return tiling;
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> run_spark_joint_v4_grad(
    const at::Tensor &dout, const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const at::Tensor &attentionOut, const at::Tensor &softmaxLse,
    const at::Tensor &blockSparseMask, const c10::optional<at::Tensor> &attentionMask,
    at::IntArrayRef blockShape, const c10::optional<std::vector<int64_t>> &actualSeqLengths,
    const c10::optional<std::vector<int64_t>> &actualSeqLengthsKv, c10::string_view inputLayout,
    int64_t numKeyValueHeads, int64_t maskType, double scaleValue)
{
    for (const auto *entry : {&dout, &query, &key, &value, &attentionOut, &softmaxLse, &blockSparseMask}) {
        CheckNpuTensor(*entry, "spark_joint_v4_grad input");
        TORCH_CHECK(entry->device() == query.device(), "all inputs must be on the same NPU");
    }
    TORCH_CHECK(c10_npu::current_device() == query.get_device(), "inputs must be on the current NPU device");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
        "query dtype must be float16 or bfloat16");
    TORCH_CHECK(dout.scalar_type() == query.scalar_type() && key.scalar_type() == query.scalar_type() &&
        value.scalar_type() == query.scalar_type() && attentionOut.scalar_type() == query.scalar_type(),
        "dout, query, key, value and attention_out must have the same dtype");
    TORCH_CHECK(dout.sizes() == query.sizes() && attentionOut.sizes() == query.sizes(),
        "dout and attention_out must have query's shape");
    TORCH_CHECK(value.sizes() == key.sizes(), "key and value must have the same shape");
    TORCH_CHECK(softmaxLse.scalar_type() == at::kFloat, "softmax_lse must be float32");
    TORCH_CHECK(blockSparseMask.scalar_type() == at::kByte ||
        blockSparseMask.scalar_type() == at::kChar ||
        blockSparseMask.scalar_type() == at::kBool,
        "block_sparse_mask must be uint8, int8 or bool");
    TORCH_CHECK(blockSparseMask.is_contiguous(), "block_sparse_mask must be contiguous");
    if (attentionMask) {
        TORCH_CHECK(attentionMask->device() == query.device() && attentionMask->is_contiguous(),
            "backward block mask must be contiguous and on the same NPU as query");
        TORCH_CHECK(attentionMask->scalar_type() == blockSparseMask.scalar_type() &&
            attentionMask->sizes() == blockSparseMask.sizes(),
            "backward block mask must use the original block_sparse_mask representation");
    }
    TORCH_CHECK(blockShape.size() == 2 && blockShape[0] > 0 && blockShape[1] > 0,
        "block_shape must contain two positive integers");
    TORCH_CHECK(blockShape[1] % 64 == 0,
        "block_shape[1] must be a multiple of 64 on Ascend 910B");
    TORCH_CHECK(numKeyValueHeads > 0 && maskType == 0, "num_key_value_heads must be positive and mask_type must be 0");

    const bool isTnd = inputLayout == "TND";
    TORCH_CHECK(isTnd || inputLayout == "BNSD", "input_layout must be TND or BNSD");
    TORCH_CHECK(query.dim() == (isTnd ? 3 : 4) && key.dim() == query.dim(), "input rank does not match layout");

    uint32_t batch, numHeads, kvHeads, headDim, maxQSeq, maxKvSeq;
    std::vector<int64_t> qSeq, kvSeq;
    if (isTnd) {
        TORCH_CHECK(actualSeqLengths && actualSeqLengthsKv && !actualSeqLengths->empty() &&
            actualSeqLengths->size() == actualSeqLengthsKv->size(),
            "TND requires equally-sized, non-empty actual sequence length lists");
        qSeq = *actualSeqLengths;
        kvSeq = *actualSeqLengthsKv;
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
        qSeq.assign(batch, maxQSeq);
        kvSeq.assign(batch, maxKvSeq);
    }
    TORCH_CHECK(kvHeads == numKeyValueHeads && numHeads % kvHeads == 0,
        "num_key_value_heads must match key and divide query heads");
    TORCH_CHECK(headDim == 64 || headDim == 128,
        "spark_joint_v4_grad supports head dimensions 64 and 128");
    TORCH_CHECK(key.sizes().back() == headDim, "Q/K/V head dimensions must match");
    TORCH_CHECK(blockSparseMask.dim() == 4 && blockSparseMask.size(0) == batch &&
        blockSparseMask.size(1) == numHeads && blockSparseMask.size(2) == CeilDiv(maxQSeq, blockShape[0]) &&
        blockSparseMask.size(3) == CeilDiv(maxKvSeq, blockShape[1]), "invalid block_sparse_mask shape");

    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "failed to query Ascend platform information");
    const uint32_t aicNum = platform->GetCoreNumAic();
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const float scale = static_cast<float>(scaleValue);
    const bool arch35 = platform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_95;

    auto dq = at::empty_like(query);
    auto dk = at::empty_like(key);
    auto dv = at::empty_like(value);
    auto dummy = at::empty({1}, query.options().dtype(at::kByte));
    auto blockShapeDevice = MakeInt64DeviceTensor({blockShape[0], blockShape[1]}, query);
    std::vector<int64_t> qDeviceSeq = qSeq;
    std::vector<int64_t> kvDeviceSeq = kvSeq;
    if (arch35 && isTnd) {
        std::partial_sum(qDeviceSeq.begin(), qDeviceSeq.end(), qDeviceSeq.begin());
        std::partial_sum(kvDeviceSeq.begin(), kvDeviceSeq.end(), kvDeviceSeq.begin());
    }
    auto qSeqDevice = isTnd ? MakeInt64DeviceTensor(qDeviceSeq, query) : dummy;
    auto kvSeqDevice = isTnd ? MakeInt64DeviceTensor(kvDeviceSeq, query) : dummy;
    auto attentionMaskDevice = attentionMask ? *attentionMask : blockSparseMask;

    if (arch35) {
        TORCH_CHECK(blockShape[0] % 64 == 0 && blockShape[1] % 64 == 0,
            "Ascend950 requires block dimensions that are multiples of 64");
        auto tiling = MakeArch35Tiling(batch, isTnd ? query.size(0) : maxQSeq,
            isTnd ? key.size(0) : maxKvSeq, numHeads, kvHeads, headDim,
            blockShape[0], blockShape[1], isTnd ? 2 : 1, scale, aicNum, query.scalar_type());
        const uint64_t userWorkspace = tiling.dvWorkspaceOffset + tiling.dkSize * sizeof(float);
        auto workspace = at::empty({static_cast<int64_t>(platform->GetLibApiWorkSpaceSize() + userWorkspace)},
            query.options().dtype(at::kByte));
        auto tilingDevice = MakeTilingTensor(tiling);
        EXEC_KERNEL_CMD(spark_joint_v4_grad, aicNum, dout, query, key, value, attentionOut,
            softmaxLse, blockSparseMask, attentionMaskDevice, blockShapeDevice, qSeqDevice, kvSeqDevice,
            dq, dk, dv, workspace, tilingDevice);
    } else {
        auto tiling = MakeArch22Tiling(qSeq, kvSeq, numHeads, kvHeads, headDim, maxQSeq, maxKvSeq,
            blockShape[0], blockShape[1], isTnd ? 0 : 1, maskType, scale,
            query.scalar_type(), aicNum, ubSize);
        const uint32_t blockDim = std::min(aicNum, tiling.totalTaskNum);
        const bool groupedSchedule = tiling.wideQTasksPerWorkItem != 0 ||
            (tiling.inputLayout == 1 &&
                tiling.basicQBlockSize <= kBasicBlock &&
                tiling.basicKVBlockSize <= kBasicBlock &&
                tiling.basicKVBlockSize == tiling.blockShapeY);
        const uint64_t packedInputSize = isTnd && groupedSchedule ?
            2 * (tiling.dqSize + tiling.dkvSize) * query.element_size() : 0;
        const uint64_t workspaceSize = platform->GetLibApiWorkSpaceSize() + tiling.sOutSize + tiling.dPOutSize +
            tiling.dQOutSize + tiling.dKOutSize + tiling.dVOutSize + tiling.gradSize + packedInputSize;
        // The fused mixed AIC/AIV kernel initializes its accumulation buffers,
        // computes all three gradients, then scales/casts them to the outputs.
        // Host code is responsible only for allocating the workspace.
        auto workspace = at::empty({static_cast<int64_t>(workspaceSize)}, query.options().dtype(at::kByte));
        auto tilingDevice = MakeTilingTensor(tiling);
        EXEC_KERNEL_CMD(spark_joint_v4_grad, blockDim, dout, query, key, value, attentionOut,
            softmaxLse, blockSparseMask, attentionMaskDevice, blockShapeDevice, qSeqDevice, kvSeqDevice,
            dq, dk, dv, workspace, tilingDevice);
    }
    return {dq, dk, dv};
}

} // namespace spark_joint_v4_ext

namespace {
TORCH_LIBRARY_FRAGMENT(spark_joint_v4_ext, m)
{
    m.def("backward(Tensor dout, Tensor query, Tensor key, Tensor value, "
          "Tensor attention_out, Tensor softmax_lse, Tensor block_sparse_mask, "
          "Tensor? attention_mask=None, int[] block_shape=[128, 128], "
          "int[]? actual_seq_lengths=None, int[]? actual_seq_lengths_kv=None, "
          "str input_layout=\"BNSD\", int num_key_value_heads=1, int mask_type=0, "
          "float scale_value=0.) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(spark_joint_v4_ext, PrivateUse1, m)
{
    m.impl("backward", TORCH_FN(spark_joint_v4_ext::run_spark_joint_v4_grad));
}
} // namespace
