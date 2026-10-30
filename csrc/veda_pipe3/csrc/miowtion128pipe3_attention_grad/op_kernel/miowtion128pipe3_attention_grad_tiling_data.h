// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the CANN Open Software License Agreement Version 2.0.

#ifndef ASCEND_KERNEL_BLOCK_SPARSE_ATTENTION_GRAD_TILING_DATA_H
#define ASCEND_KERNEL_BLOCK_SPARSE_ATTENTION_GRAD_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

// Host scheduling and the device-side wide metadata arrays must use the same
// logical-Q capacity.  Keep it next to the shared tiling ABI so changing one
// side cannot silently drop Q tasks on the other side.
constexpr uint32_t BSA_WIDE_Q_TASK_CAPACITY = 48;
constexpr uint32_t BSA_WIDE_PACKET_EDGE_CAPACITY = 8;

// Standalone Ascend-C builds do not run the op-plugin tiling-data generator.
// These PODs intentionally preserve the field order of the source operator's
// BEGIN_TILING_DATA_DEF declarations.
struct BlockSparseAttentionGradTilingDataArch35 {
    uint64_t cubeCoreNum;
    uint64_t batchNum;
    uint64_t qSeqLen;
    uint64_t kvSeqLen;
    uint64_t qGroup;
    uint64_t qHeadNum;
    uint64_t kvHeadNum;
    uint64_t headDim;
    uint64_t baseM;
    uint64_t baseN;
    uint64_t singleM;
    uint64_t dqSize;
    uint64_t dkSize;
    uint64_t dqWorkspaceOffset;
    uint64_t dkWorkspaceOffset;
    uint64_t dvWorkspaceOffset;
    uint64_t sftgWorkspaceOffset;
    float softmaxScale;
    uint32_t sftgTmpSpaceSize;
    uint32_t BlockX;
    uint32_t BlockY;
    SoftMaxTiling softmaxGradFrontTilingData;
    // Standalone dispatch key (the original op framework carried this out of
    // band, while flex-attention launches kernels directly).
    uint64_t tilingKey;
};

struct BlockSparseAttentionGradTilingData {
    uint32_t batch;
    uint32_t numHeads;
    uint32_t kvHeads;
    uint32_t headDim;
    uint32_t totalTaskNum;
    uint32_t maskType;
    float scaleValue;
    uint32_t totalQBlocks;
    uint64_t blockShapeX;
    uint64_t blockShapeY;
    uint32_t inputLayout;
    uint32_t maxQSeqlen;
    uint32_t maxKvSeqlen;
    uint64_t tilingKey;
    uint64_t gradSize;
    uint64_t sOutSize;
    uint64_t dPOutSize;
    uint64_t dQOutSize;
    uint64_t dKOutSize;
    uint64_t dVOutSize;
    uint32_t basicQBlockSize;
    uint32_t basicKVBlockSize;
    // Wide grouped schedule.  A value of zero in wideQTasksPerWorkItem keeps
    // the legacy task and workspace layout.  The workspace element counts are
    // per AIC: one stage holds widePacketEdgeCapacity 128x128 score tiles and
    // the core count includes both ping-pong stages.  wideRawPlaneBytes is the
    // byte offset of the BF16 plane inside each of sOut/dPOut across all AICs.
    uint32_t wideQTasksPerWorkItem;
    uint32_t widePacketEdgeCapacity;
    uint64_t wideWorkspaceStageElements;
    uint64_t wideWorkspaceCoreElements;
    uint64_t wideRawPlaneBytes;
    // Zero preserves contiguous core ownership; otherwise work IDs advance
    // by the AIC count. Operand residency uses at most three Q/dO pairs and
    // does not change the GM workspace layout or size.
    uint32_t wideTaskStride;
    uint32_t wideQOperandCache;
    uint32_t wideFastPipeline;
    uint32_t taskNumPerCore;
    uint32_t tailTaskNum;
    uint32_t preQSeqLengths[64];
    uint32_t preKVSeqLengths[64];
    uint32_t beginBatch[64];
    uint32_t beginHead[64];
    uint32_t beginQSeqOffset[64];
    uint32_t usedVecCoreNum;
    uint32_t qTotalSeqlen;
    uint32_t kvTotalSeqlen;
    uint64_t dqSize;
    uint64_t dkvSize;
    uint64_t postUbBaseSize;
    uint64_t ubSize;
    SoftMaxTiling softmaxGradTilingData;
};

#endif
