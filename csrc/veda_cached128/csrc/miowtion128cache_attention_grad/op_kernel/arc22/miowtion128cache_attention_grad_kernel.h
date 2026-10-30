/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file miowtion128cache_attention_grad_kernel.h
 * \brief Block Sparse Attention Grad Kernel Implementation
 */

#ifndef BLOCK_SPARSE_ATTENTION_GRAD_KERNEL_H
#define BLOCK_SPARSE_ATTENTION_GRAD_KERNEL_H

#include "attn_infra/bsag_base_defs.hpp"
#include "attn_infra/arch/bsag_arch.hpp"
#include "attn_infra/arch/bsag_cross_core_sync.hpp"
#include "attn_infra/arch/bsag_resource.hpp"
#include "attn_infra/layout/bsag_layout.hpp"

#include "attn_infra/gemm/block/bsag_block_mmad.hpp"
#include "attn_infra/gemm/bsag_gemm_dispatch_policy.hpp"
#include "attn_infra/gemm/bsag_gemm_type.hpp"
#include "attn_infra/epilogue/block/bsag_block_epilogue.hpp"
#include "attn_infra/epilogue/bsag_epilogue_dispatch_policy.hpp"
#include "attn_infra/epilogue/block/block_epilogue_fag_pre.hpp"
#include "attn_infra/epilogue/block/block_epilogue_post.hpp"
#include "attn_infra/epilogue/block/block_epilogue_softmaxgrad.hpp"
#include "attn_infra/epilogue/block/block_epilogue_simply_softmax.hpp"

using namespace NpuArch;

namespace BSA {

    constexpr uint32_t CUBE2VEC = 7;
    constexpr uint32_t VEC2CUBE = 8;
    constexpr uint32_t CUBE2POST = 9;

    constexpr uint64_t WORKSPACE_BLOCK_SIZE = 128 * 128 * 2;
    constexpr uint64_t WORKSPACE_P16_OFFSET_ELEMENT = 128 * 128 * 2; // dp, P 的在workspace相对起始地址后偏移元素个数
    constexpr uint64_t WORKSPACE_P16_OFFSET = 128 * 128 * 4; // dp, P 的在workspace相对起始地址后偏移字节数
    constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = 128 * 128 * 2 * 2;
    constexpr uint32_t GROUPED_Q_BLOCKS = 4;
    constexpr uint32_t GROUPED_ACTIVE_Q_BLOCKS = GROUPED_Q_BLOCKS;
    constexpr uint32_t GROUPED_KV_BLOCKS = 4;
    constexpr uint32_t GROUPED_PACKED_FRAGMENTS_PER_Q = 4;
    constexpr uint32_t GROUPED_TILES = GROUPED_ACTIVE_Q_BLOCKS * GROUPED_KV_BLOCKS;
    // A normal-path packet is bounded by active edges, not by a fixed number
    // of KV columns.  In the most sparse case every edge can belong to a
    // different column, so the unique-KV bound equals the tile capacity.
    constexpr uint32_t GROUPED_PACKET_KV_BLOCKS = GROUPED_TILES;
    constexpr uint64_t GROUPED_TILE_ELEMENTS = 128 * 128;
    constexpr uint64_t GROUPED_WORKSPACE_STAGE_SIZE = GROUPED_TILE_ELEMENTS * GROUPED_TILES;
    constexpr uint64_t GROUPED_WORKSPACE_CORE_SIZE = GROUPED_WORKSPACE_STAGE_SIZE * 2;
    // M1a wide-Q architecture.  These are capacity limits, not user-visible
    // block-shape specializations: one work item may contain up to 48 internal
    // (<=128 row) Q tasks.  The physical packet capacity matches the selected
    // two-stage cap-8 pipeline; dense columns resume through the generic
    // (KV,Q) cursor rather than requiring a larger workspace allocation.
    constexpr uint32_t WIDE_MAX_Q_TASKS = BSA_WIDE_Q_TASK_CAPACITY;
    constexpr uint32_t WIDE_MAX_EDGES = BSA_WIDE_PACKET_EDGE_CAPACITY;
    // dQ owns eight persistent K slots.  Edge capacity may grow independently,
    // but a physical packet must never require more resident KV columns.
    constexpr uint32_t WIDE_MAX_KV_COLUMNS = 8;
    using WideQMask = uint64_t;
    static_assert(WIDE_MAX_Q_TASKS <= sizeof(WideQMask) * 8U,
        "wide Q state mask is too narrow for the configured capacity");
    static_assert(WIDE_MAX_Q_TASKS <= (1U << 8),
        "wide edgeCode reserves only eight bits for qLocal");
    static_assert(WIDE_MAX_EDGES < (1U << 8),
        "wide Q descriptors reserve only eight bits for edge offsets");
    static_assert(WIDE_MAX_KV_COLUMNS <= WIDE_MAX_EDGES,
        "wide KV column capacity cannot exceed packet edge capacity");
    static_assert(WIDE_MAX_KV_COLUMNS <= 8,
        "wide dQ has only eight persistent K cache slots");
    static_assert(WIDE_MAX_KV_COLUMNS <= (1U << 16),
        "wide edgeCode reserves only sixteen bits for kvLocal");
    constexpr uint32_t WIDE_CUBE2VEC_STAGE0 = 5;
    constexpr uint32_t WIDE_CUBE2VEC_STAGE1 = 6;
    constexpr uint32_t WIDE_VEC2CUBE_STAGE0 = 7;
    constexpr uint32_t WIDE_VEC2CUBE_STAGE1 = 8;
    constexpr uint64_t L1_SIZE_OFFSET = 131072;
    constexpr uint32_t PINGPONG_OFFSET_2 = 2;
    constexpr uint32_t PINGPONG_OFFSET_4 = 4;

    template <
        class BlockMmadBSAG1_,
        class BlockMmadBSAG2_,
        class BlockMmadBSAG3_,
        class EpilogueFAGPre_,
        class EpilogueFAGSfmg_,
        class EpilogueFAGOp_,
        class EpilogueFAGPost_,
        uint32_t INPUT_LAYOUT>
    class BlockSparseAttentionGradKernel {
    public:
        using BlockMmadBSAG1 = BlockMmadBSAG1_;
        using BlockMmadBSAG2 = BlockMmadBSAG2_;
        using BlockMmadBSAG3 = BlockMmadBSAG3_;
        using EpilogueFAGPre = EpilogueFAGPre_;
        using EpilogueFAGSfmg = EpilogueFAGSfmg_;
        using EpilogueFAGOp = EpilogueFAGOp_;
        using EpilogueFAGPost = EpilogueFAGPost_;
        using PreParams = typename EpilogueFAGPre::Params;
        using PostParams = typename EpilogueFAGPost::Params;
        using SfmgParams = typename EpilogueFAGSfmg_::Params;
        using SfmParams = typename EpilogueFAGOp_::Params;
        using ArchTag = typename BlockMmadBSAG1_::ArchTag;
        using ElementInput = typename BlockMmadBSAG1::ElementA;

        using ElementA1 = typename BlockMmadBSAG1::ElementA;
        using LayoutA1 = typename BlockMmadBSAG1::LayoutA;
        using LayoutB1 = typename BlockMmadBSAG1::LayoutB;
        using LayoutC1 = typename BlockMmadBSAG1::LayoutC;

        using LayoutA2 = typename BlockMmadBSAG2::LayoutA;
        using LayoutB2 = typename BlockMmadBSAG2::LayoutB;
        using LayoutC2 = typename BlockMmadBSAG2::LayoutC;

        using LayoutA3 = typename BlockMmadBSAG3::LayoutA;
        using LayoutB3 = typename BlockMmadBSAG3::LayoutB;
        using LayoutC3 = typename BlockMmadBSAG3::LayoutC;

        /// Parameters structure
        struct Params {
            // Data members
            GM_ADDR dout;
            GM_ADDR q;
            GM_ADDR k;
            GM_ADDR v;
            GM_ADDR out;
            GM_ADDR softmaxLse;
            GM_ADDR blockSparseMask;
            GM_ADDR blockShape;
            GM_ADDR actualQseqlen;
            GM_ADDR actualKvseqlen;
            GM_ADDR dq;
            GM_ADDR dk;
            GM_ADDR dv;
            GM_ADDR workspace;
            GM_ADDR tiling;

            // Methods
            __aicore__ inline
            Params() {}

            __aicore__ inline
            Params(
                GM_ADDR dout_, GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR out_, GM_ADDR softmaxLse_,
                GM_ADDR blockSparseMask_,
                GM_ADDR blockShape_, GM_ADDR actualQseqlen_, GM_ADDR actualKvseqlen_,
                GM_ADDR dq_, GM_ADDR dk_, GM_ADDR dv_, GM_ADDR workspace_, GM_ADDR tiling_data_
            ) : dout(dout_), q(q_), k(k_), v(v_), out(out_), softmaxLse(softmaxLse_),
                blockSparseMask(blockSparseMask_),
                blockShape(blockShape_), actualQseqlen(actualQseqlen_), actualKvseqlen(actualKvseqlen_),
                dq(dq_), dk(dk_), dv(dv_), workspace(workspace_), tiling(tiling_data_)
            {
            }
        };

        struct TaskInfo {
            uint32_t curBatchIdx;
            uint32_t curHeadIdx;
            uint32_t curQSeqIdx;
            uint32_t curQBlcokIdx;
            uint32_t curCalQSize;
            uint32_t curCalKVSize;
            uint32_t qSeqlen;
            uint32_t kvSeqlen;
            uint64_t qOffset; //Q, dout, dq
            uint64_t kvOffset;//K, V, dk, dv
            uint64_t sOffset; //workspace : S, P, dp, ds
            uint32_t groupQIdx;
            uint32_t groupKvIdx;
            uint32_t kvBasicIdx;
            // Kept explicit because TaskInfo is copied across several AIC
            // scheduling paths; preserving its established layout avoids
            // changing device-side stack/codegen for the tuned wide kernel.
            uint32_t kvBasicIdxSecond;
            uint32_t kvFirstSize;
            uint32_t kvSecondSize;
            uint32_t kvValidSize;
            uint64_t kvSecondOffset;
            uint32_t kvPartCount;
            bool firstQSegment;
        };

        // A wide work item never crosses a batch/head boundary.  Keep the
        // invariant TaskInfo fields once in `current` and retain only the
        // four values that change from one internal Q task to the next.  The
        // 24-byte entry also leaves headroom for q128 without growing the
        // AIV stack proportionally to the full TaskInfo.
        struct WideQInfo {
            uint64_t qOffset;
            uint32_t curQSeqIdx;
            uint32_t curQBlockIdx;
            uint32_t curCalQSize;
        };

        struct SegmentInfo {
            TaskInfo tiles[GROUPED_TILES];
            uint32_t tileCount;
            uint32_t uniqueKvCount;
            uint32_t nextKvStart;
        };

        // A wide packet stores coordinates instead of duplicating TaskInfo for
        // every active edge.  The compact representation serves K-major Cube1 /
        // dK,dV traversal and Q-major dQ/vector traversal.
        struct WideKvDesc {
            uint32_t kvBasicIdx;
            uint16_t edgeBegin;
            uint16_t edgeCount;
        };

        // A physical wide packet contains at most WIDE_MAX_EDGES active Q
        // runs, regardless of the number of Q tasks in its work item.  Keep a
        // compact Q-major view so packet consumers never scan all logical Q
        // tasks.  The byte fields cover the cap-8 packet and keep every
        // descriptor naturally packed into one scalar word.
        struct WideQDesc {
            uint8_t qLocal;
            uint8_t edgeBegin;
            uint8_t edgeCount;
            uint8_t reserved;
        };

        struct WidePacketInfo {
            // low 8 bits: qLocal; high 16 bits: kvLocal.  The workspace slot
            // is the K-major edge index itself.
            uint32_t edgeCode[WIDE_MAX_EDGES];
            WideKvDesc kv[WIDE_MAX_KV_COLUMNS];
            uint16_t qOrder[WIDE_MAX_EDGES];
            WideQDesc activeQ[WIDE_MAX_EDGES];
            uint32_t edgeCount;
            uint32_t uniqueKvCount;
            uint32_t activeQCount;
            uint32_t nextKvStart;
            uint32_t nextQStart;
            uint64_t workspaceBase;
        };

        // Cache one window of the original byte mask across the physical
        // packets belonging to a wide work item.  This is not a packed mask:
        // every uint8 entry retains its original byte and is only held in a
        // uint32 scalar load window to avoid reloading the same Q rows when a
        // cap-limited packet resumes within the same four KV columns.  Four
        // bytes keep the AIV metadata stack below the architectural limit.
        struct WideMaskScanState {
            uint32_t maskWindows[WIDE_MAX_Q_TASKS];
            WideQMask loadedMaskWindows;
            uint32_t maskWindowStart;
            uint32_t maskWindowEnd;
        };

        __aicore__ inline void ResetWideMaskScanState(
            WideMaskScanState &state) const
        {
            // maskWindows is initialized lazily and is read only when the
            // corresponding loaded bit is set.
            state.loadedMaskWindows = 0;
            state.maskWindowStart = ~0U;
            state.maskWindowEnd = 0;
        }

        struct MaskLayoutInfo {
            uint32_t kvBlockCount;
            uint64_t headStride;
            uint64_t batchStride;
        };

        __aicore__ inline uint32_t GetKvBlockLimit(const Params &params, const TaskInfo &info,
            uint32_t kvBlockCount, const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
        {
            return kvBlockCount;
        }

        // Offset of a grouped tile in the head-major packed input workspace.
        __aicore__ inline uint64_t GroupedQHeadOffset(
            const TaskInfo &info,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
        {
            if (!UsePackedTndInputs(tilingData)) {
                return info.qOffset;
            }
            const uint64_t sourceRow = info.qOffset /
                (static_cast<uint64_t>(tilingData->numHeads) * tilingData->headDim);
            const uint64_t batchTokenBase = sourceRow - info.curQSeqIdx;
            return (batchTokenBase * tilingData->numHeads +
                static_cast<uint64_t>(info.curHeadIdx) * info.qSeqlen +
                info.curQSeqIdx) * tilingData->headDim;
        }

        __aicore__ inline uint64_t GroupedKvHeadOffset(
            const TaskInfo &info,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
        {
            if (!UsePackedTndInputs(tilingData)) {
                return info.kvOffset;
            }
            const uint32_t groupSize = tilingData->numHeads / tilingData->kvHeads;
            const uint64_t kvSeqIdx =
                static_cast<uint64_t>(info.kvBasicIdx) * tilingData->basicKVBlockSize;
            const uint64_t sourceRow = info.kvOffset /
                (static_cast<uint64_t>(tilingData->kvHeads) * tilingData->headDim);
            const uint64_t batchTokenBase = sourceRow - kvSeqIdx;
            return (batchTokenBase * tilingData->kvHeads +
                static_cast<uint64_t>(info.curHeadIdx / groupSize) * info.kvSeqlen +
                kvSeqIdx) * tilingData->headDim;
        }

        __aicore__ inline uint64_t GroupedPackedInputBase(
            const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
        {
            return tilingData->sOutSize + tilingData->dPOutSize +
                tilingData->dQOutSize + tilingData->dKOutSize +
                tilingData->dVOutSize + tilingData->gradSize;
        }

        __aicore__ inline void PackTndHeadMajorTensor(
            GM_ADDR source, GM_ADDR destination, uint64_t totalRows,
            AscendC::GlobalTensor<int64_t> actualSeqLengths,
            uint32_t batch, uint32_t heads, uint32_t headDim,
            uint32_t vecCoreIdx, uint32_t vecCoreCount)
        {
            constexpr uint32_t PACK_ROWS = 256;
            constexpr uint32_t PACK_STAGE_BYTES = PACK_ROWS * 128 * sizeof(ElementInput);
            AscendC::GlobalTensor<ElementInput> sourceGm;
            AscendC::GlobalTensor<ElementInput> destinationGm;
            sourceGm.SetGlobalBuffer((__gm__ ElementInput *)source);
            destinationGm.SetGlobalBuffer((__gm__ ElementInput *)destination);
            AscendC::LocalTensor<ElementInput> packUb[2] = {
                resource.ubBuf.template GetBufferByByte<ElementInput>(0),
                resource.ubBuf.template GetBufferByByte<ElementInput>(PACK_STAGE_BYTES)};

            const uint64_t beginRow = totalRows * vecCoreIdx / vecCoreCount;
            const uint64_t endRow = totalRows * (vecCoreIdx + 1) / vecCoreCount;
            if (beginRow == endRow) {
                return;
            }
            uint64_t rowInBatches = beginRow;
            uint64_t sequencePrefix = 0;
            uint32_t batchIdx = 0;
            uint32_t sequenceLength = 0;
            for (; batchIdx < batch; ++batchIdx) {
                sequenceLength = static_cast<uint32_t>(static_cast<int64_t>(
                    actualSeqLengths.GetValue(batchIdx)));
                const uint64_t batchRows = static_cast<uint64_t>(sequenceLength) * heads;
                if (rowInBatches < batchRows) {
                    break;
                }
                rowInBatches -= batchRows;
                sequencePrefix += sequenceLength;
            }
            uint32_t head = static_cast<uint32_t>(rowInBatches / sequenceLength);
            uint32_t sequenceRow = static_cast<uint32_t>(rowInBatches % sequenceLength);
            uint64_t logicalRow = beginRow;
            uint32_t stage = 0;
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            while (logicalRow < endRow) {
                uint64_t rows = sequenceLength - sequenceRow;
                if (rows > PACK_ROWS) {
                    rows = PACK_ROWS;
                }
                if (rows > endRow - logicalRow) {
                    rows = endRow - logicalRow;
                }
                const uint64_t sourceOffset =
                    ((sequencePrefix + sequenceRow) * heads + head) * headDim;
                const uint32_t eventId = stage == 0 ? EVENT_ID0 : EVENT_ID1;
                wait_flag(PIPE_MTE3, PIPE_MTE2, eventId);
                AscendC::DataCopyPad(packUb[stage], sourceGm[sourceOffset],
                    {static_cast<uint16_t>(rows),
                        static_cast<uint32_t>(headDim * sizeof(ElementInput)),
                        static_cast<uint32_t>((heads - 1) * headDim * sizeof(ElementInput)),
                        0, 0},
                    {false, 0, 0, 0});
                set_flag(PIPE_MTE2, PIPE_MTE3, eventId);
                wait_flag(PIPE_MTE2, PIPE_MTE3, eventId);
                AscendC::DataCopy(destinationGm[logicalRow * headDim],
                    packUb[stage], rows * headDim);
                set_flag(PIPE_MTE3, PIPE_MTE2, eventId);
                logicalRow += rows;
                sequenceRow += rows;
                if (sequenceRow == sequenceLength && logicalRow < endRow) {
                    sequenceRow = 0;
                    ++head;
                    if (head == heads) {
                        head = 0;
                        sequencePrefix += sequenceLength;
                        ++batchIdx;
                        sequenceLength = static_cast<uint32_t>(static_cast<int64_t>(
                            actualSeqLengths.GetValue(batchIdx)));
                    }
                }
                stage = 1 - stage;
            }
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        }

        __aicore__ inline void PackGroupedTndInputs(
            const Params &params,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData)
        {
            if (!UsePackedTndInputs(tilingData)) {
                return;
            }
            const uint32_t vecCoreIdx = AscendC::GetBlockIdx();
            const uint32_t vecCoreCount = tilingData->usedVecCoreNum;
            const uint64_t qBytes = tilingData->dqSize * sizeof(ElementInput);
            const uint64_t kvBytes = tilingData->dkvSize * sizeof(ElementInput);
            GM_ADDR packed = params.workspace + GroupedPackedInputBase(tilingData);
            AscendC::GlobalTensor<int64_t> actualQSeqLengths;
            AscendC::GlobalTensor<int64_t> actualKvSeqLengths;
            actualQSeqLengths.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            actualKvSeqLengths.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
            PackTndHeadMajorTensor(params.q, packed,
                tilingData->dqSize / tilingData->headDim,
                actualQSeqLengths, tilingData->batch, tilingData->numHeads,
                tilingData->headDim,
                vecCoreIdx, vecCoreCount);
            PackTndHeadMajorTensor(params.dout, packed + qBytes,
                tilingData->dqSize / tilingData->headDim,
                actualQSeqLengths, tilingData->batch, tilingData->numHeads,
                tilingData->headDim, vecCoreIdx, vecCoreCount);
            PackTndHeadMajorTensor(params.k, packed + 2 * qBytes,
                tilingData->dkvSize / tilingData->headDim,
                actualKvSeqLengths, tilingData->batch, tilingData->kvHeads,
                tilingData->headDim, vecCoreIdx, vecCoreCount);
            PackTndHeadMajorTensor(params.v, packed + 2 * qBytes + kvBytes,
                tilingData->dkvSize / tilingData->headDim,
                actualKvSeqLengths, tilingData->batch, tilingData->kvHeads,
                tilingData->headDim, vecCoreIdx, vecCoreCount);
        }

        __aicore__ inline bool ComputeBlockEnabled(
            const Params &params, uint64_t maskOffset) const
        {
            return ((__gm__ uint8_t *)params.blockSparseMask)[maskOffset] != 0;
        }

        // Load up to eight consecutive entries from the original byte mask.
        // This is only a wider scalar GM transaction: the bytes retain their
        // original representation and are never converted into a persistent
        // bit mask. Typed loads are issued only when naturally aligned and
        // wholly contained in the same physical mask row; unaligned prefixes
        // and row tails fall back to smaller accesses.
        __aicore__ inline uint64_t LoadMaskByteWindow(
            const Params &params, uint64_t absoluteOffset,
            uint32_t byteCount, uint64_t physicalRowEnd) const
        {
            if (byteCount == 0) {
                return 0;
            }
            __gm__ uint8_t *mask = reinterpret_cast<__gm__ uint8_t *>(
                params.blockSparseMask);
            uint64_t values = 0;
            uint32_t copiedBytes = 0;
            while (copiedBytes < byteCount) {
                const uint64_t offset = absoluteOffset + copiedBytes;
                const uint32_t remaining = byteCount - copiedBytes;
                if ((offset & 7ULL) == 0 && remaining >= sizeof(uint64_t) &&
                    offset + sizeof(uint64_t) <= physicalRowEnd) {
                    const uint64_t word =
                        *reinterpret_cast<__gm__ uint64_t *>(mask + offset);
                    values |= word << (copiedBytes * 8U);
                    copiedBytes += sizeof(uint64_t);
                } else if ((offset & 3ULL) == 0 &&
                    remaining >= sizeof(uint32_t) &&
                    offset + sizeof(uint32_t) <= physicalRowEnd) {
                    const uint32_t word =
                        *reinterpret_cast<__gm__ uint32_t *>(mask + offset);
                    values |= static_cast<uint64_t>(word) <<
                        (copiedBytes * 8U);
                    copiedBytes += sizeof(uint32_t);
                } else {
                    values |= static_cast<uint64_t>(mask[offset]) <<
                        (copiedBytes * 8U);
                    ++copiedBytes;
                }
            }
            return values;
        }

        __aicore__ inline MaskLayoutInfo GetMaskLayout(
            const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
        {
            const uint32_t qBlockCount =
                (tilingData->maxQSeqlen + tilingData->blockShapeX - 1) /
                tilingData->blockShapeX;
            const uint32_t kvBlockCount =
                (tilingData->maxKvSeqlen + tilingData->blockShapeY - 1) /
                tilingData->blockShapeY;
            MaskLayoutInfo layout;
            layout.kvBlockCount = kvBlockCount;
            layout.headStride = static_cast<uint64_t>(qBlockCount) * kvBlockCount;
            layout.batchStride = static_cast<uint64_t>(tilingData->numHeads) *
                layout.headStride;
            return layout;
        }

        __aicore__ inline uint64_t GetMaskRowOffset(
            const TaskInfo &qInfo, const MaskLayoutInfo &layout) const
        {
            return static_cast<uint64_t>(qInfo.curBatchIdx) * layout.batchStride +
                static_cast<uint64_t>(qInfo.curHeadIdx) * layout.headStride +
                static_cast<uint64_t>(qInfo.curQBlcokIdx) * layout.kvBlockCount;
        }

        // Scan the original byte mask directly.  Four adjacent uint8 values
        // are fetched together only as a scalar load transaction; no packed
        // or persistent bit mask is produced.  The zero-word fast path is
        // especially useful for highly sparse masks.
        __aicore__ inline void CollectEnabledMaskBlocks(
            const Params &params, uint64_t rowOffset, uint32_t limit,
            uint32_t capacity, uint32_t &cursor, uint32_t *selected,
            uint32_t &selectedCount) const
        {
            __gm__ uint8_t *mask = reinterpret_cast<__gm__ uint8_t *>(
                params.blockSparseMask);
            while (cursor < limit && selectedCount < capacity &&
                ((rowOffset + cursor) & 3U) != 0) {
                if (mask[rowOffset + cursor] != 0) {
                    selected[selectedCount++] = cursor;
                }
                ++cursor;
            }
            while (cursor + 4 <= limit && selectedCount < capacity) {
                const uint64_t offset = rowOffset + cursor;
                const uint32_t values = *reinterpret_cast<__gm__ uint32_t *>(mask + offset);
                if (values == 0) {
                    cursor += 4;
                    continue;
                }
#pragma unroll
                for (uint32_t lane = 0; lane < 4; ++lane) {
                    if (((values >> (lane * 8)) & 0xffU) != 0) {
                        selected[selectedCount++] = cursor;
                    }
                    ++cursor;
                    if (selectedCount == capacity) {
                        return;
                    }
                }
            }
            while (cursor < limit && selectedCount < capacity) {
                if (mask[rowOffset + cursor] != 0) {
                    selected[selectedCount++] = cursor;
                }
                ++cursor;
            }
        }

        __aicore__ inline void SetFlag() {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        }

        __aicore__ inline void WaitFlag() {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        }

        __aicore__ inline void DrainFinalCube1Flags() const
        {
            // No Cube1 operation follows the final grouped segment.  Retire
            // its L1 ping-pong tokens while the final gradient phase can
            // still hide the scalar event latency.  Both L0C tokens are left
            // for the gradient accumulator ping-pong to consume.
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        }

        __aicore__ inline void WaitGroupedFlag() const
        {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        }

        __aicore__ inline void UpdateTaskInfoCalQSize(uint32_t blockShapeX, uint32_t basicQBlockSize, TaskInfo &taskInfo) {
            taskInfo.curQBlcokIdx = taskInfo.curQSeqIdx / blockShapeX;
            uint32_t curQBlcokSeqIdx = taskInfo.curQBlcokIdx * blockShapeX;
            if (curQBlcokSeqIdx + blockShapeX <= taskInfo.qSeqlen) {
                // 没有跨Batch
                if (taskInfo.curQSeqIdx + basicQBlockSize <= curQBlcokSeqIdx + blockShapeX) {
                    // 没有跨BasicBlock，处理一个完整的basicBlock
                    taskInfo.curCalQSize = basicQBlockSize;
                } else {
                    // 处理不完一整个basicBlock，被稀疏Block阻断了，处理长度为当前起始seq到稀疏块末尾
                    taskInfo.curCalQSize = curQBlcokSeqIdx + blockShapeX - taskInfo.curQSeqIdx;
                }
            } else {
                // 跨Batch了，只能处理当前Batch内的一部分
                if (taskInfo.curQSeqIdx + basicQBlockSize <= taskInfo.qSeqlen) {
                    // 没有跨BasicBlock，处理一个完整的basicBlock
                    taskInfo.curCalQSize = basicQBlockSize;
                } else {
                    // 处理不完一整个basicBlock，被Batch阻断了，处理长度为当前起始seq到Batch末尾
                    taskInfo.curCalQSize = taskInfo.qSeqlen - taskInfo.curQSeqIdx;
                }
            }
        }

        __aicore__ inline void initTaskInfo(AscendC::GlobalTensor<int64_t> gActualQseqlen, AscendC::GlobalTensor<int64_t> gActualKvseqlen,
                                            __gm__ BlockSparseAttentionGradTilingData *tilingData,
                                            uint32_t numHeads, uint32_t kvHeads, uint32_t groupSize, uint32_t headDim,
                                            uint32_t maxQSeqlen, uint32_t maxKvSeqlen, uint32_t blockShapeX,
                                            uint32_t basicQBlockSize, uint32_t inputLayout, uint32_t coreIdx,
                                            TaskInfo &taskInfo) {
            //BNSD:curBatch * numHeads * maxQSeqlen + curQSeqOffset; TND:cusum(gActualQseqlen[0:curBatch-1]) + curQSeqOffset
            uint32_t preQSeqLengths = tilingData->preQSeqLengths[coreIdx];
            //BNSD:curBatch * kvHeads * maxKvSeqlen; TND:cusum(gActualKvseqlen[0:curBatch-1])
            uint32_t preKVSeqLengths = tilingData->preKVSeqLengths[coreIdx];
            taskInfo.curBatchIdx =  tilingData->beginBatch[coreIdx];
            taskInfo.curHeadIdx = tilingData->beginHead[coreIdx];
            taskInfo.curQSeqIdx = tilingData->beginQSeqOffset[coreIdx];
            if (inputLayout == 0) {
                taskInfo.qOffset = preQSeqLengths * numHeads * headDim + taskInfo.curHeadIdx * headDim;
                taskInfo.kvOffset = preKVSeqLengths * kvHeads * headDim + taskInfo.curHeadIdx / groupSize * headDim;
                taskInfo.qSeqlen = static_cast<uint32_t>(static_cast<int64_t>(gActualQseqlen.GetValue(taskInfo.curBatchIdx)));
                taskInfo.kvSeqlen = static_cast<uint32_t>(static_cast<int64_t>(gActualKvseqlen.GetValue(taskInfo.curBatchIdx)));
            } else {
                taskInfo.qOffset = preQSeqLengths * headDim;
                taskInfo.kvOffset = preKVSeqLengths * headDim;
                taskInfo.qSeqlen = maxQSeqlen;
                taskInfo.kvSeqlen = maxKvSeqlen;
            }
            UpdateTaskInfoCalQSize(blockShapeX, basicQBlockSize, taskInfo);
        }

        __aicore__ inline void updateNextTaskInfo(AscendC::GlobalTensor<int64_t> gActualQseqlen, AscendC::GlobalTensor<int64_t> gActualKvseqlen,
                                                  uint32_t numHeads, uint32_t kvHeads, uint32_t groupSize, uint32_t headDim,
                                                  uint32_t blockShapeX, uint32_t basicQBlockSize, uint32_t inputLayout,
                                                  const TaskInfo &taskInfo, TaskInfo &nextTask) {
            nextTask = taskInfo;
            if (inputLayout == 0) { // TND, Traverse N-axis first
                if (taskInfo.curHeadIdx == numHeads - 1) {
                    nextTask.qOffset = taskInfo.qOffset + (taskInfo.curCalQSize - 1) * numHeads * headDim + headDim;
                    if (taskInfo.curQSeqIdx + taskInfo.curCalQSize == taskInfo.qSeqlen) {
                        nextTask.curBatchIdx = taskInfo.curBatchIdx + 1;
                        nextTask.curHeadIdx = 0;
                        nextTask.curQSeqIdx = 0;
                        nextTask.qSeqlen = static_cast<uint32_t>(static_cast<int64_t>(gActualQseqlen.GetValue(nextTask.curBatchIdx)));
                        nextTask.kvSeqlen = static_cast<uint32_t>(static_cast<int64_t>(gActualKvseqlen.GetValue(nextTask.curBatchIdx)));
                        nextTask.kvOffset = taskInfo.kvOffset + (taskInfo.kvSeqlen - 1) * kvHeads * headDim + headDim;
                    } else { // batch 不变
                        nextTask.curHeadIdx = 0;
                        nextTask.curQSeqIdx = taskInfo.curQSeqIdx + taskInfo.curCalQSize;
                        nextTask.kvOffset = taskInfo.kvOffset - kvHeads * headDim + headDim;
                    }
                } else { // batch/QSeqIdx 不变
                    nextTask.qOffset = taskInfo.qOffset + headDim;
                    nextTask.curHeadIdx = taskInfo.curHeadIdx + 1;
                    if (nextTask.curHeadIdx % groupSize == 0) {
                        nextTask.kvOffset = taskInfo.kvOffset + headDim;
                    }
                }
            } else { // BNSD, Traverse S-axis first
                nextTask.qOffset = taskInfo.qOffset + taskInfo.curCalQSize * headDim;
                if (taskInfo.curQSeqIdx + taskInfo.curCalQSize == taskInfo.qSeqlen) {
                    nextTask.curQSeqIdx = 0;
                    if (taskInfo.curHeadIdx == numHeads - 1) {
                        nextTask.curBatchIdx = taskInfo.curBatchIdx + 1;
                        nextTask.curHeadIdx = 0;
                        nextTask.kvOffset = taskInfo.kvOffset + taskInfo.kvSeqlen * headDim;
                    } else {
                        nextTask.curHeadIdx = taskInfo.curHeadIdx + 1;
                        if (nextTask.curHeadIdx % groupSize == 0) {
                            nextTask.kvOffset = taskInfo.kvOffset + taskInfo.kvSeqlen * headDim;
                        }
                    }
                } else {
                    nextTask.curQSeqIdx = taskInfo.curQSeqIdx + taskInfo.curCalQSize;
                }
            }
            UpdateTaskInfoCalQSize(blockShapeX, basicQBlockSize, nextTask);
        }

        // The grouped pipeline traverses all Q blocks of one batch/head
        // before advancing the head. BNSD is contiguous in that order;
        // TND uses the same logical order with an N*D physical stride.
        __aicore__ inline void updateNextGroupedTaskInfo(
            AscendC::GlobalTensor<int64_t> gActualQseqlen,
            AscendC::GlobalTensor<int64_t> gActualKvseqlen,
            uint32_t numHeads, uint32_t kvHeads, uint32_t groupSize,
            uint32_t headDim, uint32_t blockShapeX, uint32_t basicQBlockSize,
            uint32_t inputLayout, const TaskInfo &taskInfo, TaskInfo &nextTask)
        {
            if (inputLayout == 1) {
                updateNextTaskInfo(gActualQseqlen, gActualKvseqlen,
                    numHeads, kvHeads, groupSize, headDim, blockShapeX,
                    basicQBlockSize, inputLayout, taskInfo, nextTask);
                return;
            }

            nextTask = taskInfo;
            const uint64_t qStride = static_cast<uint64_t>(numHeads) * headDim;
            const uint64_t kvStride = static_cast<uint64_t>(kvHeads) * headDim;
            nextTask.qOffset = taskInfo.qOffset + taskInfo.curCalQSize * qStride;
            if (taskInfo.curQSeqIdx + taskInfo.curCalQSize == taskInfo.qSeqlen) {
                nextTask.curQSeqIdx = 0;
                if (taskInfo.curHeadIdx == numHeads - 1) {
                    nextTask.curBatchIdx = taskInfo.curBatchIdx + 1;
                    nextTask.curHeadIdx = 0;
                    nextTask.qOffset -= static_cast<uint64_t>(taskInfo.curHeadIdx) * headDim;
                    nextTask.kvOffset += static_cast<uint64_t>(taskInfo.kvSeqlen) * kvStride -
                        static_cast<uint64_t>(kvHeads - 1) * headDim;
                    nextTask.qSeqlen = static_cast<uint32_t>(static_cast<int64_t>(
                        gActualQseqlen.GetValue(nextTask.curBatchIdx)));
                    nextTask.kvSeqlen = static_cast<uint32_t>(static_cast<int64_t>(
                        gActualKvseqlen.GetValue(nextTask.curBatchIdx)));
                } else {
                    nextTask.curHeadIdx = taskInfo.curHeadIdx + 1;
                    nextTask.qOffset = taskInfo.qOffset -
                        static_cast<uint64_t>(taskInfo.curQSeqIdx) * qStride + headDim;
                    if (nextTask.curHeadIdx % groupSize == 0) {
                        nextTask.kvOffset = taskInfo.kvOffset + headDim;
                    }
                }
            } else {
                nextTask.curQSeqIdx = taskInfo.curQSeqIdx + taskInfo.curCalQSize;
            }
            UpdateTaskInfoCalQSize(blockShapeX, basicQBlockSize, nextTask);
        }

        __aicore__ inline bool UseGroupedSchedule(
            const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
        {
            return tilingData->inputLayout == 1 &&
                tilingData->basicQBlockSize <= 128 &&
                tilingData->basicKVBlockSize <= 128 &&
                tilingData->basicKVBlockSize == tilingData->blockShapeY;
        }

        __aicore__ inline bool UseWideSchedule(
            const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
        {
            return tilingData->inputLayout == 1 &&
                tilingData->basicQBlockSize <= 128 &&
                tilingData->basicKVBlockSize <= 128 &&
                tilingData->basicKVBlockSize == tilingData->blockShapeY &&
                tilingData->wideQTasksPerWorkItem > 0 &&
                tilingData->widePacketEdgeCapacity > 0;
        }

        __aicore__ inline bool UsePackedTndInputs(
            const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
        {
            return tilingData->inputLayout == 0 && UseGroupedSchedule(tilingData);
        }

        __aicore__ inline uint32_t GetPackedFragmentCapacity(
            const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
        {
            // Bound one Q's dK/dV FP32 Fixpipe burst to roughly 64 KiB.  The
            // result is rounded to fragment pairs because two sub-128 KV
            // fragments share one 128-token Cube tile.  This scales with the
            // actual fragment/output shape and does not specialize a block.
            constexpr uint32_t FIXPIPE_BURST_BUDGET = 64 * 1024;
            const uint64_t fragmentBytes =
                static_cast<uint64_t>(tilingData->basicKVBlockSize) *
                tilingData->headDim * sizeof(float);
            uint32_t capacity = fragmentBytes == 0 ? 2 :
                static_cast<uint32_t>(FIXPIPE_BURST_BUDGET / fragmentBytes);
            if (capacity < 2) {
                capacity = 2;
            }
            if (capacity > GROUPED_PACKED_FRAGMENTS_PER_Q) {
                capacity = GROUPED_PACKED_FRAGMENTS_PER_Q;
            }
            return capacity & ~1U;
        }

        __aicore__ inline void ClearGroupedPadding(
            const Params &params,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
        {
            if (tilingData->basicKVBlockSize >= 128 ||
                tilingData->blockShapeX < 16) {
                return;
            }
            const uint32_t vecCoreIdx = AscendC::GetBlockIdx();
            const uint32_t coreIdx = vecCoreIdx / 2;
            const uint32_t stage = vecCoreIdx % 2;
            const uint64_t groupedFp32Bytes =
                static_cast<uint64_t>(tilingData->usedVecCoreNum / 2) *
                GROUPED_WORKSPACE_CORE_SIZE * sizeof(float);
            const uint64_t paddingOffset =
                static_cast<uint64_t>(coreIdx) * GROUPED_WORKSPACE_CORE_SIZE +
                static_cast<uint64_t>(stage) * GROUPED_WORKSPACE_STAGE_SIZE +
                static_cast<uint64_t>(GROUPED_TILES - 1) * GROUPED_TILE_ELEMENTS;
            AscendC::GlobalTensor<ElementInput> gPaddingP;
            AscendC::GlobalTensor<ElementInput> gPaddingDs;
            gPaddingP.SetGlobalBuffer((__gm__ ElementInput *)(
                params.workspace + groupedFp32Bytes));
            gPaddingDs.SetGlobalBuffer((__gm__ ElementInput *)(
                params.workspace + tilingData->sOutSize + groupedFp32Bytes));
            AscendC::InitOutput<ElementInput>(
                gPaddingP[paddingOffset], GROUPED_TILE_ELEMENTS, 0);
            AscendC::InitOutput<ElementInput>(
                gPaddingDs[paddingOffset], GROUPED_TILE_ELEMENTS, 0);
        }

        // Extended L1 slots are intentionally reused across the dQ, dV and
        // dK phases.  Drain the seeded event, establish a real MTE1->MTE2
        // dependency after all preceding L1 reads, then restore the token
        // consumed later by PreloadB(slot 3).
        __aicore__ inline void SyncExtendedL1Reuse() const
        {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        }

        __aicore__ inline void BuildGroupedSegment(
            const Params &params, const __gm__ BlockSparseAttentionGradTilingData *tilingData,
            const TaskInfo *qInfos, const uint32_t *qLimits, uint32_t qCount,
            uint32_t kvStart, uint32_t kvCount, uint32_t *packedKvStarts,
            const uint64_t *maskRowOffsets, uint64_t workspaceBase,
            SegmentInfo &segment) const
        {
            segment.tileCount = 0;
            segment.uniqueKvCount = 0;
            segment.nextKvStart = kvStart + kvCount;

            // When a sparse KV block occupies less than one 128-token Cube
            // tile, collect sparse fragments independently for every Q tile
            // and concatenate as many as fit in the hardware tile.  This is
            // capacity based: it applies to any supported fragment size and
            // never keys a path on one user block_shape value.  Keeping the
            // two physical offsets in TaskInfo lets Cube consume one dense
            // compute tile while dK/dV are scattered back to their sources.
            if (qCount >= 1 && tilingData->basicKVBlockSize < 128 &&
                tilingData->blockShapeX >= 16) {
                uint32_t selected[GROUPED_Q_BLOCKS][GROUPED_PACKED_FRAGMENTS_PER_Q] = {};
                uint32_t selectedCount[GROUPED_Q_BLOCKS] = {};
                bool firstSegment[GROUPED_Q_BLOCKS] = {};
                const uint32_t fragmentCapacity =
                    GetPackedFragmentCapacity(tilingData);
                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    uint32_t kvBasicIdx = packedKvStarts[qLocal];
                    firstSegment[qLocal] = kvBasicIdx == 0;
                    CollectEnabledMaskBlocks(params, maskRowOffsets[qLocal],
                        qLimits[qLocal], fragmentCapacity,
                        kvBasicIdx, selected[qLocal], selectedCount[qLocal]);
                    packedKvStarts[qLocal] = kvBasicIdx;
                }
                const uint64_t kvStride = tilingData->inputLayout == 0 ?
                    static_cast<uint64_t>(tilingData->kvHeads) * tilingData->headDim :
                    tilingData->headDim;
                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    const TaskInfo &qInfo = qInfos[qLocal];
                    const uint32_t kvTileCount =
                        (qInfo.kvSeqlen + tilingData->basicKVBlockSize - 1) /
                        tilingData->basicKVBlockSize;
                    uint32_t fragment = 0;
                    uint32_t packedLocal = 0;
                    while (fragment < selectedCount[qLocal]) {
                        const uint32_t firstKv = selected[qLocal][fragment++];
                        const uint32_t firstSize = firstKv + 1 < kvTileCount ?
                            tilingData->basicKVBlockSize :
                            qInfo.kvSeqlen - firstKv * tilingData->basicKVBlockSize;
                        TaskInfo &tile = segment.tiles[segment.tileCount];
                        tile = qInfo;
                        tile.kvBasicIdx = firstKv;
                        tile.kvBasicIdxSecond = 0;
                        tile.kvFirstSize = firstSize;
                        tile.kvSecondSize = 0;
                        tile.kvPartCount = 1;
                        tile.kvOffset = qInfo.kvOffset + static_cast<uint64_t>(firstKv) *
                            tilingData->basicKVBlockSize * kvStride;
                        tile.kvSecondOffset = 0;
                        if (fragment < selectedCount[qLocal]) {
                            const uint32_t secondKv = selected[qLocal][fragment];
                            const uint32_t secondSize = secondKv + 1 < kvTileCount ?
                                tilingData->basicKVBlockSize :
                                qInfo.kvSeqlen - secondKv * tilingData->basicKVBlockSize;
                            if (firstSize + secondSize <= 128) {
                                ++fragment;
                                tile.kvBasicIdxSecond = secondKv;
                                tile.kvSecondSize = secondSize;
                                tile.kvPartCount = 2;
                                tile.kvSecondOffset = qInfo.kvOffset +
                                    static_cast<uint64_t>(secondKv) *
                                    tilingData->basicKVBlockSize * kvStride;
                            }
                        }
                        tile.kvValidSize = tile.kvFirstSize + tile.kvSecondSize;
                        tile.curCalKVSize = 128;
                        tile.sOffset = workspaceBase +
                            static_cast<uint64_t>(segment.tileCount) *
                            GROUPED_TILE_ELEMENTS;
                        tile.groupQIdx = qLocal;
                        tile.groupKvIdx = packedLocal++;
                        tile.firstQSegment = firstSegment[qLocal];
                        ++segment.tileCount;
                    }
                    if (packedLocal > segment.uniqueKvCount) {
                        segment.uniqueKvCount = packedLocal;
                    }
                }
                return;
            }

            // Preserve a four-Q group while filling the packet by active-edge
            // capacity.  A KV column is indivisible: either all enabled Q
            // edges in that column enter this packet, or the whole column is
            // deferred to the next packet.  This keeps dK/dV at one reduction
            // and one Fixpipe flush per physical KV column while allowing a
            // sparse packet to contain as many as GROUPED_TILES unique KVs.
            // The mask remains the original Q-major uint8 tensor.
            if (qCount >= 1 && tilingData->basicKVBlockSize == 128) {
                uint32_t kvBasicIdx = kvStart;
                uint32_t selectedKvCount = 0;
                uint32_t selectedEdgeCount = 0;
                uint32_t selectedKvIndices[GROUPED_PACKET_KV_BLOCKS] = {};
                bool selectedEnabled[GROUPED_PACKET_KV_BLOCKS][GROUPED_Q_BLOCKS] = {};
                const uint64_t kvStride = tilingData->inputLayout == 0 ?
                    static_cast<uint64_t>(tilingData->kvHeads) * tilingData->headDim :
                    tilingData->headDim;
                uint32_t maxKvLimit = 0;
                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    if (qLimits[qLocal] > maxKvLimit) {
                        maxKvLimit = qLimits[qLocal];
                    }
                }
                while (kvBasicIdx < maxKvLimit &&
                    selectedKvCount < GROUPED_KV_BLOCKS &&
                    selectedEdgeCount < GROUPED_TILES) {
                    bool columnEnabled[GROUPED_Q_BLOCKS] = {};
                    uint32_t columnEdgeCount = 0;
                    for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                        if (kvBasicIdx >= qLimits[qLocal]) {
                            continue;
                        }
                        const bool enabled = ComputeBlockEnabled(
                            params, maskRowOffsets[qLocal] + kvBasicIdx);
                        columnEnabled[qLocal] = enabled;
                        columnEdgeCount += enabled ? 1U : 0U;
                    }
                    if (columnEdgeCount == 0) {
                        ++kvBasicIdx;
                        continue;
                    }
                    if (selectedEdgeCount + columnEdgeCount > GROUPED_TILES) {
                        break;
                    }
                    selectedKvIndices[selectedKvCount] = kvBasicIdx;
                    for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                        selectedEnabled[selectedKvCount][qLocal] =
                            columnEnabled[qLocal];
                    }
                    ++selectedKvCount;
                    selectedEdgeCount += columnEdgeCount;
                    ++kvBasicIdx;
                }
                segment.nextKvStart = kvBasicIdx;
                segment.uniqueKvCount = selectedKvCount;
                // The AIV completion protocol publishes one event per Q run,
                // so materialize selected columns in Q-major order.
                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    const TaskInfo &qInfo = qInfos[qLocal];
                    for (uint32_t kvLocal = 0; kvLocal < selectedKvCount; ++kvLocal) {
                        const uint32_t selectedKv = selectedKvIndices[kvLocal];
                        if (!selectedEnabled[kvLocal][qLocal]) {
                            continue;
                        }
                        const uint32_t kvTileCount =
                            (qInfo.kvSeqlen + tilingData->basicKVBlockSize - 1) /
                            tilingData->basicKVBlockSize;
                        TaskInfo &tile = segment.tiles[segment.tileCount];
                        tile = qInfo;
                        tile.curCalKVSize = selectedKv + 1 < kvTileCount ?
                            tilingData->basicKVBlockSize :
                            qInfo.kvSeqlen - selectedKv *
                                tilingData->basicKVBlockSize;
                        tile.kvOffset = qInfo.kvOffset +
                            static_cast<uint64_t>(selectedKv) *
                                tilingData->basicKVBlockSize * kvStride;
                        tile.sOffset = workspaceBase +
                            static_cast<uint64_t>(segment.tileCount) *
                                GROUPED_TILE_ELEMENTS;
                        tile.groupQIdx = qLocal;
                        tile.groupKvIdx = kvLocal;
                        tile.kvBasicIdx = selectedKv;
                        tile.kvBasicIdxSecond = 0;
                        tile.kvFirstSize = tile.curCalKVSize;
                        tile.kvSecondSize = 0;
                        tile.kvSecondOffset = 0;
                        tile.kvPartCount = 1;
                        tile.kvValidSize = tile.curCalKVSize;
                        tile.firstQSegment = kvStart == 0;
                        ++segment.tileCount;
                    }
                }
                return;
            }

            // Retain the original four-column schedule for sub-128 basic KV
            // paths that do not use fragment packing.
            if (qCount >= 1) {
                uint32_t kvBasicIdx = kvStart;
                uint32_t selectedKvCount = 0;
                uint32_t selectedKvIndices[GROUPED_KV_BLOCKS] = {};
                bool selectedEnabled[GROUPED_KV_BLOCKS][GROUPED_Q_BLOCKS] = {};
                const uint64_t kvStride = tilingData->inputLayout == 0 ?
                    static_cast<uint64_t>(tilingData->kvHeads) * tilingData->headDim :
                    tilingData->headDim;
                while (kvBasicIdx < qLimits[0] &&
                    selectedKvCount < GROUPED_KV_BLOCKS) {
                    bool anyEnabled = false;
                    for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                        if (kvBasicIdx >= qLimits[qLocal]) {
                            continue;
                        }
                        const bool enabled = ComputeBlockEnabled(
                            params, maskRowOffsets[qLocal] + kvBasicIdx);
                        selectedEnabled[selectedKvCount][qLocal] = enabled;
                        anyEnabled |= enabled;
                    }
                    if (anyEnabled) {
                        selectedKvIndices[selectedKvCount] = kvBasicIdx;
                        ++selectedKvCount;
                    }
                    ++kvBasicIdx;
                }
                segment.nextKvStart = kvBasicIdx;
                segment.uniqueKvCount = selectedKvCount;
                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    const TaskInfo &qInfo = qInfos[qLocal];
                    for (uint32_t kvLocal = 0; kvLocal < selectedKvCount; ++kvLocal) {
                        const uint32_t selectedKv = selectedKvIndices[kvLocal];
                        if (!selectedEnabled[kvLocal][qLocal]) {
                            continue;
                        }
                        const uint32_t kvTileCount =
                            (qInfo.kvSeqlen + tilingData->basicKVBlockSize - 1) /
                            tilingData->basicKVBlockSize;
                        TaskInfo &tile = segment.tiles[segment.tileCount];
                        tile = qInfo;
                        tile.curCalKVSize = selectedKv + 1 < kvTileCount ?
                            tilingData->basicKVBlockSize :
                            qInfo.kvSeqlen - selectedKv *
                                tilingData->basicKVBlockSize;
                        tile.kvOffset = qInfo.kvOffset +
                            static_cast<uint64_t>(selectedKv) *
                                tilingData->basicKVBlockSize * kvStride;
                        tile.sOffset = workspaceBase +
                            static_cast<uint64_t>(segment.tileCount) *
                                GROUPED_TILE_ELEMENTS;
                        tile.groupQIdx = qLocal;
                        tile.groupKvIdx = kvLocal;
                        tile.kvBasicIdx = selectedKv;
                        tile.kvBasicIdxSecond = 0;
                        tile.kvFirstSize = tile.curCalKVSize;
                        tile.kvSecondSize = 0;
                        tile.kvSecondOffset = 0;
                        tile.kvPartCount = 1;
                        tile.kvValidSize = tile.curCalKVSize;
                        tile.firstQSegment = kvStart == 0;
                        ++segment.tileCount;
                    }
                }
                return;
            }

        }


        __aicore__ inline void RunGroupedCube1CachedB(
            SegmentInfo &segment,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData,
            AscendC::GlobalTensor<ElementInput> gQ,
            AscendC::GlobalTensor<ElementInput> gK,
            AscendC::GlobalTensor<ElementInput> gV,
            AscendC::GlobalTensor<ElementA1> gDout,
            AscendC::GlobalTensor<float> gS,
            AscendC::GlobalTensor<float> gDp,
            BlockMmadBSAG1 &blockMmad1, uint32_t &mmadFlag)
        {
            const uint32_t headDim = tilingData->headDim;
            const uint64_t qStride = UsePackedTndInputs(tilingData) ? headDim :
                (tilingData->inputLayout == 0 ?
                    static_cast<uint64_t>(tilingData->numHeads) * headDim : headDim);
            const uint64_t kvStride = UsePackedTndInputs(tilingData) ? headDim :
                (tilingData->inputLayout == 0 ?
                    static_cast<uint64_t>(tilingData->kvHeads) * headDim : headDim);
            for (uint32_t kvLocal = 0; kvLocal < segment.uniqueKvCount; ++kvLocal) {
                int32_t first = -1;
                for (uint32_t t = 0; t < segment.tileCount; ++t) {
                    if (segment.tiles[t].groupKvIdx == kvLocal) {
                        first = static_cast<int32_t>(t);
                        break;
                    }
                }
                if (first < 0) {
                    continue;
                }
                TaskInfo &firstTile = segment.tiles[first];
                LayoutB1 layoutB(headDim, firstTile.curCalKVSize, kvStride);
                GemmCoord preloadShape{firstTile.curCalQSize, firstTile.curCalKVSize, headDim};
                const uint64_t packedKvOffset = GroupedKvHeadOffset(firstTile, tilingData);
                blockMmad1.PreloadB(gK[packedKvOffset], layoutB, preloadShape, 0);
                blockMmad1.PreloadB(gV[packedKvOffset], layoutB, preloadShape, 1);
                for (uint32_t t = 0; t < segment.tileCount; ++t) {
                    TaskInfo &tile = segment.tiles[t];
                    if (tile.groupKvIdx != kvLocal) {
                        continue;
                    }
                    LayoutA1 layoutA(tile.curCalQSize, headDim, qStride);
                    LayoutC1 layoutC(tile.curCalQSize, tile.curCalKVSize);
                    GemmCoord shape{tile.curCalQSize, tile.curCalKVSize, headDim};
                    const uint64_t packedQOffset = GroupedQHeadOffset(tile, tilingData);
                    blockMmad1.WithCachedB(0, gQ[packedQOffset], gS[tile.sOffset],
                        layoutA, layoutB, layoutC, shape, mmadFlag);
                    blockMmad1.WithCachedB(1, gDout[packedQOffset], gDp[tile.sOffset],
                        layoutA, layoutB, layoutC, shape, mmadFlag);
                }
            }
        }

        __aicore__ inline uint32_t WideEdgeQ(uint32_t edgeCode) const
        {
            return edgeCode & 0xffU;
        }

        __aicore__ inline uint32_t WideEdgeKv(uint32_t edgeCode) const
        {
            return (edgeCode >> 8) & 0xffffU;
        }

        __aicore__ inline void BuildWidePacket(
            const Params &params,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData,
            const uint32_t *qLimits,
            const uint64_t *maskRowOffsets,
            const MaskLayoutInfo &maskLayout, uint32_t qCount,
            uint32_t maxKvLimit,
            uint32_t kvStart, uint32_t qStart, uint64_t workspaceBase,
            WideMaskScanState &maskState, WidePacketInfo &packet) const
        {
            packet.edgeCount = 0;
            packet.uniqueKvCount = 0;
            packet.activeQCount = 0;
            packet.nextKvStart = kvStart;
            packet.nextQStart = qStart;
            packet.workspaceBase = workspaceBase;
            const uint32_t requestedEdgeCapacity =
                tilingData->widePacketEdgeCapacity < WIDE_MAX_EDGES ?
                tilingData->widePacketEdgeCapacity : WIDE_MAX_EDGES;
            // A malformed zero capacity must still make cursor progress; the
            // host currently supplies eight, but this keeps the generic
            // device builder total and prevents an infinite packet loop.
            const uint32_t edgeCapacity = requestedEdgeCapacity == 0 ?
                1 : requestedEdgeCapacity;

            uint32_t kvBasicIdx = kvStart;
            uint32_t qScanStart = qStart;
            if (qScanStart >= qCount && kvBasicIdx < maxKvLimit) {
                ++kvBasicIdx;
                qScanStart = 0;
                packet.nextKvStart = kvBasicIdx;
                packet.nextQStart = 0;
            }
            // Cache one stable four-column byte window per Q row.  The
            // packet remains KV-major, so column grouping and K/V reuse are
            // unchanged; inner-loop mask tests become scalar shifts instead
            // of one strided GM byte transaction per (KV,Q) candidate.
            constexpr uint32_t MASK_WINDOW_BYTES = sizeof(uint32_t);
            while (kvBasicIdx < maxKvLimit && packet.edgeCount < edgeCapacity &&
                packet.uniqueKvCount < WIDE_MAX_KV_COLUMNS) {
                if (kvBasicIdx < maskState.maskWindowStart ||
                    kvBasicIdx >= maskState.maskWindowEnd) {
                    maskState.maskWindowStart =
                        kvBasicIdx & ~(MASK_WINDOW_BYTES - 1U);
                    maskState.maskWindowEnd =
                        maskState.maskWindowStart + MASK_WINDOW_BYTES;
                    if (maskState.maskWindowEnd > maxKvLimit) {
                        maskState.maskWindowEnd = maxKvLimit;
                    }
                    maskState.loadedMaskWindows = 0;
                }
                const uint32_t kvLocal = packet.uniqueKvCount;
                const uint32_t columnBegin = packet.edgeCount;
                uint32_t qLocal = qScanStart;
                const uint32_t maskShift =
                    (kvBasicIdx - maskState.maskWindowStart) * 8U;
                for (; qLocal < qCount; ++qLocal) {
                    if (kvBasicIdx >= qLimits[qLocal]) {
                        continue;
                    }
                    const WideQMask qWindowBit =
                        static_cast<WideQMask>(1) << qLocal;
                    if ((maskState.loadedMaskWindows & qWindowBit) == 0) {
                        const uint32_t qWindowCandidateEnd =
                            maskState.maskWindowStart + MASK_WINDOW_BYTES;
                        const uint32_t qWindowEnd =
                            qWindowCandidateEnd < qLimits[qLocal] ?
                            qWindowCandidateEnd : qLimits[qLocal];
                        maskState.maskWindows[qLocal] = LoadMaskByteWindow(params,
                            maskRowOffsets[qLocal] + maskState.maskWindowStart,
                            qWindowEnd - maskState.maskWindowStart,
                            maskRowOffsets[qLocal] + maskLayout.kvBlockCount);
                        maskState.loadedMaskWindows |= qWindowBit;
                    }
                    if (((maskState.maskWindows[qLocal] >> maskShift) &
                        0xffU) == 0) {
                        continue;
                    }
                    if (packet.edgeCount == edgeCapacity) {
                        break;
                    }
                    packet.edgeCode[packet.edgeCount] =
                        qLocal | (kvLocal << 8);
                    ++packet.edgeCount;
                }

                if (packet.edgeCount != columnBegin) {
                    WideKvDesc &kv = packet.kv[packet.uniqueKvCount++];
                    kv.kvBasicIdx = kvBasicIdx;
                    kv.edgeBegin = static_cast<uint16_t>(columnBegin);
                    kv.edgeCount = static_cast<uint16_t>(
                        packet.edgeCount - columnBegin);
                }

                if (qLocal < qCount) {
                    // The physical packet is full in the middle of a dense
                    // KV column.  Resume at the first unconsumed Q edge in the
                    // next packet.  dK/dV already use atomic final stores, so
                    // the partial-column reductions remain exact while common
                    // sparse columns are still reduced into one store.
                    packet.nextKvStart = kvBasicIdx;
                    packet.nextQStart = qLocal;
                    break;
                }
                ++kvBasicIdx;
                qScanStart = 0;
                packet.nextKvStart = kvBasicIdx;
                packet.nextQStart = 0;
            }

            // Stable-sort only the packet-local edge indices.  This produces
            // exactly the same Q-major order as the old full-Q counting
            // sort (ascending qLocal, original KV order within one Q) without
            // qCounts/qCursor stack arrays or full-Q scans.
            for (uint32_t edge = 0; edge < packet.edgeCount; ++edge) {
                packet.qOrder[edge] = static_cast<uint16_t>(edge);
            }
            for (uint32_t ordered = 1; ordered < packet.edgeCount; ++ordered) {
                const uint16_t key = packet.qOrder[ordered];
                const uint32_t keyQ = WideEdgeQ(packet.edgeCode[key]);
                uint32_t insert = ordered;
                while (insert > 0) {
                    const uint16_t previous = packet.qOrder[insert - 1];
                    if (WideEdgeQ(packet.edgeCode[previous]) <= keyQ) {
                        break;
                    }
                    packet.qOrder[insert] = previous;
                    --insert;
                }
                packet.qOrder[insert] = key;
            }
            for (uint32_t ordered = 0; ordered < packet.edgeCount; ++ordered) {
                const uint32_t qLocal = WideEdgeQ(
                    packet.edgeCode[packet.qOrder[ordered]]);
                if (packet.activeQCount == 0 ||
                    packet.activeQ[packet.activeQCount - 1].qLocal != qLocal) {
                    WideQDesc &desc = packet.activeQ[packet.activeQCount++];
                    desc.qLocal = static_cast<uint8_t>(qLocal);
                    desc.edgeBegin = static_cast<uint8_t>(ordered);
                    desc.edgeCount = 1;
                    desc.reserved = 0;
                } else {
                    ++packet.activeQ[packet.activeQCount - 1].edgeCount;
                }
            }
        }

        __aicore__ inline TaskInfo MakeWideQTask(
            const TaskInfo &workInfo, const WideQInfo *qInfos,
            uint32_t qLocal) const
        {
            TaskInfo task = workInfo;
            task.qOffset = qInfos[qLocal].qOffset;
            task.curQSeqIdx = qInfos[qLocal].curQSeqIdx;
            task.curQBlcokIdx = qInfos[qLocal].curQBlockIdx;
            task.curCalQSize = qInfos[qLocal].curCalQSize;
            return task;
        }

        __aicore__ inline TaskInfo MakeWideTile(
            const TaskInfo &workInfo, const WideQInfo *qInfos,
            const WidePacketInfo &packet,
            uint32_t edgeIndex,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData) const
        {
            const uint32_t edgeCode = packet.edgeCode[edgeIndex];
            const uint32_t qLocal = WideEdgeQ(edgeCode);
            const uint32_t kvLocal = WideEdgeKv(edgeCode);
            const uint32_t kvBasicIdx = packet.kv[kvLocal].kvBasicIdx;
            TaskInfo tile = MakeWideQTask(workInfo, qInfos, qLocal);
            const uint32_t kvTileCount =
                (tile.kvSeqlen + tilingData->basicKVBlockSize - 1) /
                tilingData->basicKVBlockSize;
            tile.curCalKVSize = kvBasicIdx + 1 < kvTileCount ?
                tilingData->basicKVBlockSize :
                tile.kvSeqlen - kvBasicIdx * tilingData->basicKVBlockSize;
            tile.kvOffset += static_cast<uint64_t>(kvBasicIdx) *
                tilingData->basicKVBlockSize * tilingData->headDim;
            tile.sOffset = packet.workspaceBase +
                static_cast<uint64_t>(edgeIndex) * GROUPED_TILE_ELEMENTS;
            tile.groupQIdx = qLocal;
            tile.groupKvIdx = kvLocal;
            tile.kvBasicIdx = kvBasicIdx;
            tile.kvBasicIdxSecond = 0;
            tile.kvFirstSize = tile.curCalKVSize;
            tile.kvSecondSize = 0;
            tile.kvValidSize = tile.curCalKVSize;
            tile.kvSecondOffset = 0;
            tile.kvPartCount = 1;
            tile.firstQSegment = false;
            return tile;
        }

        __aicore__ inline void RunWideCube1(
            const WidePacketInfo &packet, const TaskInfo &workInfo,
            const WideQInfo *qInfos,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData,
            AscendC::GlobalTensor<ElementInput> gQ,
            AscendC::GlobalTensor<ElementInput> gK,
            AscendC::GlobalTensor<ElementInput> gV,
            AscendC::GlobalTensor<ElementA1> gDout,
            AscendC::GlobalTensor<float> gS,
            AscendC::GlobalTensor<float> gDp,
            BlockMmadBSAG1 &blockMmad1, uint32_t &wideCubeFlag,
            BlockMmadBSAG3 &blockMmad3)
        {
            const uint32_t headDim = tilingData->headDim;
            for (uint32_t kvLocal = 0; kvLocal < packet.uniqueKvCount; ++kvLocal) {
                const WideKvDesc &kv = packet.kv[kvLocal];
                TaskInfo firstTile = MakeWideTile(
                    workInfo, qInfos, packet, kv.edgeBegin, tilingData);
                LayoutB1 layoutB(headDim, firstTile.curCalKVSize, headDim);
                GemmCoord preloadShape{
                    firstTile.curCalQSize, firstTile.curCalKVSize, headDim};
                blockMmad1.PreloadB(gK[firstTile.kvOffset], layoutB,
                    preloadShape, 0, tilingData->wideFastPipeline != 0);
                blockMmad1.PreloadB(gV[firstTile.kvOffset], layoutB,
                    preloadShape, 1, tilingData->wideFastPipeline != 0);
                const uint32_t edgeEnd = kv.edgeBegin + kv.edgeCount;
                for (uint32_t edge = kv.edgeBegin; edge < edgeEnd; ++edge) {
                    TaskInfo tile = MakeWideTile(
                        workInfo, qInfos, packet, edge, tilingData);
                    LayoutA1 layoutA(tile.curCalQSize, headDim, headDim);
                    GemmCoord shape{
                        tile.curCalQSize, tile.curCalKVSize, headDim};
                    const bool cacheQ = tilingData->wideQOperandCache != 0;
                    auto localQ = blockMmad3.GetCachedB(cacheQ ?
                        tile.groupQIdx + GROUPED_Q_BLOCKS : 0);
                    auto localDo = blockMmad3.GetCachedB(cacheQ ? tile.groupQIdx : 0);
                    blockMmad1.WithCachedBNz(0, gQ[tile.qOffset],
                        gS[tile.sOffset], layoutA, layoutB, shape,
                        wideCubeFlag, cacheQ ? &localQ : nullptr, edge + 1 == edgeEnd);
                    blockMmad1.WithCachedBNz(1, gDout[tile.qOffset],
                        gDp[tile.sOffset], layoutA, layoutB, shape,
                        wideCubeFlag, cacheQ ? &localDo : nullptr, edge + 1 == edgeEnd);
                }
            }
        }

        __aicore__ inline void RunGroupedCube1Packed(
            SegmentInfo &segment,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData,
            AscendC::GlobalTensor<ElementInput> gQ,
            AscendC::GlobalTensor<ElementInput> gK,
            AscendC::GlobalTensor<ElementInput> gV,
            AscendC::GlobalTensor<ElementA1> gDout,
            AscendC::GlobalTensor<float> gS,
            AscendC::GlobalTensor<float> gDp,
            AscendC::GlobalTensor<ElementInput> gZero,
            BlockMmadBSAG1 &blockMmad1, uint32_t &mmadFlag)
        {
            const uint32_t headDim = tilingData->headDim;
            const uint64_t qStride = tilingData->inputLayout == 0 ?
                static_cast<uint64_t>(tilingData->numHeads) * headDim : headDim;
            const uint64_t kvStride = tilingData->inputLayout == 0 ?
                static_cast<uint64_t>(tilingData->kvHeads) * headDim : headDim;
            const uint64_t zeroOffset = segment.tiles[0].sOffset +
                static_cast<uint64_t>(GROUPED_TILES - 1) * GROUPED_TILE_ELEMENTS;
            for (uint32_t t = 0; t < segment.tileCount; ++t) {
                TaskInfo &tile = segment.tiles[t];
                LayoutB1 firstLayout(headDim, tile.kvFirstSize, kvStride);
                LayoutB1 secondLayout(headDim, tile.kvSecondSize, kvStride);
                LayoutB1 paddingLayout(headDim,
                    tile.curCalKVSize - tile.kvValidSize, headDim);
                GemmCoord shape{tile.curCalQSize, tile.curCalKVSize, headDim};
                blockMmad1.PreloadGatheredBAlongN(
                    gK[tile.kvOffset], firstLayout,
                    gK[tile.kvSecondOffset], secondLayout,
                    gZero[zeroOffset], paddingLayout,
                    tile.kvFirstSize, tile.kvSecondSize, shape, 0);
                blockMmad1.PreloadGatheredBAlongN(
                    gV[tile.kvOffset], firstLayout,
                    gV[tile.kvSecondOffset], secondLayout,
                    gZero[zeroOffset], paddingLayout,
                    tile.kvFirstSize, tile.kvSecondSize, shape, 1);
                LayoutA1 layoutA(tile.curCalQSize, headDim, qStride);
                LayoutB1 combinedLayout(headDim, tile.curCalKVSize);
                LayoutC1 layoutC(tile.curCalQSize, tile.curCalKVSize);
                const uint64_t qOffset = GroupedQHeadOffset(tile, tilingData);
                blockMmad1.WithCachedB(0, gQ[qOffset], gS[tile.sOffset],
                    layoutA, combinedLayout, layoutC, shape, mmadFlag);
                blockMmad1.WithCachedB(1, gDout[qOffset], gDp[tile.sOffset],
                    layoutA, combinedLayout, layoutC, shape, mmadFlag);
            }
        }

        __aicore__ inline void RunGroupedPackedGrad(
            SegmentInfo &segment,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData,
            AscendC::GlobalTensor<ElementInput> gK,
            AscendC::GlobalTensor<ElementInput> gQ,
            AscendC::GlobalTensor<ElementA1> gDout,
            AscendC::GlobalTensor<ElementInput> gP,
            AscendC::GlobalTensor<ElementInput> gDs,
            AscendC::GlobalTensor<float> gDq,
            AscendC::GlobalTensor<float> gDk,
            AscendC::GlobalTensor<float> gDv,
            BlockMmadBSAG2 &blockMmad2, BlockMmadBSAG3 &blockMmad3,
            uint32_t &mmadFlag, uint64_t *cachedQOffsets,
            uint32_t &cachedQMask, bool finalSegment)
        {
            const uint32_t headDim = tilingData->headDim;
            const uint64_t kvInputStride = tilingData->inputLayout == 0 ?
                static_cast<uint64_t>(tilingData->kvHeads) * headDim : headDim;
            const uint64_t qInputStride = tilingData->inputLayout == 0 ?
                static_cast<uint64_t>(tilingData->numHeads) * headDim : headDim;
            const uint64_t qOutputStride = qInputStride;
            const uint64_t kvOutputStride = kvInputStride;
            const uint64_t zeroOffset = segment.tiles[0].sOffset +
                static_cast<uint64_t>(GROUPED_TILES - 1) * GROUPED_TILE_ELEMENTS;

            // Q and dOut are invariant for all gathered KV tiles of the same
            // Q task.  Populate their persistent L1 slots before dQ so dK
            // can consume the same dS L1 tile immediately after dQ.
            SyncExtendedL1Reuse();
            bool cacheUpdated = false;
            for (uint32_t qLocal = 0; qLocal < GROUPED_Q_BLOCKS; ++qLocal) {
                int32_t first = -1;
                for (uint32_t t = 0; t < segment.tileCount; ++t) {
                    if (segment.tiles[t].groupQIdx == qLocal) {
                        first = static_cast<int32_t>(t);
                        break;
                    }
                }
                if (first < 0) {
                    continue;
                }
                TaskInfo &tile = segment.tiles[first];
                LayoutB3 layoutB(tile.curCalQSize, headDim, qInputStride);
                GemmCoord shape{tile.curCalKVSize, headDim, tile.curCalQSize};
                if (((cachedQMask >> qLocal) & 1U) == 0 ||
                    cachedQOffsets[qLocal] != tile.qOffset) {
                    const uint64_t qOffset = GroupedQHeadOffset(tile, tilingData);
                    blockMmad3.PreloadFullB(gDout[qOffset], layoutB, shape, qLocal);
                    blockMmad3.PreloadFullB(gQ[qOffset], layoutB, shape,
                        qLocal + GROUPED_Q_BLOCKS);
                    cachedQOffsets[qLocal] = tile.qOffset;
                    cachedQMask |= 1U << qLocal;
                    cacheUpdated = true;
                }
            }
            if (cacheUpdated) {
                blockMmad3.FinishFullPreload();
            }

            // dQ: each gathered K tile is one full reduction operand.  Keep
            // the at-most-two tiles for one Q in L1 and accumulate in L0C.
            // After each dQ MMAD, reuse its zN dS tile for dK by transposing
            // only the L1->L0A transfer into the other L0C bank.
            for (uint32_t qLocal = 0; qLocal < GROUPED_Q_BLOCKS; ++qLocal) {
                uint32_t tileIndices[GROUPED_PACKED_FRAGMENTS_PER_Q];
                uint32_t matches = 0;
                bool firstKvSegment = false;
                for (uint32_t t = 0; t < segment.tileCount; ++t) {
                    if (segment.tiles[t].groupQIdx == qLocal) {
                        tileIndices[matches++] = t;
                        firstKvSegment |= segment.tiles[t].firstQSegment;
                    }
                }
                if (matches == 0) {
                    continue;
                }
                AscendC::WaitEvent(VEC2CUBE);
                for (uint32_t i = 0; i < matches; ++i) {
                    TaskInfo &tile = segment.tiles[tileIndices[i]];
                    LayoutB2 firstLayout(tile.kvFirstSize, headDim, kvInputStride);
                    LayoutB2 secondLayout(tile.kvSecondSize, headDim, kvInputStride);
                    LayoutB2 paddingLayout(tile.curCalKVSize - tile.kvValidSize,
                        headDim, headDim);
                    GemmCoord shape{tile.curCalQSize, headDim, tile.curCalKVSize};
                    blockMmad2.PreloadGatheredBAlongK(
                        gK[tile.kvOffset], firstLayout,
                        gK[tile.kvSecondOffset], secondLayout,
                        gDs[zeroOffset], paddingLayout,
                        tile.kvFirstSize, tile.kvSecondSize, shape, i);
                }
                const uint32_t accumulatorSlot = qLocal & 1U;
                for (uint32_t i = 0; i < matches; ++i) {
                    TaskInfo &tile = segment.tiles[tileIndices[i]];
                    LayoutA2 layoutA(tile.curCalQSize, tile.curCalKVSize);
                    LayoutB2 layoutB(tile.curCalKVSize, headDim);
                    GemmCoord shape{tile.curCalQSize, headDim, tile.curCalKVSize};
                    blockMmad2.PreloadAccumA(gDs[tile.sOffset], layoutA, shape, i & 1U);
                    blockMmad2.AccumulatePreloadedA(i & 1U, i,
                        layoutA, layoutB, shape, i == 0, i + 1 == matches,
                        accumulatorSlot);

                    const bool contiguousOutput = tile.kvPartCount == 2 &&
                        tile.kvSecondOffset == tile.kvOffset +
                            static_cast<uint64_t>(tile.kvFirstSize) * kvOutputStride;
                    const uint32_t dkAccumulatorSlot = 1U - accumulatorSlot;
                    LayoutB3 dkLayoutB(tile.curCalQSize, headDim);
                    LayoutC3 dkLayoutC(tile.kvValidSize, headDim,
                        kvOutputStride);
                    GemmCoord dkShape{
                        tile.kvValidSize, headDim, tile.curCalQSize};
                    blockMmad3.AccumulateTransposeFromExternalZNL1(
                        blockMmad2.GetAccumL1A(i & 1U),
                        (i & 1U) + PINGPONG_OFFSET_2,
                        tile.groupQIdx + GROUPED_Q_BLOCKS,
                        dkLayoutB, dkShape, dkAccumulatorSlot);
                    if (tile.kvPartCount == 1 || contiguousOutput) {
                        blockMmad3.FlushAccumulator(gDk[tile.kvOffset],
                            dkLayoutC, dkShape, true, dkAccumulatorSlot);
                    } else {
                        LayoutC3 firstLayout(tile.kvFirstSize, headDim,
                            kvOutputStride);
                        LayoutC3 secondLayout(tile.kvSecondSize, headDim,
                            kvOutputStride);
                        blockMmad3.FlushAccumulatorSplitRows(
                            gDk[tile.kvOffset], firstLayout,
                            gDk[tile.kvSecondOffset], secondLayout,
                            dkShape, tile.kvFirstSize, true,
                            dkAccumulatorSlot);
                    }
                }
                TaskInfo &last = segment.tiles[tileIndices[matches - 1]];
                LayoutC2 layoutC(last.curCalQSize, headDim, qOutputStride);
                GemmCoord outShape{last.curCalQSize, headDim, last.curCalKVSize};
                blockMmad2.FlushAccumulator(gDq[last.qOffset], layoutC, outShape,
                    !firstKvSegment, accumulatorSlot);

            }
            if (finalSegment) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            }

            // dK was fused with dQ above.  The remaining P^T*dOut pass
            // computes dV from its persistent dOut cache.
            for (uint32_t pass = 0; pass < 1; ++pass) {
                for (uint32_t t = 0; t < segment.tileCount; ++t) {
                    TaskInfo &tile = segment.tiles[t];
                    // All gathered fragments are contiguous in the logical
                    // compute tile even when their GM destinations differ.
                    // Accumulate them with one MMAD, then either issue one
                    // contiguous store or scatter two row ranges directly
                    // from L0C.  This is capacity driven and does not depend
                    // on a particular user block_shape value.
                    const bool contiguousOutput = tile.kvPartCount == 2 &&
                        tile.kvSecondOffset == tile.kvOffset +
                            static_cast<uint64_t>(tile.kvFirstSize) * kvOutputStride;
                    const uint32_t slot = t & 1U;
                    LayoutA3 layoutA(tile.kvValidSize, tile.curCalQSize,
                        tile.curCalKVSize);
                    LayoutB3 layoutB(tile.curCalQSize, headDim);
                    LayoutC3 layoutC(tile.kvValidSize, headDim,
                        kvOutputStride);
                    GemmCoord shape{
                        tile.kvValidSize, headDim, tile.curCalQSize};
                    AscendC::GlobalTensor<ElementInput> source = gP[tile.sOffset];
                    blockMmad3.PreloadAccumA(source, layoutA, shape, slot);
                    blockMmad3.AccumulatePreloadedA(slot,
                        tile.groupQIdx,
                        layoutA, layoutB, shape, true, true, slot);
                    AscendC::GlobalTensor<float> firstOutput =
                        gDv[tile.kvOffset];
                    if (tile.kvPartCount == 1 || contiguousOutput) {
                        blockMmad3.FlushAccumulator(
                            firstOutput, layoutC, shape, true, slot);
                    } else {
                        AscendC::GlobalTensor<float> secondOutput =
                            gDv[tile.kvSecondOffset];
                        LayoutC3 firstLayout(tile.kvFirstSize, headDim,
                            kvOutputStride);
                        LayoutC3 secondLayout(tile.kvSecondSize, headDim,
                            kvOutputStride);
                        blockMmad3.FlushAccumulatorSplitRows(
                            firstOutput, firstLayout, secondOutput, secondLayout,
                            shape, tile.kvFirstSize, true, slot);
                    }
                }
                if (finalSegment) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
                        pass == 0 ? EVENT_ID6 : EVENT_ID7);
                }
            }
            if (finalSegment) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
            }
        }

        __aicore__ inline void RunGroupedGrad(
            SegmentInfo &segment,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData,
            AscendC::GlobalTensor<ElementInput> gK,
            AscendC::GlobalTensor<ElementInput> gQ,
            AscendC::GlobalTensor<ElementA1> gDout,
            AscendC::GlobalTensor<ElementInput> gP,
            AscendC::GlobalTensor<ElementInput> gDs,
            AscendC::GlobalTensor<float> gDq,
            AscendC::GlobalTensor<float> gDk,
            AscendC::GlobalTensor<float> gDv,
            BlockMmadBSAG2 &blockMmad2, BlockMmadBSAG3 &blockMmad3,
            uint32_t &mmadFlag, bool exclusiveKvOwner, uint64_t &writtenKvBlocks,
            uint64_t *cachedQOffsets, uint32_t &cachedQMask, bool finalSegment)
        {
            const uint32_t headDim = tilingData->headDim;
            const uint64_t qInputStride = UsePackedTndInputs(tilingData) ? headDim :
                (tilingData->inputLayout == 0 ?
                    static_cast<uint64_t>(tilingData->numHeads) * headDim : headDim);
            const uint64_t kvInputStride = UsePackedTndInputs(tilingData) ? headDim :
                (tilingData->inputLayout == 0 ?
                    static_cast<uint64_t>(tilingData->kvHeads) * headDim : headDim);
            const uint64_t qOutputStride = tilingData->inputLayout == 0 ?
                static_cast<uint64_t>(tilingData->numHeads) * headDim : headDim;
            const uint64_t kvOutputStride = tilingData->inputLayout == 0 ?
                static_cast<uint64_t>(tilingData->kvHeads) * headDim : headDim;
            // The legacy sub-128 path keeps at most four K tiles in extended
            // B slots. A normal 128-basic-tile packet can contain up to 16
            // unique KVs, so wider packets refill the same four slots in
            // batches and retain two Q accumulators in the two L0C banks.
            SyncExtendedL1Reuse();
            // Preserve the four-slot CachedB path when the packet fits it
            // (notably a dense 4Q x 4KV packet).  Only wider sparse packets
            // need the streaming path.
            const bool streamDqK = tilingData->basicKVBlockSize == 128 &&
                segment.uniqueKvCount > GROUPED_KV_BLOCKS;
            if (!streamDqK) {
                for (uint32_t kvLocal = 0; kvLocal < segment.uniqueKvCount; ++kvLocal) {
                    int32_t first = -1;
                    for (uint32_t t = 0; t < segment.tileCount; ++t) {
                        if (segment.tiles[t].groupKvIdx == kvLocal) {
                            first = static_cast<int32_t>(t);
                            break;
                        }
                    }
                    if (first < 0) {
                        continue;
                    }
                    TaskInfo &tile = segment.tiles[first];
                    LayoutB2 layoutB(tile.curCalKVSize, headDim, kvInputStride);
                    GemmCoord shape{tile.curCalQSize, headDim, tile.curCalKVSize};
                    blockMmad2.PreloadB(gK[GroupedKvHeadOffset(tile, tilingData)],
                        layoutB, shape, kvLocal);
                }
            }

            if (streamDqK) {
                // The first K batch reuses the phase-boundary fence above.
                // Later batches establish a real MTE1->MTE2 dependency before
                // overwriting the four compact B-cache slots. Pairing Q rows
                // keeps both L0C banks live and lets one K load serve either
                // row when both sparse edges use the same KV column.
                bool firstStreamKBatch = true;
                for (uint32_t qStart = 0; qStart < GROUPED_Q_BLOCKS; qStart += 2) {
                    int32_t tileByKv[2][GROUPED_PACKET_KV_BLOCKS];
                    uint32_t total[2] = {};
                    uint32_t issued[2] = {};
                    uint32_t lastTileIndex[2] = {};
                    bool firstKvSegment[2] = {};
                    bool eventConsumed[2] = {};
                    for (uint32_t qPair = 0; qPair < 2; ++qPair) {
                        for (uint32_t kvLocal = 0;
                            kvLocal < GROUPED_PACKET_KV_BLOCKS; ++kvLocal) {
                            tileByKv[qPair][kvLocal] = -1;
                        }
                    }
                    for (uint32_t t = 0; t < segment.tileCount; ++t) {
                        TaskInfo &tile = segment.tiles[t];
                        if (tile.groupQIdx < qStart ||
                            tile.groupQIdx >= qStart + 2) {
                            continue;
                        }
                        const uint32_t qPair = tile.groupQIdx - qStart;
                        tileByKv[qPair][tile.groupKvIdx] =
                            static_cast<int32_t>(t);
                        ++total[qPair];
                        lastTileIndex[qPair] = t;
                        firstKvSegment[qPair] |= tile.firstQSegment;
                    }
                    if (total[0] == 0 && total[1] == 0) {
                        continue;
                    }

                    uint32_t kvCursor = 0;
                    while (kvCursor < segment.uniqueKvCount) {
                        uint32_t batchKv[GROUPED_KV_BLOCKS];
                        uint32_t batchFirstTile[GROUPED_KV_BLOCKS];
                        uint32_t batchCount = 0;
                        while (kvCursor < segment.uniqueKvCount &&
                            batchCount < GROUPED_KV_BLOCKS) {
                            const uint32_t kvLocal = kvCursor++;
                            const int32_t first = tileByKv[0][kvLocal] >= 0 ?
                                tileByKv[0][kvLocal] : tileByKv[1][kvLocal];
                            if (first < 0) {
                                continue;
                            }
                            batchKv[batchCount] = kvLocal;
                            batchFirstTile[batchCount] =
                                static_cast<uint32_t>(first);
                            ++batchCount;
                        }
                        if (batchCount == 0) {
                            continue;
                        }
                        if (!firstStreamKBatch) {
                            SyncExtendedL1Reuse();
                        }
                        firstStreamKBatch = false;
                        for (uint32_t cacheSlot = 0;
                            cacheSlot < batchCount; ++cacheSlot) {
                            TaskInfo &tile =
                                segment.tiles[batchFirstTile[cacheSlot]];
                            LayoutB2 layoutB(
                                tile.curCalKVSize, headDim, kvInputStride);
                            GemmCoord shape{
                                tile.curCalQSize, headDim, tile.curCalKVSize};
                            blockMmad2.PreloadFullB(
                                gK[GroupedKvHeadOffset(tile, tilingData)],
                                layoutB, shape, cacheSlot);
                        }
                        // One MTE2->MTE1 fence covers all independent K
                        // copies in this batch.
                        blockMmad2.FinishFullPreload();

                        for (uint32_t qPair = 0; qPair < 2; ++qPair) {
                            uint32_t batchTiles[GROUPED_KV_BLOCKS];
                            uint32_t batchCacheSlots[GROUPED_KV_BLOCKS];
                            uint32_t matches = 0;
                            for (uint32_t cacheSlot = 0;
                                cacheSlot < batchCount; ++cacheSlot) {
                                const int32_t tileIndex =
                                    tileByKv[qPair][batchKv[cacheSlot]];
                                if (tileIndex < 0) {
                                    continue;
                                }
                                batchTiles[matches] =
                                    static_cast<uint32_t>(tileIndex);
                                batchCacheSlots[matches] = cacheSlot;
                                ++matches;
                            }
                            if (matches == 0) {
                                continue;
                            }
                            // K is independent of the vector result and was
                            // prefetched above. Cross-core completions are an
                            // ordered token stream rather than tagged by Q,
                            // so consume every preceding live Q token before
                            // reading this Q's first dS tile.
                            if (!eventConsumed[qPair]) {
                                for (uint32_t readyPair = 0;
                                    readyPair <= qPair; ++readyPair) {
                                    if (total[readyPair] != 0 &&
                                        !eventConsumed[readyPair]) {
                                        AscendC::WaitEvent(VEC2CUBE);
                                        eventConsumed[readyPair] = true;
                                    }
                                }
                            }
                            const uint32_t initialPreloads =
                                matches < 2 ? matches : 2;
                            for (uint32_t i = 0;
                                i < initialPreloads; ++i) {
                                TaskInfo &tile = segment.tiles[batchTiles[i]];
                                LayoutA2 layoutA(
                                    tile.curCalQSize, tile.curCalKVSize);
                                GemmCoord shape{
                                    tile.curCalQSize, headDim,
                                    tile.curCalKVSize};
                                blockMmad2.PreloadAccumA(
                                    gDs[tile.sOffset], layoutA, shape, i);
                            }
                            for (uint32_t i = 0; i < matches; ++i) {
                                const uint32_t slot = i & 1U;
                                TaskInfo &tile = segment.tiles[batchTiles[i]];
                                LayoutA2 layoutA(
                                    tile.curCalQSize, tile.curCalKVSize);
                                LayoutB2 layoutB(
                                    tile.curCalKVSize, headDim);
                                GemmCoord shape{
                                    tile.curCalQSize, headDim,
                                    tile.curCalKVSize};
                                const bool initC = issued[qPair] == 0;
                                const bool finalC =
                                    issued[qPair] + 1 == total[qPair];
                                blockMmad2.AccumulatePreloadedA(
                                    slot, batchCacheSlots[i], layoutA,
                                    layoutB, shape, initC, finalC, qPair);
                                ++issued[qPair];
                                if (i + 2 < matches) {
                                    TaskInfo &nextTile =
                                        segment.tiles[batchTiles[i + 2]];
                                    LayoutA2 nextLayoutA(
                                        nextTile.curCalQSize,
                                        nextTile.curCalKVSize);
                                    GemmCoord nextShape{
                                        nextTile.curCalQSize, headDim,
                                        nextTile.curCalKVSize};
                                    blockMmad2.PreloadAccumA(
                                        gDs[nextTile.sOffset], nextLayoutA,
                                        nextShape, slot);
                                }
                            }
                        }
                    }

                    for (uint32_t qPair = 0; qPair < 2; ++qPair) {
                        if (total[qPair] == 0) {
                            continue;
                        }
                        TaskInfo &tile = segment.tiles[lastTileIndex[qPair]];
                        LayoutC2 layoutC(
                            tile.curCalQSize, headDim, qOutputStride);
                        GemmCoord outShape{
                            tile.curCalQSize, headDim, tile.curCalKVSize};
                        blockMmad2.FlushAccumulator(
                            gDq[tile.qOffset], layoutC, outShape,
                            !firstKvSegment[qPair], qPair);
                    }
                }
                // Protect the compact region before blockMmad1 reuses it in
                // the next Cube1 segment. High-L1 dK/dV work can overlap the
                // resulting dependency.
                if (!firstStreamKBatch) {
                    SyncExtendedL1Reuse();
                }
            } else {
                for (uint32_t qLocal = 0;
                    qLocal < GROUPED_Q_BLOCKS; ++qLocal) {
                    uint32_t tileIndices[GROUPED_TILES];
                    uint32_t matches = 0;
                    bool firstKvSegment = false;
                    for (uint32_t t = 0; t < segment.tileCount; ++t) {
                        if (segment.tiles[t].groupQIdx == qLocal) {
                            tileIndices[matches++] = t;
                            firstKvSegment |=
                                segment.tiles[t].firstQSegment;
                        }
                    }
                    if (matches == 0) {
                        continue;
                    }
                    AscendC::WaitEvent(VEC2CUBE);
                    const uint32_t accumulatorSlot = qLocal & 1U;
                    const uint32_t initialPreloads =
                        matches < 2 ? matches : 2;
                    for (uint32_t i = 0; i < initialPreloads; ++i) {
                        TaskInfo &tile = segment.tiles[tileIndices[i]];
                        LayoutA2 layoutA(tile.curCalQSize, tile.curCalKVSize);
                        GemmCoord shape{
                            tile.curCalQSize, headDim, tile.curCalKVSize};
                        blockMmad2.PreloadAccumA(
                            gDs[tile.sOffset], layoutA, shape, i);
                    }
                    for (uint32_t i = 0; i < matches; ++i) {
                        const uint32_t slot = i & 1U;
                        TaskInfo &tile = segment.tiles[tileIndices[i]];
                        LayoutA2 layoutA(tile.curCalQSize, tile.curCalKVSize);
                        LayoutB2 layoutB(tile.curCalKVSize, headDim);
                        GemmCoord shape{
                            tile.curCalQSize, headDim, tile.curCalKVSize};
                        blockMmad2.AccumulatePreloadedA(
                            slot, tile.groupKvIdx, layoutA, layoutB, shape,
                            i == 0, i + 1 == matches, accumulatorSlot);
                        if (i + 2 < matches) {
                            TaskInfo &nextTile = segment.tiles[tileIndices[i + 2]];
                            LayoutA2 nextLayoutA(
                                nextTile.curCalQSize, nextTile.curCalKVSize);
                            GemmCoord nextShape{
                                nextTile.curCalQSize, headDim,
                                nextTile.curCalKVSize};
                            blockMmad2.PreloadAccumA(
                                gDs[nextTile.sOffset], nextLayoutA,
                                nextShape, slot);
                        }
                    }
                    TaskInfo &tile = segment.tiles[tileIndices[matches - 1]];
                    LayoutC2 layoutC(
                        tile.curCalQSize, headDim, qOutputStride);
                    GemmCoord outShape{
                        tile.curCalQSize, headDim, tile.curCalKVSize};
                    blockMmad2.FlushAccumulator(
                        gDq[tile.qOffset], layoutC, outShape,
                        !firstKvSegment, accumulatorSlot);
                }
            }
            if (finalSegment) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            }

            // dOut and Q are shared by all dV/dK work in this gradient
            // segment. Their eight L1 slots do not overlap dQ, so reload a
            // pair only when Cube1 invalidated it or the Q offset changed.
            bool cacheUpdated = false;
            for (uint32_t qLocal = 0; qLocal < GROUPED_Q_BLOCKS; ++qLocal) {
                int32_t first = -1;
                for (uint32_t t = 0; t < segment.tileCount; ++t) {
                    if (segment.tiles[t].groupQIdx == qLocal) {
                        first = static_cast<int32_t>(t);
                        break;
                    }
                }
                if (first < 0) {
                    continue;
                }
                TaskInfo &tile = segment.tiles[first];
                LayoutB3 layoutB(tile.curCalQSize, headDim, qInputStride);
                GemmCoord shape{tile.curCalKVSize, headDim, tile.curCalQSize};
                if (((cachedQMask >> qLocal) & 1U) == 0 ||
                    cachedQOffsets[qLocal] != tile.qOffset) {
                    const uint64_t packedQOffset = GroupedQHeadOffset(tile, tilingData);
                    blockMmad3.PreloadFullB(gDout[packedQOffset], layoutB, shape, qLocal);
                    blockMmad3.PreloadFullB(gQ[packedQOffset], layoutB, shape,
                        qLocal + GROUPED_Q_BLOCKS);
                    cachedQOffsets[qLocal] = tile.qOffset;
                    cachedQMask |= 1U << qLocal;
                    cacheUpdated = true;
                }
            }
            if (cacheUpdated) {
                blockMmad3.FinishFullPreload();
            }
            bool firstKvOutput[GROUPED_PACKET_KV_BLOCKS] = {};
            for (uint32_t kvLocal = 0; kvLocal < segment.uniqueKvCount; ++kvLocal) {
                uint32_t tileIndices[GROUPED_Q_BLOCKS];
                uint32_t matches = 0;
                uint32_t kvBasicIdx = 0;
                for (uint32_t t = 0; t < segment.tileCount; ++t) {
                    if (segment.tiles[t].groupKvIdx == kvLocal) {
                        tileIndices[matches++] = t;
                        kvBasicIdx = segment.tiles[t].kvBasicIdx;
                    }
                }
                if (matches == 0) {
                    continue;
                }
                const uint32_t accumulatorSlot = kvLocal & 1U;
                firstKvOutput[kvLocal] = exclusiveKvOwner && kvBasicIdx < 64 &&
                    ((writtenKvBlocks >> kvBasicIdx) & 1ULL) == 0;
                const uint32_t initialPreloads = matches < 2 ? matches : 2;
                for (uint32_t i = 0; i < initialPreloads; ++i) {
                    TaskInfo &tile = segment.tiles[tileIndices[i]];
                    LayoutA3 layoutA(tile.curCalKVSize, tile.curCalQSize);
                    GemmCoord shape{tile.curCalKVSize, headDim, tile.curCalQSize};
                    blockMmad3.PreloadAccumA(
                        gP[tile.sOffset], layoutA, shape, i);
                }
                for (uint32_t i = 0; i < matches; ++i) {
                    const uint32_t slot = i & 1U;
                    TaskInfo &tile = segment.tiles[tileIndices[i]];
                    LayoutA3 layoutA(tile.curCalKVSize, tile.curCalQSize);
                    LayoutB3 layoutB(tile.curCalQSize, headDim);
                    GemmCoord shape{tile.curCalKVSize, headDim, tile.curCalQSize};
                    blockMmad3.AccumulatePreloadedA(slot, tile.groupQIdx,
                        layoutA, layoutB, shape, i == 0, i + 1 == matches,
                        accumulatorSlot);
                    if (i + 2 < matches) {
                        TaskInfo &nextTile = segment.tiles[tileIndices[i + 2]];
                        LayoutA3 nextLayoutA(
                            nextTile.curCalKVSize, nextTile.curCalQSize);
                        GemmCoord nextShape{
                            nextTile.curCalKVSize, headDim, nextTile.curCalQSize};
                        blockMmad3.PreloadAccumA(gP[nextTile.sOffset],
                            nextLayoutA, nextShape, slot);
                    }
                }
                TaskInfo &lastTile = segment.tiles[tileIndices[matches - 1]];
                LayoutC3 layoutC(lastTile.curCalKVSize, headDim, kvOutputStride);
                GemmCoord outShape{lastTile.curCalKVSize, headDim, lastTile.curCalQSize};
                blockMmad3.FlushAccumulator(gDv[lastTile.kvOffset], layoutC, outShape,
                    !firstKvOutput[kvLocal], accumulatorSlot);
            }

            if (finalSegment) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
            }
            for (uint32_t kvLocal = 0; kvLocal < segment.uniqueKvCount; ++kvLocal) {
                uint32_t tileIndices[GROUPED_Q_BLOCKS];
                uint32_t matches = 0;
                for (uint32_t t = 0; t < segment.tileCount; ++t) {
                    if (segment.tiles[t].groupKvIdx == kvLocal) {
                        tileIndices[matches++] = t;
                    }
                }
                if (matches == 0) {
                    continue;
                }
                const uint32_t accumulatorSlot = kvLocal & 1U;

                const uint32_t initialPreloads = matches < 2 ? matches : 2;
                for (uint32_t i = 0; i < initialPreloads; ++i) {
                    TaskInfo &tile = segment.tiles[tileIndices[i]];
                    LayoutA3 layoutA(tile.curCalKVSize, tile.curCalQSize);
                    GemmCoord shape{tile.curCalKVSize, headDim, tile.curCalQSize};
                    blockMmad3.PreloadAccumA(
                        gDs[tile.sOffset], layoutA, shape, i);
                }
                for (uint32_t i = 0; i < matches; ++i) {
                    const uint32_t slot = i & 1U;
                    TaskInfo &tile = segment.tiles[tileIndices[i]];
                    LayoutA3 layoutA(tile.curCalKVSize, tile.curCalQSize);
                    LayoutB3 layoutB(tile.curCalQSize, headDim);
                    GemmCoord shape{tile.curCalKVSize, headDim, tile.curCalQSize};
                    blockMmad3.AccumulatePreloadedA(slot,
                        tile.groupQIdx + GROUPED_Q_BLOCKS,
                        layoutA, layoutB, shape, i == 0, i + 1 == matches,
                        accumulatorSlot);
                    if (i + 2 < matches) {
                        TaskInfo &nextTile = segment.tiles[tileIndices[i + 2]];
                        LayoutA3 nextLayoutA(nextTile.curCalKVSize, nextTile.curCalQSize);
                        GemmCoord nextShape{
                            nextTile.curCalKVSize, headDim, nextTile.curCalQSize};
                        blockMmad3.PreloadAccumA(gDs[nextTile.sOffset],
                            nextLayoutA, nextShape, slot);
                    }
                }
                TaskInfo &lastTile = segment.tiles[tileIndices[matches - 1]];
                LayoutC3 layoutC(lastTile.curCalKVSize, headDim, kvOutputStride);
                GemmCoord outShape{lastTile.curCalKVSize, headDim, lastTile.curCalQSize};
                blockMmad3.FlushAccumulator(gDk[lastTile.kvOffset], layoutC, outShape,
                    !firstKvOutput[kvLocal], accumulatorSlot);
                if (firstKvOutput[kvLocal]) {
                    writtenKvBlocks |= 1ULL << lastTile.kvBasicIdx;
                }
            }
        }

        __aicore__ inline void RunGroupedGradDispatch(
            SegmentInfo &segment,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData,
            AscendC::GlobalTensor<ElementInput> gK,
            AscendC::GlobalTensor<ElementInput> gQ,
            AscendC::GlobalTensor<ElementA1> gDout,
            AscendC::GlobalTensor<ElementInput> gP,
            AscendC::GlobalTensor<ElementInput> gDs,
            AscendC::GlobalTensor<float> gDq,
            AscendC::GlobalTensor<float> gDk,
            AscendC::GlobalTensor<float> gDv,
            BlockMmadBSAG2 &blockMmad2, BlockMmadBSAG3 &blockMmad3,
            uint32_t &mmadFlag, bool exclusiveKvOwner,
            uint64_t &writtenKvBlocks, uint64_t *cachedQOffsets,
            uint32_t &cachedQMask, bool finalSegment)
        {
            if (tilingData->basicKVBlockSize < 128 &&
                tilingData->blockShapeX >= 16) {
                RunGroupedPackedGrad(segment, tilingData, gK, gQ, gDout,
                    gP, gDs, gDq, gDk, gDv, blockMmad2, blockMmad3,
                    mmadFlag, cachedQOffsets, cachedQMask, finalSegment);
            } else {
                RunGroupedGrad(segment, tilingData, gK, gQ, gDout,
                    gP, gDs, gDq, gDk, gDv, blockMmad2, blockMmad3,
                    mmadFlag, exclusiveKvOwner, writtenKvBlocks,
                    cachedQOffsets, cachedQMask, finalSegment);
            }
        }

        __aicore__ inline void WaitWideFlag() const
        {
            // MTE1_MTE2 event 7 is the explicit wide L1-region reuse token.
            // Events 0/1 belong to Cube1, 2/3 to the native-zN dQ/P/dS
            // streams, and 4/5 guard the paired dOut/Q cache.  Event 6 stays
            // seeded so all SetFlag() tokens are drained exactly once.
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        }

        // Reduce every active K column of a packet into one dQ Fixpipe store.
        // qSeenMask spans all packets of the work item: the first non-empty
        // packet is a direct write (the Q task has a unique core owner), and
        // later packets atomically add their disjoint K-range contribution.
        __aicore__ inline void RunWideDq(
            const WidePacketInfo &packet, const TaskInfo &workInfo,
            const WideQInfo *qInfos,
            uint32_t qCount,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData,
            AscendC::GlobalTensor<ElementInput> gK,
            AscendC::GlobalTensor<ElementInput> gDs,
            AscendC::GlobalTensor<float> gDq,
            BlockMmadBSAG2 &blockMmad2, WideQMask &qSeenMask)
        {
            const uint32_t headDim = tilingData->headDim;

            // A wide packet is K-major and contains at most eight distinct KV
            // columns.  Cache each K tile once in the shared high-L1 region;
            // the Q-major reduction below addresses it through kvLocal.
            for (uint32_t kvLocal = 0;
                kvLocal < packet.uniqueKvCount; ++kvLocal) {
                const uint32_t edge = packet.kv[kvLocal].edgeBegin;
                TaskInfo tile = MakeWideTile(
                    workInfo, qInfos, packet, edge, tilingData);
                LayoutB2 layoutB(
                    tile.curCalKVSize, headDim, headDim);
                GemmCoord shape{
                    tile.curCalQSize, headDim, tile.curCalKVSize};
                blockMmad2.PreloadFullB(
                    gK[tile.kvOffset], layoutB, shape, kvLocal);
            }
            blockMmad2.FinishFullPreload();

            for (uint32_t activeQ = 0;
                activeQ < packet.activeQCount; ++activeQ) {
                const WideQDesc &qDesc = packet.activeQ[activeQ];
                const uint32_t qLocal = qDesc.qLocal;
                const uint32_t qBegin = qDesc.edgeBegin;
                const uint32_t edgeCount = qDesc.edgeCount;
                TaskInfo lastTile;
                const uint32_t initial = edgeCount < 2 ? edgeCount : 2;
                for (uint32_t i = 0; i < initial; ++i) {
                    const uint32_t edge = packet.qOrder[qBegin + i];
                    TaskInfo tile = MakeWideTile(
                        workInfo, qInfos, packet, edge, tilingData);
                    GemmCoord shape{
                        tile.curCalQSize, headDim, tile.curCalKVSize};
                    blockMmad2.PreloadAccumANz(
                        gDs[tile.sOffset], shape, i);
                }
                for (uint32_t i = 0; i < edgeCount; ++i) {
                    const uint32_t slot = i & 1U;
                    const uint32_t edge = packet.qOrder[qBegin + i];
                    const uint32_t kvLocal =
                        WideEdgeKv(packet.edgeCode[edge]);
                    TaskInfo tile = MakeWideTile(
                        workInfo, qInfos, packet, edge, tilingData);
                    LayoutA2 layoutA(tile.curCalQSize, tile.curCalKVSize);
                    LayoutB2 layoutB(tile.curCalKVSize, headDim);
                    GemmCoord shape{
                        tile.curCalQSize, headDim, tile.curCalKVSize};
                    blockMmad2.AccumulatePreloadedA(slot, kvLocal, layoutA,
                        layoutB, shape, i == 0, i + 1 == edgeCount,
                        qLocal & 1U);
                    lastTile = tile;
                    if (i + 2 < edgeCount) {
                        const uint32_t nextEdge =
                            packet.qOrder[qBegin + i + 2];
                        TaskInfo nextTile = MakeWideTile(
                            workInfo, qInfos, packet, nextEdge, tilingData);
                        GemmCoord nextShape{
                            nextTile.curCalQSize, headDim,
                            nextTile.curCalKVSize};
                        blockMmad2.PreloadAccumANz(
                            gDs[nextTile.sOffset], nextShape, slot);
                    }
                }
                LayoutC2 layoutC(lastTile.curCalQSize, headDim, headDim);
                GemmCoord outShape{
                    lastTile.curCalQSize, headDim, lastTile.curCalKVSize};
                const WideQMask qBit =
                    static_cast<WideQMask>(1) << qLocal;
                blockMmad2.FlushAccumulator(gDq[lastTile.qOffset], layoutC,
                    outShape, (qSeenMask & qBit) != 0, qLocal & 1U);
                qSeenMask |= qBit;
            }
        }

        // dK/dV may receive contributions from several Q work items and
        // cores.  Traverse each KV column once and use the eight persistent-B
        // slots as four (dO,Q) pairs.  P and dS stream through independent A
        // slots and L0C accumulators, retaining one atomic Fixpipe store per
        // KV column and matrix while halving cache fences and descriptor work.
        __aicore__ inline void RunWideDkvPair(
            const WidePacketInfo &packet, const TaskInfo &workInfo,
            const WideQInfo *qInfos,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData,
            AscendC::GlobalTensor<ElementInput> gQ,
            AscendC::GlobalTensor<ElementA1> gDout,
            AscendC::GlobalTensor<ElementInput> gP,
            AscendC::GlobalTensor<ElementInput> gDs,
            AscendC::GlobalTensor<float> gDk,
            AscendC::GlobalTensor<float> gDv,
            BlockMmadBSAG2 &blockMmad2, BlockMmadBSAG3 &blockMmad3)
        {
            const uint32_t headDim = tilingData->headDim;
            for (uint32_t kvLocal = 0;
                kvLocal < packet.uniqueKvCount; ++kvLocal) {
                const WideKvDesc &kv = packet.kv[kvLocal];
                const uint32_t edgeEnd = kv.edgeBegin + kv.edgeCount;
                uint32_t issued = 0;
                uint32_t cursor = kv.edgeBegin;
                TaskInfo lastTile;
                while (cursor < edgeEnd) {
                    const uint32_t batchCount = edgeEnd - cursor <
                        GROUPED_Q_BLOCKS ? edgeEnd - cursor : GROUPED_Q_BLOCKS;
                    // Events 4/5 were published by the final dV/dK L1 reads
                    // of the preceding batch (or seeded at kernel entry).
                    // Consume them before replacing its eight B-cache slots.
                    if (tilingData->wideQOperandCache == 0) {
                        blockMmad3.AcquireFullPreloadReuse();
                        for (uint32_t i = 0; i < batchCount; ++i) {
                            TaskInfo tile = MakeWideTile(
                                workInfo, qInfos, packet, cursor + i, tilingData);
                            LayoutB3 layoutB(tile.curCalQSize, headDim, headDim);
                            GemmCoord shape{
                                tile.curCalKVSize, headDim, tile.curCalQSize};
                            blockMmad3.PreloadFullB(
                                gDout[tile.qOffset], layoutB, shape, i);
                            blockMmad3.PreloadFullB(gQ[tile.qOffset], layoutB,
                                shape, i + GROUPED_Q_BLOCKS);
                        }
                        blockMmad3.FinishFullPreload();
                    }

                    TaskInfo firstTile = MakeWideTile(
                        workInfo, qInfos, packet, cursor, tilingData);
                    GemmCoord firstSourceShape{
                        firstTile.curCalQSize, headDim,
                        firstTile.curCalKVSize};
                    // P and dS are zN[Q,KV].  Stream them through the compact
                    // low-L1 slots, then transpose each 16x16 fractal only on
                    // L1->L0A for P^T*dO and dS^T*Q.
                    blockMmad2.PreloadAccumANz(
                        gP[firstTile.sOffset], firstSourceShape, 0);
                    blockMmad2.PreloadAccumANz(
                        gDs[firstTile.sOffset], firstSourceShape, 1);
                    for (uint32_t i = 0; i < batchCount; ++i) {
                        TaskInfo tile = MakeWideTile(
                            workInfo, qInfos, packet, cursor + i, tilingData);
                        LayoutB3 layoutB(
                            tile.curCalQSize, headDim, headDim);
                        GemmCoord shape{
                            tile.curCalKVSize, headDim, tile.curCalQSize};
                        const bool initC = issued == 0;
                        const bool finalC = issued + 1 == kv.edgeCount;

                        blockMmad2.FinishAccumPreloadForExternal(0);
                        blockMmad3.AccumulateTransposeFromExternalZNL1(
                            blockMmad2.GetAccumL1A(0),
                            PINGPONG_OFFSET_2, tilingData->wideQOperandCache != 0 ?
                                tile.groupQIdx : i, layoutB, shape, 0,
                            initC, finalC, tilingData->wideQOperandCache == 0 && i + 1 == batchCount);
                        if (i + 1 < batchCount) {
                            TaskInfo nextTile = MakeWideTile(
                                workInfo, qInfos, packet, cursor + i + 1,
                                tilingData);
                            GemmCoord nextSourceShape{
                                nextTile.curCalQSize, headDim,
                                nextTile.curCalKVSize};
                            blockMmad2.PreloadAccumANz(
                                gP[nextTile.sOffset], nextSourceShape, 0);
                        }

                        blockMmad2.FinishAccumPreloadForExternal(1);
                        blockMmad3.AccumulateTransposeFromExternalZNL1(
                            blockMmad2.GetAccumL1A(1),
                            PINGPONG_OFFSET_2 + 1,
                            (tilingData->wideQOperandCache != 0 ? tile.groupQIdx : i) +
                                GROUPED_Q_BLOCKS, layoutB, shape, 1,
                            initC, finalC, tilingData->wideQOperandCache == 0 && i + 1 == batchCount);
                        ++issued;
                        lastTile = tile;
                        if (i + 1 < batchCount) {
                            TaskInfo nextTile = MakeWideTile(
                                workInfo, qInfos, packet, cursor + i + 1,
                                tilingData);
                            GemmCoord nextSourceShape{
                                nextTile.curCalQSize, headDim,
                                nextTile.curCalKVSize};
                            blockMmad2.PreloadAccumANz(
                                gDs[nextTile.sOffset], nextSourceShape, 1);
                        }
                    }
                    cursor += batchCount;
                }
                LayoutC3 layoutC(lastTile.curCalKVSize, headDim, headDim);
                GemmCoord outShape{
                    lastTile.curCalKVSize, headDim, lastTile.curCalQSize};
                blockMmad3.FlushAccumulator(gDv[lastTile.kvOffset],
                    layoutC, outShape, true, 0);
                blockMmad3.FlushAccumulator(gDk[lastTile.kvOffset],
                    layoutC, outShape, true, 1);
            }
        }

        __aicore__ inline void RunWideGrad(
            const WidePacketInfo &packet, const TaskInfo &workInfo,
            const WideQInfo *qInfos,
            uint32_t qCount,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData,
            AscendC::GlobalTensor<ElementInput> gK,
            AscendC::GlobalTensor<ElementInput> gQ,
            AscendC::GlobalTensor<ElementA1> gDout,
            AscendC::GlobalTensor<ElementInput> gP,
            AscendC::GlobalTensor<ElementInput> gDs,
            AscendC::GlobalTensor<float> gDq,
            AscendC::GlobalTensor<float> gDk,
            AscendC::GlobalTensor<float> gDv,
            BlockMmadBSAG2 &blockMmad2, BlockMmadBSAG3 &blockMmad3,
            WideQMask &qSeenMask)
        {
            // Cube1 and dQ alias the compact low-L1 region; the high eight-B
            // cache may still be feeding the preceding packet's dKV pair.
            // Complete dQ first because both BlockMmad objects alias the same
            // physical L0 banks, then reuse low L1 as native-zN P/dS streams
            // while dOut/Q remain resident in the high eight-slot cache.
            SyncExtendedL1Reuse();
            if (tilingData->wideQOperandCache == 0) {
                blockMmad3.AcquireFullPreloadReuse();
            }
            RunWideDq(packet, workInfo, qInfos, qCount, tilingData,
                gK, gDs, gDq, blockMmad2, qSeenMask);
            if (tilingData->wideQOperandCache == 0) {
                blockMmad3.ReleaseFullPreloadReuse();
            }
            RunWideDkvPair(packet, workInfo, qInfos, tilingData,
                gQ, gDout, gP, gDs, gDk, gDv,
                blockMmad2, blockMmad3);
        }

        __aicore__ inline uint32_t BuildWideWorkItem(
            AscendC::GlobalTensor<int64_t> gActualQseqlen,
            AscendC::GlobalTensor<int64_t> gActualKvseqlen,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData,
            uint32_t groupSize, const TaskInfo &current,
            WideQInfo *qInfos, TaskInfo &nextWork, bool needNext)
        {
            const uint32_t runtimeCapacity =
                tilingData->wideQTasksPerWorkItem < WIDE_MAX_Q_TASKS ?
                tilingData->wideQTasksPerWorkItem : WIDE_MAX_Q_TASKS;
            TaskInfo cursor = current;
            uint32_t qCount = 0;
            while (qCount < runtimeCapacity) {
                WideQInfo &qInfo = qInfos[qCount++];
                qInfo.qOffset = cursor.qOffset;
                qInfo.curQSeqIdx = cursor.curQSeqIdx;
                qInfo.curQBlockIdx = cursor.curQBlcokIdx;
                qInfo.curCalQSize = cursor.curCalQSize;
                if (qCount == runtimeCapacity ||
                    cursor.curQSeqIdx + cursor.curCalQSize == cursor.qSeqlen) {
                    break;
                }
                TaskInfo candidate;
                updateNextGroupedTaskInfo(gActualQseqlen, gActualKvseqlen,
                    tilingData->numHeads, tilingData->kvHeads, groupSize,
                    tilingData->headDim, tilingData->blockShapeX,
                    tilingData->basicQBlockSize, tilingData->inputLayout,
                    cursor, candidate);
                cursor = candidate;
            }
            if (needNext) {
                if (tilingData->wideTaskStride != 0) {
                    // BNSD batches have fixed physical sequence lengths.
                    // Map a cyclic work-item index back to the original
                    // user-block fragments without crossing their boundaries.
                    const uint32_t blockX = tilingData->blockShapeX;
                    const uint32_t fragments = (blockX + 127) / 128;
                    const uint32_t fullBlocks = current.qSeqlen / blockX;
                    const uint32_t qTasks = fullBlocks * fragments +
                        (current.qSeqlen - fullBlocks * blockX + 127) / 128;
                    const uint32_t workItems = (qTasks + runtimeCapacity - 1) /
                        runtimeCapacity;
                    const uint32_t qTask = (current.curQSeqIdx / blockX) * fragments +
                        (current.curQSeqIdx % blockX) / 128;
                    const uint32_t workIndex =
                        (current.curBatchIdx * tilingData->numHeads + current.curHeadIdx) *
                            workItems + qTask / runtimeCapacity + tilingData->wideTaskStride;
                    const uint32_t headIndex = workIndex / workItems;
                    const uint32_t nextQTask = (workIndex % workItems) * runtimeCapacity;
                    nextWork = current;
                    nextWork.curBatchIdx = headIndex / tilingData->numHeads;
                    nextWork.curHeadIdx = headIndex % tilingData->numHeads;
                    nextWork.curQSeqIdx = (nextQTask / fragments) * blockX +
                        (nextQTask % fragments) * 128;
                    nextWork.qOffset = (static_cast<uint64_t>(headIndex) *
                        tilingData->maxQSeqlen + nextWork.curQSeqIdx) * tilingData->headDim;
                    nextWork.kvOffset = (static_cast<uint64_t>(nextWork.curBatchIdx) *
                        tilingData->kvHeads + nextWork.curHeadIdx / groupSize) *
                        tilingData->maxKvSeqlen * tilingData->headDim;
                    UpdateTaskInfoCalQSize(blockX, tilingData->basicQBlockSize, nextWork);
                } else {
                    updateNextGroupedTaskInfo(gActualQseqlen, gActualKvseqlen,
                        tilingData->numHeads, tilingData->kvHeads, groupSize,
                        tilingData->headDim, tilingData->blockShapeX,
                        tilingData->basicQBlockSize, tilingData->inputLayout,
                        cursor, nextWork);
                }
            }
            return qCount;
        }

        __aicore__ inline void ProcessWideAic(
            const Params &params,
            __gm__ BlockSparseAttentionGradTilingData *tilingData)
        {
            const uint32_t coreIdx = AscendC::GetBlockIdx();
            const uint32_t groupSize = tilingData->numHeads / tilingData->kvHeads;
            const uint32_t workLength = tilingData->taskNumPerCore +
                (tilingData->tailTaskNum > coreIdx ? 1 : 0);

            AscendC::GlobalTensor<ElementInput> gQ;
            AscendC::GlobalTensor<ElementInput> gK;
            AscendC::GlobalTensor<ElementInput> gV;
            AscendC::GlobalTensor<ElementA1> gDout;
            AscendC::GlobalTensor<int64_t> gActualQseqlen;
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;
            AscendC::GlobalTensor<float> gS;
            AscendC::GlobalTensor<float> gDp;
            AscendC::GlobalTensor<ElementInput> gP;
            AscendC::GlobalTensor<ElementInput> gDs;
            AscendC::GlobalTensor<float> gDq;
            AscendC::GlobalTensor<float> gDk;
            AscendC::GlobalTensor<float> gDv;
            gQ.SetGlobalBuffer((__gm__ ElementInput *)params.q);
            gK.SetGlobalBuffer((__gm__ ElementInput *)params.k);
            gV.SetGlobalBuffer((__gm__ ElementInput *)params.v);
            gDout.SetGlobalBuffer((__gm__ ElementA1 *)params.dout);
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
            gS.SetGlobalBuffer((__gm__ float *)params.workspace);
            gP.SetGlobalBuffer((__gm__ ElementInput *)(
                params.workspace + tilingData->wideRawPlaneBytes));
            gDp.SetGlobalBuffer((__gm__ float *)(
                params.workspace + tilingData->sOutSize));
            gDs.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace +
                tilingData->sOutSize + tilingData->wideRawPlaneBytes));
            gDq.SetGlobalBuffer((__gm__ float *)(params.workspace +
                tilingData->sOutSize + tilingData->dPOutSize));
            gDk.SetGlobalBuffer((__gm__ float *)(params.workspace +
                tilingData->sOutSize + tilingData->dPOutSize +
                tilingData->dQOutSize));
            gDv.SetGlobalBuffer((__gm__ float *)(params.workspace +
                tilingData->sOutSize + tilingData->dPOutSize +
                tilingData->dQOutSize + tilingData->dKOutSize));

            TaskInfo current;
            initTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData,
                tilingData->numHeads, tilingData->kvHeads, groupSize,
                tilingData->headDim, tilingData->maxQSeqlen,
                tilingData->maxKvSeqlen, tilingData->blockShapeX,
                tilingData->basicQBlockSize, tilingData->inputLayout,
                coreIdx, current);

            // The resident path partitions L1 into P/dS streams [0,64) KiB,
            // eight K slots [64,320) KiB, and Q/dO pairs [320,512) KiB.
            // Q/dO survive Cube1 and both gradient phases of the whole work
            // item. The legacy path retains its four-Q refill batches.
            BlockMmadBSAG1 blockMmad1(resource);
            BlockMmadBSAG2 blockMmad2(resource, 0, PINGPONG_OFFSET_2,
                true, false, true, false, tilingData->wideQOperandCache != 0 ? 1 : 0);
            BlockMmadBSAG3 blockMmad3(resource, L1_SIZE_OFFSET * 7 / 2,
                PINGPONG_OFFSET_4, true, false, true, false,
                tilingData->wideQOperandCache != 0 ? 2 : 0);
            blockMmad1.SetFastCachedInputs(tilingData->wideFastPipeline != 0);
            blockMmad3.SetFastCachedInputs(tilingData->wideFastPipeline != 0);
            const MaskLayoutInfo maskLayout = GetMaskLayout(tilingData);
            uint32_t mmapFlag = 0;
            uint32_t processedWorks = 0;

            AscendC::SyncAll<false>();
            SetFlag();
            // Match FA's ND/NZ load semantics and make every padded lane
            // deterministic for short Q/KV tails.
            AscendC::SetLoadDataPaddingValue<uint64_t>(0);
            AscendC::SetNdParaImpl(0x1);
            while (processedWorks < workLength) {
                WideQInfo qInfos[WIDE_MAX_Q_TASKS];
                TaskInfo nextWork;
                const uint32_t qCount = BuildWideWorkItem(
                    gActualQseqlen, gActualKvseqlen, tilingData,
                    groupSize, current, qInfos, nextWork,
                    processedWorks + 1 < workLength);
                if (tilingData->wideQOperandCache != 0) {
                    blockMmad3.AcquireFullPreloadReuse();
                    for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                        const auto &q = qInfos[qLocal];
                        const LayoutB3 layout(q.curCalQSize, tilingData->headDim,
                            tilingData->headDim);
                        const GemmCoord shape{128, tilingData->headDim, q.curCalQSize};
                        blockMmad3.PreloadFullB(gDout[q.qOffset], layout, shape, qLocal);
                        blockMmad3.PreloadFullB(gQ[q.qOffset], layout, shape,
                            qLocal + GROUPED_Q_BLOCKS);
                    }
                    blockMmad3.FinishFullPreload();
                }
                uint32_t qLimits[WIDE_MAX_Q_TASKS] = {};
                uint64_t maskRowOffsets[WIDE_MAX_Q_TASKS] = {};
                uint32_t maxKvLimit = 0;
                const uint32_t kvBlockCount =
                    (current.kvSeqlen + tilingData->blockShapeY - 1) /
                    tilingData->blockShapeY;
                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    const TaskInfo qTask = MakeWideQTask(
                        current, qInfos, qLocal);
                    qLimits[qLocal] = GetKvBlockLimit(
                        params, qTask, kvBlockCount, tilingData);
                    maskRowOffsets[qLocal] =
                        GetMaskRowOffset(qTask, maskLayout);
                    if (qLimits[qLocal] > maxKvLimit) {
                        maxKvLimit = qLimits[qLocal];
                    }
                }

                WideMaskScanState maskScanState;
                ResetWideMaskScanState(maskScanState);
                WidePacketInfo packets[2];
                uint32_t stage = 0;
                uint32_t previousStage = 0;
                bool havePrevious = false;
                WideQMask qSeenMask = 0;
                uint32_t kvStart = 0;
                uint32_t qStart = 0;
                while (kvStart < maxKvLimit) {
                    WidePacketInfo &packet = packets[stage];
                    const uint64_t workspaceBase =
                        static_cast<uint64_t>(coreIdx) *
                            tilingData->wideWorkspaceCoreElements +
                        static_cast<uint64_t>(stage) *
                            tilingData->wideWorkspaceStageElements;
                    BuildWidePacket(params, tilingData, qLimits,
                        maskRowOffsets, maskLayout, qCount, maxKvLimit,
                        kvStart, qStart, workspaceBase, maskScanState, packet);
                    kvStart = packet.nextKvStart;
                    qStart = packet.nextQStart;
                    if (packet.edgeCount == 0) {
                        continue;
                    }

                    // Gradient P/dS reads and the following Cube1 operand
                    // loads share PIPE_MTE2 and retire in program order.  The
                    // first Cube1 MTE2->MTE1 completion therefore also proves
                    // that the old low-precision stage has been consumed
                    // before raw-ready lets AIV overwrite it.  A separate
                    // whole-pipeline MTE2->FIX fence would only serialize the
                    // independent gradient-read and raw-write GM planes.
                    SyncExtendedL1Reuse();
                    RunWideCube1(packet, current, qInfos, tilingData,
                        gQ, gK, gV, gDout, gS, gDp,
                        blockMmad1, mmapFlag, blockMmad3);
                    const uint32_t cubeToVec = stage == 0 ?
                        WIDE_CUBE2VEC_STAGE0 : WIDE_CUBE2VEC_STAGE1;
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(cubeToVec);

                    if (havePrevious) {
                        const uint32_t vecToCube = previousStage == 0 ?
                            WIDE_VEC2CUBE_STAGE0 : WIDE_VEC2CUBE_STAGE1;
                        AscendC::WaitEvent(vecToCube);
                        RunWideGrad(packets[previousStage], current, qInfos,
                            qCount, tilingData, gK, gQ, gDout, gP, gDs,
                            gDq, gDk, gDv, blockMmad2, blockMmad3,
                            qSeenMask);
                    }
                    previousStage = stage;
                    stage = 1 - stage;
                    havePrevious = true;
                }
                if (havePrevious) {
                    const uint32_t vecToCube = previousStage == 0 ?
                        WIDE_VEC2CUBE_STAGE0 : WIDE_VEC2CUBE_STAGE1;
                    AscendC::WaitEvent(vecToCube);
                    RunWideGrad(packets[previousStage], current, qInfos,
                        qCount, tilingData, gK, gQ, gDout, gP, gDs,
                        gDq, gDk, gDv, blockMmad2, blockMmad3,
                        qSeenMask);
                    // Publish a real reuse token for the final high-L1 reads.
                    SyncExtendedL1Reuse();
                }
                if (tilingData->wideQOperandCache != 0) {
                    blockMmad3.ReleaseFullPreloadReuse();
                }
                ++processedWorks;
                if (processedWorks < workLength) {
                    current = nextWork;
                }
            }
            WaitWideFlag();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2POST);
        }


        __aicore__ inline void ProcessGroupedAic(
            const Params &params, __gm__ BlockSparseAttentionGradTilingData *tilingData)
        {
            const uint32_t coreIdx = AscendC::GetBlockIdx();
            const uint32_t groupSize = tilingData->numHeads / tilingData->kvHeads;
            const uint32_t taskLength = tilingData->taskNumPerCore +
                (tilingData->tailTaskNum > coreIdx ? 1 : 0);
            const uint64_t groupedFp32Bytes =
                static_cast<uint64_t>(tilingData->usedVecCoreNum / 2) *
                GROUPED_WORKSPACE_CORE_SIZE * sizeof(float);

            AscendC::GlobalTensor<ElementInput> gQ;
            AscendC::GlobalTensor<ElementInput> gK;
            AscendC::GlobalTensor<ElementInput> gV;
            AscendC::GlobalTensor<ElementA1> gDout;
            AscendC::GlobalTensor<int64_t> gActualQseqlen;
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;
            AscendC::GlobalTensor<float> gS;
            AscendC::GlobalTensor<float> gDp;
            AscendC::GlobalTensor<ElementInput> gP;
            AscendC::GlobalTensor<ElementInput> gDs;
            AscendC::GlobalTensor<float> gDq;
            AscendC::GlobalTensor<float> gDk;
            AscendC::GlobalTensor<float> gDv;
            if (UsePackedTndInputs(tilingData)) {
                GM_ADDR packed = params.workspace + GroupedPackedInputBase(tilingData);
                const uint64_t qBytes = tilingData->dqSize * sizeof(ElementInput);
                const uint64_t kvBytes = tilingData->dkvSize * sizeof(ElementInput);
                gQ.SetGlobalBuffer((__gm__ ElementInput *)packed);
                gDout.SetGlobalBuffer((__gm__ ElementInput *)(packed + qBytes));
                gK.SetGlobalBuffer((__gm__ ElementInput *)(packed + 2 * qBytes));
                gV.SetGlobalBuffer((__gm__ ElementInput *)(packed + 2 * qBytes + kvBytes));
            } else {
                gQ.SetGlobalBuffer((__gm__ ElementInput *)params.q);
                gK.SetGlobalBuffer((__gm__ ElementInput *)params.k);
                gV.SetGlobalBuffer((__gm__ ElementInput *)params.v);
                gDout.SetGlobalBuffer((__gm__ ElementInput *)params.dout);
            }
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
            gS.SetGlobalBuffer((__gm__ float *)params.workspace);
            gP.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + WORKSPACE_P16_OFFSET));
            gDp.SetGlobalBuffer((__gm__ float *)(params.workspace + tilingData->sOutSize));
            gDs.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace +
                tilingData->sOutSize + WORKSPACE_P16_OFFSET));
            gDq.SetGlobalBuffer((__gm__ float *)(params.workspace +
                tilingData->sOutSize + tilingData->dPOutSize));
            gDk.SetGlobalBuffer((__gm__ float *)(params.workspace + tilingData->sOutSize +
                tilingData->dPOutSize + tilingData->dQOutSize));
            gDv.SetGlobalBuffer((__gm__ float *)(params.workspace + tilingData->sOutSize +
                tilingData->dPOutSize + tilingData->dQOutSize + tilingData->dKOutSize));

            TaskInfo current;
            initTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData,
                tilingData->numHeads, tilingData->kvHeads, groupSize, tilingData->headDim,
                tilingData->maxQSeqlen, tilingData->maxKvSeqlen,
                tilingData->blockShapeX, tilingData->basicQBlockSize,
                tilingData->inputLayout, coreIdx, current);

            // The active Cube1 CachedB/Packed schedules use the two A
            // ping-pong slots and B cache slots 0/1 in the compact region below
            // 192 KiB.  This leaves blockMmad3's persistent dOut/Q slots
            // [192 KiB, 448 KiB) untouched across all KV segments of the
            // current Q group.
            BlockMmadBSAG1 blockMmad1(resource);
            BlockMmadBSAG2 blockMmad2(resource, 0, PINGPONG_OFFSET_2, true);
            BlockMmadBSAG3 blockMmad3(resource, L1_SIZE_OFFSET * 7 / 2,
                PINGPONG_OFFSET_4, true, false, true);
            gP.SetGlobalBuffer(
                (__gm__ ElementInput *)(params.workspace + groupedFp32Bytes));
            gDs.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace +
                tilingData->sOutSize + groupedFp32Bytes));
            uint32_t mmapFlag = 0;
            SegmentInfo segments[2];
            uint32_t stage = 0;
            uint32_t previousStage = 0;
            bool havePrevious = false;
            bool previousExclusiveKvOwner = false;
            uint32_t processedTasks = 0;
            uint64_t writtenKvBlocks = 0;
            uint64_t cachedQOffsets[GROUPED_Q_BLOCKS] = {};
            uint32_t cachedQMask = 0;
            const uint32_t tasksPerFullQBlock =
                (tilingData->blockShapeX + tilingData->basicQBlockSize - 1) /
                tilingData->basicQBlockSize;
            const uint32_t fullQBlocks =
                tilingData->maxQSeqlen / tilingData->blockShapeX;
            const uint32_t qRemainder = tilingData->maxQSeqlen -
                fullQBlocks * tilingData->blockShapeX;
            const uint32_t qBlocksPerHead = fullQBlocks * tasksPerFullQBlock +
                (qRemainder + tilingData->basicQBlockSize - 1) /
                    tilingData->basicQBlockSize;
            const MaskLayoutInfo maskLayout = GetMaskLayout(tilingData);
            bool exclusiveKvOwner = false;
            AscendC::SyncAll<false>();
            SetFlag();
            while (processedTasks < taskLength) {
                const bool startsWholeHead = current.curQSeqIdx == 0;
                if (startsWholeHead) {
                    if (processedTasks != 0 && havePrevious) {
                        RunGroupedGradDispatch(segments[previousStage], tilingData,
                            gK, gQ, gDout, gP, gDs,
                            gDq, gDk, gDv, blockMmad2, blockMmad3, mmapFlag,
                            previousExclusiveKvOwner, writtenKvBlocks,
                            cachedQOffsets, cachedQMask, false);
                        havePrevious = false;
                    }
                    writtenKvBlocks = 0;
                    cachedQMask = 0;
                }
                if (startsWholeHead) {
                    exclusiveKvOwner =
                        tilingData->numHeads == tilingData->kvHeads &&
                        taskLength - processedTasks >= qBlocksPerHead;
                }
                TaskInfo qInfos[GROUPED_Q_BLOCKS];
                uint32_t qLimits[GROUPED_Q_BLOCKS];
                qInfos[0] = current;
                uint32_t qCount = 1;
                TaskInfo nextGroup;
                bool nextReady = false;
                while (qCount < GROUPED_ACTIVE_Q_BLOCKS &&
                    processedTasks + qCount < taskLength) {
                    TaskInfo candidate;
                    updateNextGroupedTaskInfo(gActualQseqlen, gActualKvseqlen,
                        tilingData->numHeads, tilingData->kvHeads, groupSize,
                        tilingData->headDim, tilingData->blockShapeX,
                        tilingData->basicQBlockSize, tilingData->inputLayout,
                        qInfos[qCount - 1], candidate);
                    if (candidate.curBatchIdx != current.curBatchIdx ||
                        candidate.curHeadIdx != current.curHeadIdx) {
                        nextGroup = candidate;
                        nextReady = true;
                        break;
                    }
                    qInfos[qCount++] = candidate;
                }
                if (processedTasks + qCount < taskLength && !nextReady) {
                    updateNextGroupedTaskInfo(gActualQseqlen, gActualKvseqlen,
                        tilingData->numHeads, tilingData->kvHeads, groupSize,
                        tilingData->headDim, tilingData->blockShapeX,
                        tilingData->basicQBlockSize, tilingData->inputLayout,
                        qInfos[qCount - 1], nextGroup);
                    nextReady = true;
                }

                uint32_t maxKvLimit = 0;
                const uint32_t kvBlockCount =
                    (qInfos[0].kvSeqlen + tilingData->blockShapeY - 1) /
                    tilingData->blockShapeY;
                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    qLimits[qLocal] = GetKvBlockLimit(params, qInfos[qLocal], kvBlockCount, tilingData);
                    if (qLimits[qLocal] > maxKvLimit) {
                        maxKvLimit = qLimits[qLocal];
                    }
                }
                uint64_t maskRowOffsets[GROUPED_Q_BLOCKS] = {};
                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    maskRowOffsets[qLocal] = GetMaskRowOffset(qInfos[qLocal], maskLayout);
                }
                const bool packedSegments = tilingData->basicKVBlockSize < 128 &&
                    tilingData->blockShapeX >= 16;
                uint32_t packedKvStarts[GROUPED_Q_BLOCKS] = {};
                uint32_t kvStart = 0;
                while (true) {
                    bool traversalDone = kvStart >= maxKvLimit;
                    if (packedSegments) {
                        traversalDone = true;
                        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                            traversalDone &= packedKvStarts[qLocal] >= qLimits[qLocal];
                        }
                    }
                    if (traversalDone) {
                        break;
                    }
                    const uint32_t kvCount = maxKvLimit - kvStart < GROUPED_KV_BLOCKS ?
                        maxKvLimit - kvStart : GROUPED_KV_BLOCKS;
                    SegmentInfo &segment = segments[stage];
                    const uint64_t workspaceBase = static_cast<uint64_t>(coreIdx) *
                        GROUPED_WORKSPACE_CORE_SIZE + static_cast<uint64_t>(stage) *
                        GROUPED_WORKSPACE_STAGE_SIZE;
                    BuildGroupedSegment(params, tilingData, qInfos, qLimits, qCount,
                        kvStart, kvCount, packedKvStarts, maskRowOffsets,
                        workspaceBase, segment);
                    if (!packedSegments) {
                        kvStart = segment.nextKvStart;
                    }
                    if (segment.tileCount == 0) {
                        continue;
                    }
                    if (tilingData->basicKVBlockSize < 128 &&
                        tilingData->blockShapeX >= 16) {
                        RunGroupedCube1Packed(segment, tilingData,
                            gQ, gK, gV, gDout, gS, gDp, gP,
                            blockMmad1, mmapFlag);
                    } else if (tilingData->headDim == 128) {
                        // Use the stable L1 CachedB pipeline.  Keeping K/V in
                        // L0B across unrelated gradient phases aliases event
                        // lifetimes under sustained execution.  Keep the raw
                        // QK^T and dO*V^T results in FP32, matching FAG: only
                        // P and dS are rounded to the input dtype after the
                        // vector softmax/softmax-grad arithmetic is complete.
                        RunGroupedCube1CachedB(segment, tilingData,
                            gQ, gK, gV, gDout, gS, gDp,
                            blockMmad1, mmapFlag);
                    } else {
                        RunGroupedCube1CachedB(segment, tilingData,
                            gQ, gK, gV, gDout, gS, gDp, blockMmad1, mmapFlag);
                    }
                    // Compact Cube1 does not alias blockMmad3's persistent
                    // dOut/Q slots.  Keep cached rows across KV segments;
                    // cachedQOffsets invalidates individual slots when the
                    // Q group changes, while the whole-head boundary above
                    // resets the mask after draining the pending segment.
                    AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);
                    if (havePrevious) {
                        RunGroupedGradDispatch(segments[previousStage], tilingData,
                            gK, gQ, gDout, gP, gDs,
                            gDq, gDk, gDv, blockMmad2, blockMmad3, mmapFlag,
                            previousExclusiveKvOwner, writtenKvBlocks, cachedQOffsets,
                            cachedQMask, false);
                    }
                    previousStage = stage;
                    previousExclusiveKvOwner = exclusiveKvOwner;
                    stage = 1 - stage;
                    havePrevious = true;
                }
                processedTasks += qCount;
                if (processedTasks < taskLength && nextReady) {
                    current = nextGroup;
                }
            }
            if (havePrevious) {
                DrainFinalCube1Flags();
                RunGroupedGradDispatch(segments[previousStage], tilingData,
                    gK, gQ, gDout, gP, gDs,
                    gDq, gDk, gDv, blockMmad2, blockMmad3, mmapFlag,
                    previousExclusiveKvOwner, writtenKvBlocks, cachedQOffsets,
                    cachedQMask, true);
            }
            WaitGroupedFlag();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2POST);
        }

        __aicore__ inline void ProcessGroupedAiv(
            const Params &params, __gm__ BlockSparseAttentionGradTilingData *tilingData)
        {
            const uint32_t vecCoreIdx = AscendC::GetBlockIdx();
            const uint32_t coreIdx = vecCoreIdx / 2;
            const uint32_t subBlockIdx = vecCoreIdx % 2;
            const uint32_t groupSize = tilingData->numHeads / tilingData->kvHeads;
            const uint32_t taskLength = tilingData->taskNumPerCore +
                (tilingData->tailTaskNum > coreIdx ? 1 : 0);
            const uint64_t groupedFp32Bytes =
                static_cast<uint64_t>(tilingData->usedVecCoreNum / 2) *
                GROUPED_WORKSPACE_CORE_SIZE * sizeof(float);

            AscendC::GlobalTensor<int64_t> gActualQseqlen;
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
            TaskInfo current;
            initTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData,
                tilingData->numHeads, tilingData->kvHeads, groupSize, tilingData->headDim,
                tilingData->maxQSeqlen, tilingData->maxKvSeqlen,
                tilingData->blockShapeX, tilingData->basicQBlockSize,
                tilingData->inputLayout, coreIdx, current);
            GM_ADDR sftmgGm = params.workspace + tilingData->sOutSize + tilingData->dPOutSize +
                tilingData->dQOutSize + tilingData->dKOutSize + tilingData->dVOutSize;
            SfmgParams sfmgParams(params.dout, params.out, params.actualQseqlen,
                sftmgGm, params.tiling);
            EpilogueFAGSfmg vecSftmg(sfmgParams);
            EpilogueFAGOp softmaxOp;
            softmaxOp.BeginGroupedPipeline(params.softmaxLse, params.actualQseqlen,
                params.actualKvseqlen, sftmgGm, params.tiling);
            const MaskLayoutInfo maskLayout = GetMaskLayout(tilingData);
            uint32_t stage = 0;
            uint32_t processedTasks = 0;

            while (processedTasks < taskLength) {
                TaskInfo qInfos[GROUPED_Q_BLOCKS];
                uint32_t qLimits[GROUPED_Q_BLOCKS];
                qInfos[0] = current;
                uint32_t qCount = 1;
                TaskInfo nextGroup;
                bool nextReady = false;
                while (qCount < GROUPED_ACTIVE_Q_BLOCKS &&
                    processedTasks + qCount < taskLength) {
                    TaskInfo candidate;
                    updateNextGroupedTaskInfo(gActualQseqlen, gActualKvseqlen,
                        tilingData->numHeads, tilingData->kvHeads, groupSize,
                        tilingData->headDim, tilingData->blockShapeX,
                        tilingData->basicQBlockSize, tilingData->inputLayout,
                        qInfos[qCount - 1], candidate);
                    if (candidate.curBatchIdx != current.curBatchIdx ||
                        candidate.curHeadIdx != current.curHeadIdx) {
                        nextGroup = candidate;
                        nextReady = true;
                        break;
                    }
                    qInfos[qCount++] = candidate;
                }
                if (processedTasks + qCount < taskLength && !nextReady) {
                    updateNextGroupedTaskInfo(gActualQseqlen, gActualKvseqlen,
                        tilingData->numHeads, tilingData->kvHeads, groupSize,
                        tilingData->headDim, tilingData->blockShapeX,
                        tilingData->basicQBlockSize, tilingData->inputLayout,
                        qInfos[qCount - 1], nextGroup);
                    nextReady = true;
                }

                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    const uint32_t firstHalfRows = (qInfos[qLocal].curCalQSize + 1) / 2;
                    const uint32_t subRows = subBlockIdx == 0 ? firstHalfRows :
                        qInfos[qLocal].curCalQSize - firstHalfRows;
                    // startIdx addresses the flattened (t, head) D rows
                    // ([T, numHeads, 8] workspace), so the half-row offset of
                    // the second AIV must be scaled by numHeads exactly like
                    // the VecOp call site.  Without it AIV-1 both reads
                    // (dOut, out) and writes D at rows belonging to the wrong
                    // tokens/heads, corrupting every dS downstream.
                    if (tilingData->inputLayout == 0) {
                        vecSftmg(qInfos[qLocal].qOffset / tilingData->headDim +
                            subBlockIdx * firstHalfRows * tilingData->numHeads, subRows);
                    } else {
                        vecSftmg(qInfos[qLocal].qOffset / tilingData->headDim +
                            subBlockIdx * firstHalfRows, subRows);
                    }
                }
                // vecSftmg returns after queuing D's UB->GM copy.  The grouped
                // input prefetch reads D immediately, so explicitly order the
                // first GM->UB read after all of those MTE3 writes.  The old
                // per-tile path had enough P/vector work before CopyDIn to hide
                // this dependency; prefetching exposed the RAW hazard.
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

                uint32_t maxKvLimit = 0;
                const uint32_t kvBlockCount =
                    (qInfos[0].kvSeqlen + tilingData->blockShapeY - 1) /
                    tilingData->blockShapeY;
                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    qLimits[qLocal] = GetKvBlockLimit(params, qInfos[qLocal], kvBlockCount, tilingData);
                    if (qLimits[qLocal] > maxKvLimit) {
                        maxKvLimit = qLimits[qLocal];
                    }
                }
                uint64_t maskRowOffsets[GROUPED_Q_BLOCKS] = {};
                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    maskRowOffsets[qLocal] = GetMaskRowOffset(qInfos[qLocal], maskLayout);
                }
                // D and LSE depend only on q, so retain all four metadata
                // slots across the group's KV segments.  The epilogue owns
                // non-overlapping UB storage for the full group.
                softmaxOp.ResetGroupedQMetadata();
                const bool packedSegments = tilingData->basicKVBlockSize < 128 &&
                    tilingData->blockShapeX >= 16;
                uint32_t packedKvStarts[GROUPED_Q_BLOCKS] = {};
                uint32_t kvStart = 0;
                while (true) {
                    bool traversalDone = kvStart >= maxKvLimit;
                    if (packedSegments) {
                        traversalDone = true;
                        for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                            traversalDone &= packedKvStarts[qLocal] >= qLimits[qLocal];
                        }
                    }
                    if (traversalDone) {
                        break;
                    }
                    const uint32_t kvCount = maxKvLimit - kvStart < GROUPED_KV_BLOCKS ?
                        maxKvLimit - kvStart : GROUPED_KV_BLOCKS;
                    SegmentInfo segment;
                    const uint64_t workspaceBase = static_cast<uint64_t>(coreIdx) *
                        GROUPED_WORKSPACE_CORE_SIZE + static_cast<uint64_t>(stage) *
                        GROUPED_WORKSPACE_STAGE_SIZE;
                    BuildGroupedSegment(params, tilingData, qInfos, qLimits, qCount,
                        kvStart, kvCount, packedKvStarts, maskRowOffsets,
                        workspaceBase, segment);
                    if (!packedSegments) {
                        kvStart = segment.nextKvStart;
                    }
                    if (segment.tileCount == 0) {
                        continue;
                    }
                    AscendC::WaitEvent(CUBE2VEC);
                    // Tiles are q-major inside one segment.  Four metadata
                    // slots retain D/LSE for the complete q group.
                    SfmParams tileParams[GROUPED_TILES];
                    for (uint32_t t = 0; t < segment.tileCount; ++t) {
                        TaskInfo &tile = segment.tiles[t];
                        const uint64_t firstHalfRows = (tile.curCalQSize + 1) / 2;
                        const uint64_t executeRow = subBlockIdx == 0 ? firstHalfRows :
                            tile.curCalQSize - firstHalfRows;
                        const uint64_t coreOffset =
                            subBlockIdx * firstHalfRows * tile.curCalKVSize;
                        const uint64_t curVecCoreS1Idx = tile.curQSeqIdx +
                            subBlockIdx * firstHalfRows;
                        const uint64_t vector16Soffset =
                            groupedFp32Bytes +
                            (tile.sOffset + coreOffset) * sizeof(ElementInput);
                        const uint64_t vectorInputOffset =
                            (tile.sOffset + coreOffset) * sizeof(float);
                        tileParams[t] = SfmParams(
                            params.workspace + vectorInputOffset,
                            params.softmaxLse,
                            params.workspace + tilingData->sOutSize + vectorInputOffset,
                            params.actualQseqlen,
                            params.actualKvseqlen,
                            sftmgGm,
                            params.workspace + vector16Soffset,
                            params.workspace + tilingData->sOutSize + vector16Soffset,
                            params.tiling,
                            executeRow,
                            tile.curCalKVSize,
                            executeRow * tile.curCalKVSize,
                            tile.curBatchIdx,
                            tile.curHeadIdx,
                            curVecCoreS1Idx,
                            tile.groupQIdx,
                            tile.kvValidSize);
                    }
                    uint32_t inputStage = 0;
                    softmaxOp.PrefetchGrouped(tileParams[0], inputStage);
                    for (uint32_t t = 0; t < segment.tileCount; ++t) {
                        if (t + 1 < segment.tileCount) {
                            softmaxOp.PrefetchGrouped(tileParams[t + 1], 1 - inputStage);
                        }
                        softmaxOp.ComputeGrouped(tileParams[t], inputStage);
                        if (t + 1 == segment.tileCount ||
                            segment.tiles[t + 1].groupQIdx != segment.tiles[t].groupQIdx) {
                            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(VEC2CUBE);
                        }
                        inputStage = 1 - inputStage;
                    }
                    stage = 1 - stage;
                }
                processedTasks += qCount;
                if (processedTasks < taskLength && nextReady) {
                    current = nextGroup;
                }
            }
            softmaxOp.EndGroupedPipeline();
        }

        __aicore__ inline void FillWideSfmParams(
            SfmParams &tileParams, const Params &params,
            const TaskInfo &tile,
            const __gm__ BlockSparseAttentionGradTilingData *tilingData,
            GM_ADDR sftmgGm, uint32_t subBlockIdx,
            uint32_t metadataSlot) const
        {
            const uint64_t firstHalfRows = (tile.curCalQSize + 1) / 2;
            const uint64_t executeRow = subBlockIdx == 0 ? firstHalfRows :
                tile.curCalQSize - firstHalfRows;
            const uint64_t curVecCoreS1Idx = tile.curQSeqIdx +
                subBlockIdx * firstHalfRows;
            // Raw S/dP are standard FP32 NZ tiles.  Keep the tile base here;
            // the vector epilogue applies rawRowOffset * C0 internally.
            const uint64_t rawByteOffset = tile.sOffset * sizeof(float);
            // P/dS remain in native zN, so both vector subcores address the
            // tile base and scatter their own row range using rawRowOffset.
            const uint64_t lowpByteOffset = tilingData->wideRawPlaneBytes +
                tile.sOffset * sizeof(ElementInput);
            tileParams = SfmParams(
                params.workspace + rawByteOffset,
                params.softmaxLse,
                params.workspace + tilingData->sOutSize + rawByteOffset,
                params.actualQseqlen,
                params.actualKvseqlen,
                sftmgGm,
                params.workspace + lowpByteOffset,
                params.workspace + tilingData->sOutSize + lowpByteOffset,
                params.tiling,
                executeRow,
                tile.curCalKVSize,
                executeRow * tile.curCalKVSize,
                tile.curBatchIdx,
                tile.curHeadIdx,
                curVecCoreS1Idx,
                metadataSlot,
                tile.kvValidSize,
                tile.curCalQSize,
                subBlockIdx * firstHalfRows,
                true);
        }

        __aicore__ inline void ProcessWideAiv(
            const Params &params,
            __gm__ BlockSparseAttentionGradTilingData *tilingData)
        {
            const uint32_t vecCoreIdx = AscendC::GetBlockIdx();
            const uint32_t coreIdx = vecCoreIdx / 2;
            const uint32_t subBlockIdx = vecCoreIdx % 2;
            const uint32_t groupSize = tilingData->numHeads / tilingData->kvHeads;
            const uint32_t workLength = tilingData->taskNumPerCore +
                (tilingData->tailTaskNum > coreIdx ? 1 : 0);

            AscendC::GlobalTensor<int64_t> gActualQseqlen;
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
            TaskInfo current;
            initTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData,
                tilingData->numHeads, tilingData->kvHeads, groupSize,
                tilingData->headDim, tilingData->maxQSeqlen,
                tilingData->maxKvSeqlen, tilingData->blockShapeX,
                tilingData->basicQBlockSize, tilingData->inputLayout,
                coreIdx, current);

            GM_ADDR sftmgGm = params.workspace + tilingData->sOutSize +
                tilingData->dPOutSize + tilingData->dQOutSize +
                tilingData->dKOutSize + tilingData->dVOutSize;
            SfmgParams sfmgParams(params.dout, params.out,
                params.actualQseqlen, sftmgGm, params.tiling);
            EpilogueFAGSfmg vecSftmg(sfmgParams);
            EpilogueFAGOp softmaxOp;
            softmaxOp.BeginGroupedPipeline(params.softmaxLse,
                params.actualQseqlen, params.actualKvseqlen,
                sftmgGm, params.tiling);
            const MaskLayoutInfo maskLayout = GetMaskLayout(tilingData);
            uint32_t processedWorks = 0;

            while (processedWorks < workLength) {
                WideQInfo qInfos[WIDE_MAX_Q_TASKS];
                TaskInfo nextWork;
                const uint32_t qCount = BuildWideWorkItem(
                    gActualQseqlen, gActualKvseqlen, tilingData,
                    groupSize, current, qInfos, nextWork,
                    processedWorks + 1 < workLength);

                // D = rowsum(dO * O) is Q-only.  Produce it once per Q task,
                // then order all following metadata MTE2 reads after the GM
                // writes from both paired AIVs.
                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    const uint32_t firstHalfRows =
                        (qInfos[qLocal].curCalQSize + 1) / 2;
                    const uint32_t subRows = subBlockIdx == 0 ? firstHalfRows :
                        qInfos[qLocal].curCalQSize - firstHalfRows;
                    if (subRows != 0) {
                        vecSftmg(qInfos[qLocal].qOffset / tilingData->headDim +
                            subBlockIdx * firstHalfRows, subRows);
                    }
                }
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

                uint32_t qLimits[WIDE_MAX_Q_TASKS] = {};
                uint64_t maskRowOffsets[WIDE_MAX_Q_TASKS] = {};
                uint32_t maxKvLimit = 0;
                const uint32_t kvBlockCount =
                    (current.kvSeqlen + tilingData->blockShapeY - 1) /
                    tilingData->blockShapeY;
                for (uint32_t qLocal = 0; qLocal < qCount; ++qLocal) {
                    const TaskInfo qTask = MakeWideQTask(
                        current, qInfos, qLocal);
                    qLimits[qLocal] = GetKvBlockLimit(
                        params, qTask, kvBlockCount, tilingData);
                    maskRowOffsets[qLocal] =
                        GetMaskRowOffset(qTask, maskLayout);
                    if (qLimits[qLocal] > maxKvLimit) {
                        maxKvLimit = qLimits[qLocal];
                    }
                }

                WideMaskScanState maskScanState;
                ResetWideMaskScanState(maskScanState);
                WidePacketInfo packet;
                uint32_t stage = 0;
                uint32_t kvStart = 0;
                uint32_t qStart = 0;
                softmaxOp.BeginWideRawStageReuse();
                while (kvStart < maxKvLimit) {
                    const uint64_t workspaceBase =
                        static_cast<uint64_t>(coreIdx) *
                            tilingData->wideWorkspaceCoreElements +
                        static_cast<uint64_t>(stage) *
                            tilingData->wideWorkspaceStageElements;
                    BuildWidePacket(params, tilingData, qLimits,
                        maskRowOffsets, maskLayout, qCount, maxKvLimit,
                        kvStart, qStart, workspaceBase, maskScanState, packet);
                    kvStart = packet.nextKvStart;
                    qStart = packet.nextQStart;
                    if (packet.edgeCount == 0) {
                        continue;
                    }
                    const uint32_t cubeToVec = stage == 0 ?
                        WIDE_CUBE2VEC_STAGE0 : WIDE_CUBE2VEC_STAGE1;
                    AscendC::WaitEvent(cubeToVec);

                    // One cross-core token covers the complete packet.  Keep
                    // the q-major edge view, but flatten all non-empty Q runs
                    // into one two-stage input pipeline.  Sparse packets
                    // usually contain only one edge per Q; restarting
                    // prefetch at every Q therefore serialized nearly every
                    // GM->UB transfer behind the previous vector tile.
                    uint16_t validEdges[WIDE_MAX_EDGES];
                    uint8_t validMetadataSlots[WIDE_MAX_EDGES];
                    uint32_t validEdgeCount = 0;
                    uint32_t activeQCount = 0;
                    for (uint32_t activeQ = 0;
                        activeQ < packet.activeQCount; ++activeQ) {
                        const WideQDesc &qDesc = packet.activeQ[activeQ];
                        const uint32_t qLocal = qDesc.qLocal;
                        const uint32_t qBegin = qDesc.edgeBegin;
                        const uint32_t qEnd = qBegin + qDesc.edgeCount;
                        // A one-row Q tail is owned entirely by AIV0.  AIV1
                        // still publishes the packet-level cross-core token,
                        // but must not issue a zero-block DMA.
                        const uint32_t firstHalfRows =
                            (qInfos[qLocal].curCalQSize + 1) / 2;
                        const uint32_t executeRows = subBlockIdx == 0 ?
                            firstHalfRows :
                            qInfos[qLocal].curCalQSize - firstHalfRows;
                        if (executeRows == 0) {
                            continue;
                        }
                        const uint8_t metadataSlot = static_cast<uint8_t>(
                            activeQCount & 3U);
                        ++activeQCount;
                        for (uint32_t qEdge = qBegin;
                            qEdge < qEnd; ++qEdge) {
                            validEdges[validEdgeCount] = packet.qOrder[qEdge];
                            validMetadataSlots[validEdgeCount] = metadataSlot;
                            ++validEdgeCount;
                        }
                    }

                    if (validEdgeCount != 0) {
                        // Rotate four metadata slots by active Q, independently
                        // of the two raw input stages.  All edges of one Q can
                        // then reuse one LSE/D copy.  Before a slot wraps, at
                        // least three intervening Q runs have already forced
                        // both raw stages through their V->MTE2 reuse fences,
                        // so every previous vector consumer has retired.
                        softmaxOp.ResetGroupedQMetadata();
                        SfmParams tileParams[2];
                        uint32_t inputStage = 0;
                        const uint32_t firstEdge = validEdges[0];
                        const uint32_t firstMetadataSlot =
                            validMetadataSlots[0];
                        TaskInfo firstTile = MakeWideTile(current, qInfos,
                            packet, firstEdge, tilingData);
                        FillWideSfmParams(tileParams[0], params, firstTile,
                            tilingData, sftmgGm, subBlockIdx,
                            firstMetadataSlot);
                        softmaxOp.InvalidateGroupedQMetadataSlot(
                            firstMetadataSlot);
                        softmaxOp.PrefetchGrouped(tileParams[0], 0);

                        for (uint32_t edgeIdx = 0;
                            edgeIdx < validEdgeCount; ++edgeIdx) {
                            if (edgeIdx + 1 < validEdgeCount) {
                                const uint32_t nextStage = 1 - inputStage;
                                const uint32_t nextEdge =
                                    validEdges[edgeIdx + 1];
                                const uint32_t currentEdge =
                                    validEdges[edgeIdx];
                                const uint32_t nextQ = WideEdgeQ(
                                    packet.edgeCode[nextEdge]);
                                const uint32_t currentQ = WideEdgeQ(
                                    packet.edgeCode[currentEdge]);
                                const uint32_t nextMetadataSlot =
                                    validMetadataSlots[edgeIdx + 1];
                                TaskInfo nextTile = MakeWideTile(current, qInfos,
                                    packet, nextEdge, tilingData);
                                FillWideSfmParams(tileParams[nextStage], params,
                                    nextTile, tilingData, sftmgGm, subBlockIdx,
                                    nextMetadataSlot);
                                if (nextQ != currentQ) {
                                    softmaxOp.InvalidateGroupedQMetadataSlot(
                                        nextMetadataSlot);
                                }
                                softmaxOp.PrefetchGrouped(
                                    tileParams[nextStage], nextStage);
                            }
                            softmaxOp.ComputeGrouped(
                                tileParams[inputStage], inputStage);
                            inputStage = 1 - inputStage;
                        }
                    }
                    const uint32_t vecToCube = stage == 0 ?
                        WIDE_VEC2CUBE_STAGE0 : WIDE_VEC2CUBE_STAGE1;
                    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(vecToCube);
                    stage = 1 - stage;
                }
                softmaxOp.EndWideRawStageReuse();
                ++processedWorks;
                if (processedWorks < workLength) {
                    current = nextWork;
                }
            }
            softmaxOp.EndGroupedPipeline();
        }

        // Methods
        __aicore__ inline
        BlockSparseAttentionGradKernel() {}

        template <int32_t CORE_TYPE = g_coreType>
        __aicore__ inline
        void operator()(Params const &params);

        template <>
        __aicore__ inline
        void operator()<AscendC::AIC>(Params const &params)
        {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);
            uint32_t batch = tilingData->batch;
            uint32_t numHeads = tilingData->numHeads;
            uint32_t kvHeads = tilingData->kvHeads;
            uint32_t groupSize = numHeads / kvHeads;
            uint32_t headDim = tilingData->headDim;
            uint32_t maxQSeqlen = tilingData->maxQSeqlen;
            uint32_t maxKvSeqlen = tilingData->maxKvSeqlen;
            uint32_t inputLayout = tilingData->inputLayout;
            uint32_t blockShapeX = tilingData->blockShapeX;
            uint32_t blockShapeY = tilingData->blockShapeY;

            uint64_t sOutSize = tilingData->sOutSize;
            uint64_t dPOutSize = tilingData->dPOutSize;
            uint64_t dQOutSize = tilingData->dQOutSize;
            uint64_t dKOutSize = tilingData->dKOutSize;
            uint64_t dVOutSize = tilingData->dVOutSize;

            uint32_t basicQBlockSize = tilingData->basicQBlockSize;
            uint32_t basicKVBlockSize = tilingData->basicKVBlockSize;
            uint32_t taskNumPerCore = tilingData->taskNumPerCore;
            uint32_t tailTaskNum = tilingData->tailTaskNum;
            uint32_t taskLength = tailTaskNum > coreIdx ? taskNumPerCore + 1 : taskNumPerCore;

            if (groupSize == 0 || blockShapeX == 0 || blockShapeY == 0) {
                return;
            }
            // Wide is a strict subset of the grouped eligibility predicate,
            // so it must be dispatched first. Host-enabled 64x64 tiles use
            // runtime extents; other inputs retain grouped/legacy dispatch.
            if (UseWideSchedule(tilingData)) {
                ProcessWideAic(params, tilingData);
                return;
            }
            if (UseGroupedSchedule(tilingData)) {
                ProcessGroupedAic(params, tilingData);
                return;
            }
            // Initialize global tensors
            AscendC::GlobalTensor<ElementA1> gDout;
            gDout.SetGlobalBuffer((__gm__ ElementInput *)params.dout);
            AscendC::GlobalTensor<ElementInput> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementInput *)params.q);
            AscendC::GlobalTensor<ElementInput> gK;
             gK.SetGlobalBuffer((__gm__ ElementInput *)params.k);
             AscendC::GlobalTensor<ElementInput> gV;
             gV.SetGlobalBuffer((__gm__ ElementInput *)params.v);
             AscendC::GlobalTensor<int64_t> gActualQseqlen;
             gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
             AscendC::GlobalTensor<int64_t> gActualKvseqlen;
             gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);

            AscendC::GlobalTensor<float> gS;
            gS.SetGlobalBuffer((__gm__ float *)params.workspace);
            AscendC::GlobalTensor<ElementInput> gP;
            gP.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + WORKSPACE_P16_OFFSET));

            AscendC::GlobalTensor<float> gDp;
            gDp.SetGlobalBuffer((__gm__ float *)(params.workspace + sOutSize));
            AscendC::GlobalTensor<ElementInput> gDs;
            gDs.SetGlobalBuffer((__gm__ ElementInput *)(params.workspace + sOutSize + WORKSPACE_P16_OFFSET));
            AscendC::GlobalTensor<float> gDq;
            gDq.SetGlobalBuffer((__gm__ float *)(params.workspace + sOutSize + dPOutSize));
            AscendC::GlobalTensor<float> gDk;
            gDk.SetGlobalBuffer((__gm__ float *)(params.workspace + sOutSize + dPOutSize + dQOutSize));
            AscendC::GlobalTensor<float> gDv;
            gDv.SetGlobalBuffer((__gm__ float *)(params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize));

            TaskInfo taskInfo[2];
            TaskInfo preTaskInfo;
            initTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData, numHeads, kvHeads, groupSize, headDim,
                maxQSeqlen, maxKvSeqlen, blockShapeX, basicQBlockSize, inputLayout, coreIdx, taskInfo[0]);
            uint32_t maskQBlockNum = (maxQSeqlen + blockShapeX - 1) / blockShapeX;
            uint32_t maskKvBlockNum = (maxKvSeqlen + blockShapeY - 1) / blockShapeY;
            uint32_t batchBlocks = numHeads * maskQBlockNum * maskKvBlockNum;
            uint32_t headBlocks = maskQBlockNum * maskKvBlockNum;

            uint64_t actualStrideQ = headDim;
            uint64_t actualStrideKV = headDim;
            if (inputLayout == 0) {
                actualStrideQ = numHeads * headDim;
                actualStrideKV = kvHeads * headDim;
            }

            BlockMmadBSAG1 blockMmad1(resource);
            BlockMmadBSAG2 blockMmad2(resource, L1_SIZE_OFFSET, PINGPONG_OFFSET_2, true);
            BlockMmadBSAG3 blockMmad3(resource, L1_SIZE_OFFSET * 2, PINGPONG_OFFSET_4, true);
            uint32_t count = 0;
            uint32_t pingpongFlag = 0;
            uint64_t gSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB;
            uint32_t mmadFlag = 0;

            AscendC::SyncAll<false>();

            SetFlag();
            uint32_t vec2CubeFlag = 0;
            for (uint32_t i = 0; i < taskLength; i++) {
                TaskInfo curInfo = taskInfo[i % 2];
                LayoutA1 cachedLayoutA1;
                LayoutB3 cachedLayoutB3;
                if (inputLayout == 0) {
                    cachedLayoutA1 = LayoutA1(curInfo.curCalQSize, headDim, numHeads * headDim);
                    cachedLayoutB3 = LayoutB3(curInfo.curCalQSize, headDim, numHeads * headDim);
                } else {
                    cachedLayoutA1 = LayoutA1(curInfo.curCalQSize, headDim);
                    cachedLayoutB3 = LayoutB3(curInfo.curCalQSize, headDim);
                }
                const GemmCoord cachedShape{curInfo.curCalQSize, headDim, headDim};
                blockMmad1.PreloadA(gQ[curInfo.qOffset], cachedLayoutA1, cachedShape, 0);
                blockMmad1.PreloadA(gDout[curInfo.qOffset], cachedLayoutA1, cachedShape, 1);
                uint64_t kvBlockOffset = 0;
                uint64_t beginKVOffset = curInfo.kvOffset;
                uint32_t curKvBlockNum = (curInfo.kvSeqlen + blockShapeY - 1) / blockShapeY;
                const uint32_t kvBlockLimit = GetKvBlockLimit(params, curInfo, curKvBlockNum, tilingData);
                for (uint32_t idx = 0; idx < kvBlockLimit; idx++) {
                    // BlcokSpaseMask shape : [batch, numhead, CeilDiv(maxQSeqlen, blockShapeX), CeilDiv(maxKvSeqlen, blockShapeY)]
                    uint64_t maskOffset = curInfo.curBatchIdx * batchBlocks + curInfo.curHeadIdx * headBlocks +
                                          curInfo.curQBlcokIdx * maskKvBlockNum + idx;
                    if (ComputeBlockEnabled(params, maskOffset)) {
                        vec2CubeFlag++;
                        uint64_t kvBlockBasicOffset = 0;
                        uint32_t kvBlockSize =
                            (idx != curKvBlockNum - 1) ? blockShapeY : curInfo.kvSeqlen - blockShapeY * idx;
                        uint32_t kvLoop = (kvBlockSize + basicKVBlockSize - 1) / basicKVBlockSize;
                        for (uint32_t loop = 0; loop < kvLoop; loop++) {
                            curInfo.curCalKVSize = (loop != kvLoop - 1) ? basicKVBlockSize : kvBlockSize - basicKVBlockSize * loop;
                            if (inputLayout == 0) {
                                curInfo.kvOffset = beginKVOffset + (kvBlockOffset + kvBlockBasicOffset) * kvHeads * headDim;
                            } else {
                                curInfo.kvOffset = beginKVOffset + (kvBlockOffset + kvBlockBasicOffset) * headDim;
                            }
                            curInfo.sOffset = gSOffset + WORKSPACE_BLOCK_SIZE * pingpongFlag;
                            LayoutA1 layoutA1;
                            LayoutB1 layoutB1;
                            LayoutC1 layoutC1(curInfo.curCalQSize, curInfo.curCalKVSize);
                            if (inputLayout == 0) {
                                layoutA1 = LayoutA1(curInfo.curCalQSize, headDim, numHeads * headDim);
                                layoutB1 = LayoutB1(headDim, curInfo.curCalKVSize, kvHeads * headDim);
                            } else {
                                layoutA1 = LayoutA1(curInfo.curCalQSize, headDim);
                                layoutB1 = LayoutB1(headDim, curInfo.curCalKVSize);
                            }
                            GemmCoord actualShape1{curInfo.curCalQSize, curInfo.curCalKVSize, headDim};
                            blockMmad1.WithCachedA(0,
                                gK[curInfo.kvOffset],
                                gS[curInfo.sOffset],
                                layoutA1,
                                layoutB1,
                                layoutC1,
                                actualShape1,
                                mmadFlag);

                            blockMmad1.WithCachedA(1,
                                gV[curInfo.kvOffset],
                                gDp[curInfo.sOffset],
                                layoutA1,
                                layoutB1,
                                layoutC1,
                                actualShape1,
                                mmadFlag);
                            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2VEC);
                            if (count > 0) {
                                AscendC::WaitEvent(VEC2CUBE);

                                LayoutA2 layoutA2(preTaskInfo.curCalQSize, preTaskInfo.curCalKVSize);
                                LayoutB2 layoutB2;
                                LayoutC2 layoutC2;

                                LayoutA3 layoutA3(preTaskInfo.curCalKVSize, preTaskInfo.curCalQSize);
                                LayoutB3 layoutB3;
                                LayoutC3 layoutC3;
                                if (inputLayout == 0) {
                                    layoutB2 = LayoutB2(preTaskInfo.curCalKVSize, headDim, kvHeads * headDim);
                                    layoutC2 = LayoutC2(preTaskInfo.curCalQSize, headDim, numHeads * headDim);
                                    layoutB3 = LayoutB3(preTaskInfo.curCalQSize, headDim, numHeads * headDim);
                                    layoutC3 = LayoutC3(preTaskInfo.curCalKVSize, headDim, kvHeads * headDim);
                                } else {
                                    layoutB2 = LayoutB2(preTaskInfo.curCalKVSize, headDim);
                                    layoutC2 = LayoutC2(preTaskInfo.curCalQSize, headDim);
                                    layoutB3 = LayoutB3(preTaskInfo.curCalQSize, headDim);
                                    layoutC3 = LayoutC3(preTaskInfo.curCalKVSize, headDim);
                                }
                                GemmCoord actualShape2{preTaskInfo.curCalQSize, headDim, preTaskInfo.curCalKVSize};
                                GemmCoord actualShape3{preTaskInfo.curCalKVSize, headDim, preTaskInfo.curCalQSize};

                                blockMmad2(gDs[preTaskInfo.sOffset],
                                    gK[preTaskInfo.kvOffset],
                                    gDq[preTaskInfo.qOffset],
                                    layoutA2,
                                    layoutB2,
                                    layoutC2,
                                    actualShape2,
                                    mmadFlag);

                                blockMmad3.WithCachedB(0,
                                    gP[preTaskInfo.sOffset],
                                    gDv[preTaskInfo.kvOffset],
                                    layoutA3,
                                    layoutB3,
                                    layoutC3,
                                    actualShape3,
                                    mmadFlag);

                                blockMmad3.WithCachedB(1,
                                    gDs[preTaskInfo.sOffset],
                                    gDk[preTaskInfo.kvOffset],
                                    layoutA3,
                                    layoutB3,
                                    layoutC3,
                                    actualShape3,
                                    mmadFlag);
                            }
                            if (count == 0 || preTaskInfo.qOffset != curInfo.qOffset) {
                                const GemmCoord cachedShapeB{headDim, headDim, curInfo.curCalQSize};
                                blockMmad3.PreloadB(gDout[curInfo.qOffset], cachedLayoutB3, cachedShapeB, 0);
                                blockMmad3.PreloadB(gQ[curInfo.qOffset], cachedLayoutB3, cachedShapeB, 1);
                            }
                            preTaskInfo = curInfo;
                            preTaskInfo.sOffset = curInfo.sOffset * 2; // float32偏移转成bf16/half偏移
                            pingpongFlag = 1 - pingpongFlag;
                            count++;
                            kvBlockBasicOffset += basicKVBlockSize;
                        }
                    }
                    kvBlockOffset += blockShapeY;
                }
                if (i != taskLength - 1) {
                    updateNextTaskInfo(gActualQseqlen, gActualKvseqlen, numHeads, kvHeads, groupSize, headDim,
                        blockShapeX, basicQBlockSize, inputLayout, taskInfo[i % 2], taskInfo[(i + 1) % 2]);
                }
            }
            if (vec2CubeFlag != 0) {
                AscendC::WaitEvent(VEC2CUBE);
                LayoutA2 layoutA2(preTaskInfo.curCalQSize, preTaskInfo.curCalKVSize);
                LayoutB2 layoutB2;
                LayoutC2 layoutC2;

                LayoutA3 layoutA3(preTaskInfo.curCalKVSize, preTaskInfo.curCalQSize);
                LayoutB3 layoutB3;
                LayoutC3 layoutC3;
                if (inputLayout == 0) {
                    layoutB2 = LayoutB2(preTaskInfo.curCalKVSize, headDim, kvHeads * headDim);
                    layoutC2 = LayoutC2(preTaskInfo.curCalQSize, headDim, numHeads * headDim);
                    layoutB3 = LayoutB3(preTaskInfo.curCalQSize, headDim, numHeads * headDim);
                    layoutC3 = LayoutC3(preTaskInfo.curCalKVSize, headDim, kvHeads * headDim);
                } else {
                    layoutB2 = LayoutB2(preTaskInfo.curCalKVSize, headDim);
                    layoutC2 = LayoutC2(preTaskInfo.curCalQSize, headDim);
                    layoutB3 = LayoutB3(preTaskInfo.curCalQSize, headDim);
                    layoutC3 = LayoutC3(preTaskInfo.curCalKVSize, headDim);
                }
                GemmCoord actualShape2{preTaskInfo.curCalQSize, headDim, preTaskInfo.curCalKVSize};
                GemmCoord actualShape3{preTaskInfo.curCalKVSize, headDim, preTaskInfo.curCalQSize};

                blockMmad2(gDs[preTaskInfo.sOffset],
                    gK[preTaskInfo.kvOffset],
                    gDq[preTaskInfo.qOffset],
                    layoutA2,
                    layoutB2,
                    layoutC2,
                    actualShape2,
                    mmadFlag);
                blockMmad3.WithCachedB(0,
                    gP[preTaskInfo.sOffset],
                    gDv[preTaskInfo.kvOffset],
                    layoutA3,
                    layoutB3,
                    layoutC3,
                    actualShape3,
                    mmadFlag);
                blockMmad3.WithCachedB(1,
                    gDs[preTaskInfo.sOffset],
                    gDk[preTaskInfo.kvOffset],
                    layoutA3,
                    layoutB3,
                    layoutC3,
                    actualShape3,
                    mmadFlag);
            }
            WaitFlag();
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(CUBE2POST);
        }

  template <>
        __aicore__ inline
        void operator()<AscendC::AIV>(Params const &params)
        {
            __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);
            // InitOutput obtains its temporary UB stack and event IDs from
            // the active TPipe.  Keep this pre-stage pipe alive through the
            // mixed AIC/AIV barrier, exactly as FlashAttentionScoreGradPre.
            AscendC::TPipe pipePre;
            // AIVs clear disjoint FP32 gradient-workspace slices in VecPre.
            // Every AIC path reaches the matching SyncAll below before it can
            // start accumulation.
            VecPre(params);
            ClearGroupedPadding(params, tilingData);
            PackGroupedTndInputs(params, tilingData);
            AscendC::SyncAll<false>();
            pipePre.Destroy();

            // simply softmax
            if (UseWideSchedule(tilingData)) {
                ProcessWideAiv(params, tilingData);
            } else if (UseGroupedSchedule(tilingData)) {
                ProcessGroupedAiv(params, tilingData);
            } else {
                VecOp(params);
            }

            // Each AIC publishes completion to its two paired AIVs.  The AIV
            // barrier turns those pair-local notifications into a global
            // gradient-workspace completion barrier before output finalize.
            AscendC::WaitEvent(CUBE2POST);
            AscendC::SyncAll<true>();
            VecPost(params);
        }

        __aicore__ inline
        void VecOp(Params const &params)
        {
            uint32_t vecCoreIdx = AscendC::GetBlockIdx(); // vecore 核数idx
            uint32_t coreIdx = vecCoreIdx / 2; // cube 核数idx
            uint32_t subBlockIdx = vecCoreIdx % 2;

            __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);
            uint32_t batch = tilingData->batch;
            uint32_t numHeads = tilingData->numHeads;
            uint32_t kvHeads = tilingData->kvHeads;
            uint32_t groupSize = numHeads / kvHeads;
            uint32_t inputLayout = tilingData->inputLayout;
            uint32_t blockShapeX = tilingData->blockShapeX;
            uint32_t blockShapeY = tilingData->blockShapeY;
            uint32_t headDim = tilingData->headDim;
            uint32_t maxQSeqlen = tilingData->maxQSeqlen;
            uint32_t maxKvSeqlen = tilingData->maxKvSeqlen;

            uint32_t basicQBlockSize = tilingData->basicQBlockSize;
            uint32_t basicKVBlockSize = tilingData->basicKVBlockSize;
            uint32_t taskNumPerCore = tilingData->taskNumPerCore;
            uint32_t tailTaskNum = tilingData->tailTaskNum;
            uint32_t taskLengthVec = tailTaskNum > coreIdx ? taskNumPerCore + 1 : taskNumPerCore;

            uint64_t sOutSize = tilingData->sOutSize;
            uint64_t dPOutSize = tilingData->dPOutSize;
            uint64_t dQOutSize = tilingData->dQOutSize;
            uint64_t dKOutSize = tilingData->dKOutSize;
            uint64_t dVOutSize = tilingData->dVOutSize;

            // Initialize global tensors
            AscendC::GlobalTensor<int64_t> gActualQseqlen;
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);

            TaskInfo taskInfoVec[2]; // 索引以及shape信息
            TaskInfo preTaskInfo;
            initTaskInfo(gActualQseqlen, gActualKvseqlen, tilingData, numHeads, kvHeads, groupSize, headDim,
                maxQSeqlen, maxKvSeqlen, blockShapeX, basicQBlockSize, inputLayout, coreIdx, taskInfoVec[0]);
            uint32_t maskQBlockNum = (maxQSeqlen + blockShapeX - 1) / blockShapeX;
            uint32_t maskKvBlockNum = (maxKvSeqlen + blockShapeY - 1) / blockShapeY;
            uint32_t batchBlocks = numHeads * maskQBlockNum * maskKvBlockNum;
            uint32_t headBlocks = maskQBlockNum * maskKvBlockNum;

            // uint32_t count = 0;
            uint32_t pingpongFlag = 0;
            uint64_t gSOffset = coreIdx * WORKSPACE_BLOCK_SIZE_DB;

            GM_ADDR sftmgGm = params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize + dVOutSize;
            SfmgParams SfmgParams(params.dout, params.out, params.actualQseqlen, sftmgGm, params.tiling);
            EpilogueFAGSfmg vecSftmg(SfmgParams);
            EpilogueFAGOp sStmOp;

            for (uint32_t i = 0; i < taskLengthVec; i++) {
                TaskInfo curInfo = taskInfoVec[i % 2];
                uint64_t beginKVOffset = curInfo.kvOffset;

                // idx, 计算的行数
                // 一个ai core里的aiv 0执行的行数：actualRow / 2 + actualRow % 2
                // 一个ai core里的aiv 1执行的行数：actualRow / 2
                // if (vecCoreIdx % 2 != 0) {
                //     vecSftmg(curInfo.qOffset / headDim + curInfo.curCalQSize / 2 + curInfo.curCalQSize % 2,
                //         curInfo.curCalQSize / 2);
                // } else {
                //     vecSftmg(curInfo.qOffset / headDim, curInfo.curCalQSize / 2 + curInfo.curCalQSize % 2);
                // }

                // Split the row-wise SoftmaxGradFront between the two AIVs.
                const uint32_t firstHalfRows = (curInfo.curCalQSize + 1) / 2;
                const uint32_t subRows = subBlockIdx == 0 ?
                    firstHalfRows : curInfo.curCalQSize - firstHalfRows;
                if (inputLayout == 0) {
                    vecSftmg(curInfo.qOffset / headDim +
                        subBlockIdx * firstHalfRows * numHeads, subRows);
                } else {
                    vecSftmg(curInfo.qOffset / headDim +
                        subBlockIdx * firstHalfRows, subRows);
                }

                uint32_t curKvBlockNum = (curInfo.kvSeqlen + blockShapeY - 1) / blockShapeY;
                const uint32_t kvBlockLimit = GetKvBlockLimit(params, curInfo, curKvBlockNum, tilingData);
                for (uint32_t idx = 0; idx < kvBlockLimit; idx++) {
                    // BlcokSpaseMask shape : [batch, numhead, CeilDiv(maxQSeqlen, blockShapeX), CeilDiv(maxKvSeqlen, blockShapeY)]
                    uint64_t maskOffset = curInfo.curBatchIdx * batchBlocks + curInfo.curHeadIdx * headBlocks +
                                          curInfo.curQBlcokIdx * maskKvBlockNum + idx;
                    if (ComputeBlockEnabled(params, maskOffset)) {
                        uint32_t kvBlockSize =
                            (idx != curKvBlockNum - 1) ? blockShapeY : curInfo.kvSeqlen - blockShapeY * idx;
                        uint32_t kvLoop = (kvBlockSize + basicKVBlockSize - 1) / basicKVBlockSize;
                        for (uint32_t loop = 0; loop < kvLoop; loop++) {
                            curInfo.curCalKVSize = (loop != kvLoop - 1) ? basicKVBlockSize : kvBlockSize - basicKVBlockSize * loop;
                            curInfo.sOffset = gSOffset + WORKSPACE_BLOCK_SIZE * pingpongFlag;

                            uint64_t actualRow = curInfo.curCalQSize;
                            uint64_t actualCol = curInfo.curCalKVSize;
                            uint64_t processNums = curInfo.curCalQSize * curInfo.curCalKVSize;
                            uint64_t curCoreBatch = curInfo.curBatchIdx;
                            uint64_t curCoreN1Idx = curInfo.curHeadIdx; // 当前q的N1的idx
                            uint64_t curCoreS1Idx = curInfo.curQSeqIdx; // 当前q的s1的idx
                            uint64_t curT1Idx = 0;
                            uint64_t sOutSize = tilingData->sOutSize;
                            uint64_t dPOutSize = tilingData->dPOutSize;
                            uint64_t dQOutSize = tilingData->dQOutSize;
                            uint64_t dKOutSize = tilingData->dKOutSize;
                            uint64_t dVOutSize = tilingData->dVOutSize;

                            const uint64_t firstHalfRows = (actualRow + 1) / 2;
                            uint64_t executeRow = subBlockIdx == 0 ?
                                firstHalfRows : actualRow - firstHalfRows;
                            uint64_t coreOffset = subBlockIdx * firstHalfRows * actualCol;
                            uint64_t curVecCoreS1Idx = curCoreS1Idx + subBlockIdx * firstHalfRows;
                            uint64_t vector16Soffset =
                                (curInfo.sOffset * 2 + WORKSPACE_P16_OFFSET_ELEMENT + coreOffset) *
                                sizeof(ElementInput);
                            uint64_t vector32Soffset = (curInfo.sOffset + coreOffset) * sizeof(float);

                            GM_ADDR s = params.workspace + vector32Soffset;
                            GM_ADDR softmaxLse = params.softmaxLse;
                            GM_ADDR dp = params.workspace + sOutSize + vector32Soffset;
                            GM_ADDR actualSeqQlen = params.actualQseqlen;
                            GM_ADDR actualSeqKvlen = params.actualKvseqlen;
                            GM_ADDR sftmgGm = params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize + dVOutSize;
                            GM_ADDR pWorkspace = params.workspace + vector16Soffset; // 连续
                            GM_ADDR dsWorkspace = params.workspace + sOutSize + vector16Soffset; // 连续
                            GM_ADDR tiling = params.tiling;

                            AscendC::WaitEvent(CUBE2VEC);

                            SfmParams sfmParams(s,
                                softmaxLse,
                                dp,
                                actualSeqQlen,
                                actualSeqKvlen,
                                sftmgGm,
                                pWorkspace,
                                dsWorkspace,
                                tiling,
                                executeRow,
                                actualCol,
                                executeRow * actualCol,
                                curCoreBatch,
                                curCoreN1Idx,
                                curVecCoreS1Idx,
                                curT1Idx);
                            sStmOp(sfmParams);
                            // Mode 2 aggregates the completion events from the
                            // two AIV sub-cores paired with this AIC.  Do not
                            // use a global AIV barrier here: cores may own a
                            // different number of tasks.
                            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(VEC2CUBE);

                            preTaskInfo = curInfo;
                            pingpongFlag = 1 - pingpongFlag;
                            preTaskInfo.sOffset = curInfo.sOffset * 2; // float32偏移转成bf16/half偏移
                        }
                    }
                }

                if (i != taskLengthVec - 1) {
                    updateNextTaskInfo(gActualQseqlen, gActualKvseqlen, numHeads, kvHeads, groupSize, headDim,
                        blockShapeX, basicQBlockSize, inputLayout, taskInfoVec[i % 2], taskInfoVec[(i + 1) % 2]);
                }
            }
        }

        __aicore__ inline
        void VecPost(Params const &params)
        {
            __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);

            uint64_t sOutSize = tilingData->sOutSize;
            uint64_t dPOutSize = tilingData->dPOutSize;
            uint64_t dQOutSize = tilingData->dQOutSize;
            uint64_t dKOutSize = tilingData->dKOutSize;
            uint64_t dVOutSize = tilingData->dVOutSize;

            GM_ADDR gDqGm = params.workspace + sOutSize + dPOutSize;
            GM_ADDR gDkGm = params.workspace + sOutSize + dPOutSize + dQOutSize;
            GM_ADDR gDvGm = params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize;
            GM_ADDR actualSeqQlen = params.actualQseqlen;
            GM_ADDR actualSeqKvlen = params.actualKvseqlen;

            PostParams postParams(params.dq, params.dk, params.dv, gDqGm, gDkGm, gDvGm, params.tiling, actualSeqQlen, actualSeqKvlen);
            EpilogueFAGPost vecPost(postParams);
            vecPost();
        }

        __aicore__ inline
        void VecPre(Params const &params)
        {
            __gm__ BlockSparseAttentionGradTilingData *tilingData = reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tiling);

            uint64_t sOutSize = tilingData->sOutSize;
            uint64_t dPOutSize = tilingData->dPOutSize;
            uint64_t dQOutSize = tilingData->dQOutSize;
            uint64_t dKOutSize = tilingData->dKOutSize;
            uint64_t dVOutSize = tilingData->dVOutSize;


            GM_ADDR gDqWrkGm = params.workspace + sOutSize + dPOutSize;
            GM_ADDR gDkWrkGm = params.workspace + sOutSize + dPOutSize + dQOutSize;
            GM_ADDR gDvWrkGm = params.workspace + sOutSize + dPOutSize + dQOutSize + dKOutSize;

            PreParams preParms(gDqWrkGm, gDkWrkGm, gDvWrkGm, params.tiling);
            EpilogueFAGPre vecPre(preParms);
            vecPre();
        }

    private:
        NpuArch::Arch::Resource<ArchTag> resource;
    };

} // namespace BSA

#endif // BLOCK_SPARSE_ATTENTION_GRAD_KERNEL_H
