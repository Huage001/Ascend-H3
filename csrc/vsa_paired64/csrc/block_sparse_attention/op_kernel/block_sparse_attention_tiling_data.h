// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Licensed under the CANN Open Software License Agreement Version 2.0.

#ifndef ASCEND_KERNEL_BLOCK_SPARSE_ATTENTION_TILING_DATA_H
#define ASCEND_KERNEL_BLOCK_SPARSE_ATTENTION_TILING_DATA_H

#include <cstdint>

// Standalone Ascend-C builds do not run the op-plugin tiling-data generator.
// Keep this POD in the same field order as BlockSparseAttentionTilingData in
// ops-transformer/attention/block_sparse_attention/op_host.
struct BsaMask2IdxTiling {
    uint32_t xBlockNumAligned;
    uint32_t yBlockNumAligned;
    uint32_t avgRowPerSubCore;
    uint32_t preActiveSubCoreNum;
};

struct BsaBaseTiling {
    uint32_t qBaseTile;
    uint32_t kvBaseTile;
};

struct BsaMmPhaseL1Tiling {
    uint32_t mm1L1TileM;
    uint32_t mm1L1TileN;
    uint32_t mm1L1TileKLeft;
    uint32_t mm1L1TileKRight;
    uint32_t mm2L1TileM;
    uint32_t mm2L1TileN;
    uint32_t mm2L1TileKLeft;
    uint32_t mm2L1TileKRight;
    uint32_t qL1BufNum;
    uint32_t kL1BufNum;
    uint32_t vL1BufNum;
    uint32_t pL1BufNum;
};

struct BlockSparseAttentionTilingData {
    uint32_t batch;
    uint32_t numHeads;
    uint32_t kvHeads;
    uint32_t embeddingSize;
    uint32_t blockSize;
    uint32_t maxNumBlocksPerBatch;
    uint32_t firstBatchTaskNum;
    uint32_t totalTaskNum;
    uint32_t maskType;
    float scaleValue;
    uint32_t totalQBlocks;
    uint32_t firstQBlockNum;
    uint64_t blockShapeX;
    uint64_t blockShapeY;
    uint32_t maxKvBlockNum;
    uint32_t maxQBlockNum;
    uint32_t avgRowNumPerSubCore;
    uint32_t preActivateSubCoreNum;
    uint32_t queryLayout;
    uint32_t kvCacheLayout;
    uint32_t maxQSeqlen;
    uint32_t maxKvSeqlen;
    uint32_t useUniformQSeqlen;
    uint32_t useUniformKvSeqlen;
    uint64_t tilingKey;
    uint64_t selectNumIdxSize;
    uint64_t selectIdxSize;
    uint64_t mm1OutSize;
    uint64_t smOnlineOutSize;
    uint64_t mm2OutSize;
    uint64_t updateSize;
    uint64_t workSpaceSize;
    BsaMask2IdxTiling BsaMask2IdxTileInfo;
    BsaBaseTiling BsaBaseTileInfo;
    BsaMmPhaseL1Tiling BsaMmPhaseL1TileInfo;
};

#endif
