/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "miowtion128pipe3_attention_tiling_data.h"
#include "miowtion128pipe3_attention_tilingkey.h"
#include "miowtion128pipe3_attention_kernel_interface.cpp"

extern "C" __global__ __aicore__ void miowtion128pipe3_attention(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
    __gm__ uint8_t* blockSparseMask, __gm__ uint8_t* mask, __gm__ uint8_t* blockShape,
    __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKv, __gm__ uint8_t* blockTable,
    __gm__ uint8_t* qDequantScale, __gm__ uint8_t* kDequantScale, __gm__ uint8_t* vDequantScale,
    __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
    __gm__ uint8_t* tiling)
{
    __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    __gm__ BlockSparseAttentionTilingData *tilingDataPtr =
        reinterpret_cast<__gm__ BlockSparseAttentionTilingData *>(tiling);
    uint64_t tilingKey = tilingDataPtr->tilingKey;

#if (__CCE_AICORE__ == 220)
    if (tilingDataPtr->queryLayout == 0 && tilingDataPtr->blockSize == 128) {
        BlockSparse::BlockSparseAttentionInfer<bfloat16_t, float, Epilogue::LseMode::OUT_ONLY, 0, 0, 128>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    }
    if (tilingDataPtr->queryLayout == 1 && tilingDataPtr->blockSize == 128) {
        BlockSparse::BlockSparseAttentionInfer<bfloat16_t, float, Epilogue::LseMode::OUT_ONLY, 1, 1, 128>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut,
            actualSeqLengths, actualSeqLengthsKv, blockShape, user, softmaxLse, tiling);
    }
#endif
}
