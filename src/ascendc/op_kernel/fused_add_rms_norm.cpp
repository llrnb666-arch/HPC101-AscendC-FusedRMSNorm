#include "kernel_operator.h"
 
namespace {
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ALIGN_NUM = 16;
constexpr int32_t TILE_MAX_ELEMS = 4096;
}
 
class KernelFusedAddRmsNorm {
public:
    __aicore__ inline KernelFusedAddRmsNorm() {}
 
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR residual, GM_ADDR weight,
                                GM_ADDR y, GM_ADDR residual_out,
                                FusedAddRmsNormTilingData& tiling, AscendC::TPipe* pipeIn) {
        this->pipe = pipeIn;
        this->blockIdx = AscendC::GetBlockIdx();
        this->batchSize = tiling.batchSize;
        this->hiddenSize = tiling.hiddenSize;
        this->alignedHidden = tiling.alignedHidden;
        this->alignNum = tiling.alignNum;
        this->eps = tiling.eps;
        this->useNR = true;
        this->tileElems = this->alignedHidden;
        if (this->tileElems > TILE_MAX_ELEMS) this->tileElems = TILE_MAX_ELEMS;
        if (this->tileElems < this->alignNum) this->tileElems = this->alignNum;
        int32_t totalRows = this->batchSize;
        int32_t blockNum = static_cast<int32_t>(AscendC::GetBlockNum());
        int32_t rowsPerBlock = (totalRows + blockNum - 1) / blockNum;
        this->startRow = static_cast<int64_t>(this->blockIdx) * rowsPerBlock;
        this->endRow = this->startRow + rowsPerBlock;
        if (this->endRow > totalRows) this->endRow = totalRows;
        uint64_t totalElems = static_cast<uint64_t>(this->batchSize) * static_cast<uint64_t>(this->hiddenSize);
        if (totalElems == 0) totalElems = 1;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), totalElems);
        residualGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(residual), totalElems);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y), totalElems);
        residualOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(residual_out), totalElems);
        uint64_t weightElems = static_cast<uint64_t>(this->hiddenSize > 0 ? this->hiddenSize : 1);
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weight), weightElems);
        uint32_t tileBytesFp16 = static_cast<uint32_t>(this->tileElems) * sizeof(half);
        uint32_t tileBytesFp32 = static_cast<uint32_t>(this->tileElems) * sizeof(float);
        pipe->InitBuffer(inQueX, BUFFER_NUM, tileBytesFp16);
        pipe->InitBuffer(inQueRes, BUFFER_NUM, tileBytesFp16);
        pipe->InitBuffer(outQueY, BUFFER_NUM, tileBytesFp16);
        pipe->InitBuffer(outQueResOut, BUFFER_NUM, tileBytesFp16);
        pipe->InitBuffer(weightHalfBuf, tileBytesFp16);
        pipe->InitBuffer(weightFp32Buf, tileBytesFp32);
        pipe->InitBuffer(resoFp32Buf, tileBytesFp32);
        pipe->InitBuffer(sqBuf, tileBytesFp32);
        pipe->InitBuffer(meanBuf, tileBytesFp32);
        pipe->InitBuffer(nrBuf, tileBytesFp32);
        pipe->InitBuffer(resOutHalfBuf0, tileBytesFp16);
        pipe->InitBuffer(resOutHalfBuf1, tileBytesFp16);
        pipe->InitBuffer(yHalfBuf0, tileBytesFp16);
        pipe->InitBuffer(yHalfBuf1, tileBytesFp16);
        pipe->InitBuffer(scalarBuf, 32);
        pipe->InitBuffer(reduceTmpBuf, 32);
    }
 
    __aicore__ inline void Process() {
        if (this->startRow >= this->endRow) return;
        if (this->hiddenSize <= 0) return;
        if (this->alignedHidden <= this->tileElems) { ProcessWholeRows(); }
        else { ProcessChunkedRows(); }
    }
 
private:
    __aicore__ inline void ProcessWholeRows() {
        AscendC::LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();
        AscendC::LocalTensor<float> resoFp32 = resoFp32Buf.Get<float>();
        AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
        AscendC::LocalTensor<float> mean = meanBuf.Get<float>();
        AscendC::LocalTensor<float> nr = nrBuf.Get<float>();
        AscendC::LocalTensor<float> scalar = scalarBuf.Get<float>();
        AscendC::LocalTensor<half> resOutHalf0 = resOutHalfBuf0.Get<half>();
        AscendC::LocalTensor<half> resOutHalf1 = resOutHalfBuf1.Get<half>();
        AscendC::LocalTensor<half> yHalf0 = yHalfBuf0.Get<half>();
        AscendC::LocalTensor<half> yHalf1 = yHalfBuf1.Get<half>();
        LoadWeightRow(weightFp32, 0, this->hiddenSize, this->alignedHidden);
        const int32_t H = this->hiddenSize;
        const int32_t alignH = this->alignedHidden;
        const float invH = 1.0f / static_cast<float>(H);
        // Prologue: issue first row load (MTE2 prefetch)
        {
            uint64_t base0 = static_cast<uint64_t>(this->startRow) * static_cast<uint64_t>(H);
            AscendC::LocalTensor<half> xPrefetch = inQueX.AllocTensor<half>();
            AscendC::LocalTensor<half> resPrefetch = inQueRes.AllocTensor<half>();
            CopyInRow(xPrefetch, xGm, base0);
            CopyInRow(resPrefetch, residualGm, base0);
            inQueX.EnQue(xPrefetch);
            inQueRes.EnQue(resPrefetch);
        }
        uint64_t base = static_cast<uint64_t>(this->startRow) * static_cast<uint64_t>(H);
        for (int64_t row = this->startRow; row < this->endRow; ++row) {
            int32_t eid = static_cast<int32_t>(row & 1);
            // Prefetch next row (MTE2 overlaps with current V compute)
            if (row + 1 < this->endRow) {
                uint64_t nextBase = base + static_cast<uint64_t>(H);
                AscendC::LocalTensor<half> xNext = inQueX.AllocTensor<half>();
                AscendC::LocalTensor<half> resNext = inQueRes.AllocTensor<half>();
                CopyInRow(xNext, xGm, nextBase);
                CopyInRow(resNext, residualGm, nextBase);
                inQueX.EnQue(xNext);
                inQueRes.EnQue(resNext);
            }
            // Wait for MTE3 from 2 iterations ago before reusing output buffers
            if (row >= this->startRow + 2) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eid);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eid);
            }
            // DeQue current row (waits for MTE2 to complete)
            AscendC::LocalTensor<half> xLocal = inQueX.DeQue<half>();
            AscendC::LocalTensor<half> resLocal = inQueRes.DeQue<half>();
            // residual_out = x + residual (FP16 add directly to output buffer)
            if (eid == 0) AscendC::Add(resOutHalf0, xLocal, resLocal, alignH);
            else AscendC::Add(resOutHalf1, xLocal, resLocal, alignH);
            // Cast to FP32 for RMSNorm
            if (eid == 0) AscendC::Cast(resoFp32, resOutHalf0, AscendC::RoundMode::CAST_NONE, alignH);
            else AscendC::Cast(resoFp32, resOutHalf1, AscendC::RoundMode::CAST_NONE, alignH);
            // --- RMSNorm: reduce, rsqrt+NR ---
            AscendC::Mul(sq, resoFp32, resoFp32, alignH);
            ReduceNormal(scalar, sq, H);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            float sumSq = scalar.GetValue(0);
            float meanPlusEps = sumSq * invH + this->eps;
            AscendC::Duplicate<float>(mean, meanPlusEps, alignH);
            AscendC::Rsqrt<float>(sq, mean, alignH);
            if (this->useNR) {
                AscendC::Mul(nr, sq, sq, alignH);
                AscendC::Mul(nr, mean, nr, alignH);
                AscendC::Muls(nr, nr, -0.5f, alignH);
                AscendC::Adds(nr, nr, 1.5f, alignH);
                AscendC::Mul(sq, sq, nr, alignH);
            }
            // Precompute rstd*weight
            AscendC::Mul(sq, sq, weightFp32, alignH);
            // y = resoFp32 * (rstd * weight)
            AscendC::Mul(resoFp32, resoFp32, sq, alignH);
            // Cast y to FP16
            if (eid == 0) AscendC::Cast(yHalf0, resoFp32, AscendC::RoundMode::CAST_NONE, alignH);
            else AscendC::Cast(yHalf1, resoFp32, AscendC::RoundMode::CAST_NONE, alignH);
            // --- Write both outputs (single V_MTE3 event, MTE3 copies both) ---
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eid);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eid);
            if (eid == 0) {
                CopyOutRow(resOutHalf0, residualOutGm, base);
                CopyOutRow(yHalf0, yGm, base);
            } else {
                CopyOutRow(resOutHalf1, residualOutGm, base);
                CopyOutRow(yHalf1, yGm, base);
            }
            inQueX.FreeTensor(xLocal);
            inQueRes.FreeTensor(resLocal);
            base += static_cast<uint64_t>(H);
        }
        // Drain: wait for last 2 MTE3 operations
        int64_t nRows = this->endRow - this->startRow;
        if (nRows >= 1) {
            int32_t eidLast = static_cast<int32_t>((this->endRow - 1) & 1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eidLast);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eidLast);
        }
        if (nRows >= 2) {
            int32_t eidPrev = static_cast<int32_t>((this->endRow - 2) & 1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eidPrev);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eidPrev);
        }
    }
    __aicore__ inline void ProcessChunkedRows() {
        AscendC::LocalTensor<float> weightFp32 = weightFp32Buf.Get<float>();
        AscendC::LocalTensor<float> resoFp32 = resoFp32Buf.Get<float>();
        AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
        AscendC::LocalTensor<float> mean = meanBuf.Get<float>();
        AscendC::LocalTensor<float> nr = nrBuf.Get<float>();
        AscendC::LocalTensor<float> scalar = scalarBuf.Get<float>();
        const int32_t H = this->hiddenSize;
        const int32_t chunkElems = this->tileElems;
        const float invH = 1.0f / static_cast<float>(H);
        for (int64_t row = this->startRow; row < this->endRow; ++row) {
            uint64_t base = static_cast<uint64_t>(row) * static_cast<uint64_t>(H);
            float sumSq = 0.0f;
            int32_t off = 0;
            while (off < H) {
                int32_t n = (H - off > chunkElems) ? chunkElems : (H - off);
                int32_t nAlign = (n + this->alignNum - 1) / this->alignNum * this->alignNum;
                AscendC::LocalTensor<half> xLocal = inQueX.AllocTensor<half>();
                AscendC::LocalTensor<half> resLocal = inQueRes.AllocTensor<half>();
                CopyInChunk(xLocal, xGm, base + off, n, nAlign);
                CopyInChunk(resLocal, residualGm, base + off, n, nAlign);
                inQueX.EnQue(xLocal);
                inQueRes.EnQue(resLocal);
                xLocal = inQueX.DeQue<half>();
                resLocal = inQueRes.DeQue<half>();
                AscendC::Cast(resoFp32, xLocal, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::Cast(sq, resLocal, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(resoFp32, resoFp32, sq, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(sq, resoFp32, resoFp32, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                ReduceNormal(scalar, sq, n);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                sumSq += scalar.GetValue(0);
                inQueX.FreeTensor(xLocal);
                inQueRes.FreeTensor(resLocal);
                off += n;
            }
            float meanPlusEps = sumSq * invH + this->eps;
            off = 0;
            while (off < H) {
                int32_t n = (H - off > chunkElems) ? chunkElems : (H - off);
                int32_t nAlign = (n + this->alignNum - 1) / this->alignNum * this->alignNum;
                AscendC::LocalTensor<half> xLocal = inQueX.AllocTensor<half>();
                AscendC::LocalTensor<half> resLocal = inQueRes.AllocTensor<half>();
                CopyInChunk(xLocal, xGm, base + off, n, nAlign);
                CopyInChunk(resLocal, residualGm, base + off, n, nAlign);
                inQueX.EnQue(xLocal);
                inQueRes.EnQue(resLocal);
                xLocal = inQueX.DeQue<half>();
                resLocal = inQueRes.DeQue<half>();
                AscendC::Cast(resoFp32, xLocal, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::Cast(sq, resLocal, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(resoFp32, resoFp32, sq, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                inQueX.FreeTensor(xLocal);
                inQueRes.FreeTensor(resLocal);
                AscendC::LocalTensor<half> resOutLocal = outQueResOut.AllocTensor<half>();
                AscendC::Cast(resOutLocal, resoFp32, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                outQueResOut.EnQue(resOutLocal);
                resOutLocal = outQueResOut.DeQue<half>();
                CopyOutChunk(resOutLocal, residualOutGm, base + off, n);
                outQueResOut.FreeTensor(resOutLocal);
                AscendC::Duplicate<float>(mean, meanPlusEps, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Rsqrt<float>(sq, mean, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                if (this->useNR) {
                    AscendC::Mul(nr, sq, sq, nAlign);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(nr, mean, nr, nAlign);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(nr, nr, -0.5f, nAlign);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Adds(nr, nr, 1.5f, nAlign);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(sq, sq, nr, nAlign);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Mul(resoFp32, resoFp32, sq, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                LoadWeightRow(weightFp32, off, n, nAlign);
                AscendC::Mul(resoFp32, resoFp32, weightFp32, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::LocalTensor<half> yLocal = outQueY.AllocTensor<half>();
                AscendC::Cast(yLocal, resoFp32, AscendC::RoundMode::CAST_NONE, nAlign);
                AscendC::PipeBarrier<PIPE_V>();
                outQueY.EnQue(yLocal);
                yLocal = outQueY.DeQue<half>();
                CopyOutChunk(yLocal, yGm, base + off, n);
                outQueY.FreeTensor(yLocal);
                off += n;
            }
        }
    }
 
    __aicore__ inline void LoadWeightRow(AscendC::LocalTensor<float>& wFp32, int32_t off, int32_t realN, int32_t nAlign) {
        AscendC::LocalTensor<half> wHalf = weightHalfBuf.Get<half>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1; copyParams.blockLen = static_cast<uint32_t>(realN * sizeof(half));
        copyParams.srcStride = 0; copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<half> padParams;
        padParams.isPad = (realN < nAlign); padParams.leftPadding = 0;
        padParams.rightPadding = static_cast<uint16_t>(nAlign - realN); padParams.paddingValue = 0;
        AscendC::DataCopyPad(wHalf, weightGm[static_cast<uint64_t>(off)], copyParams, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(wFp32, wHalf, AscendC::RoundMode::CAST_NONE, nAlign);
        AscendC::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CopyInRow(AscendC::LocalTensor<half>& dst, AscendC::GlobalTensor<half>& src, uint64_t off) {
        if (this->hiddenSize == this->alignedHidden) {
            AscendC::DataCopyParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint16_t>(this->hiddenSize * sizeof(half) / 32);
            params.srcStride = 0;
            params.dstStride = 0;
            AscendC::DataCopy(dst, src[off], params);
            return;
        }
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1; copyParams.blockLen = static_cast<uint32_t>(this->hiddenSize * sizeof(half));
        copyParams.srcStride = 0; copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<half> padParams;
        padParams.isPad = (this->hiddenSize < this->alignedHidden); padParams.leftPadding = 0;
        padParams.rightPadding = static_cast<uint16_t>(this->alignedHidden - this->hiddenSize); padParams.paddingValue = 0;
        AscendC::DataCopyPad(dst, src[off], copyParams, padParams);
    }
    __aicore__ inline void CopyInChunk(AscendC::LocalTensor<half>& dst, AscendC::GlobalTensor<half>& src, uint64_t off, int32_t n, int32_t nAlign) {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1; copyParams.blockLen = static_cast<uint32_t>(n * sizeof(half));
        copyParams.srcStride = 0; copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<half> padParams;
        padParams.isPad = (n < nAlign); padParams.leftPadding = 0;
        padParams.rightPadding = static_cast<uint16_t>(nAlign - n); padParams.paddingValue = 0;
        AscendC::DataCopyPad(dst, src[off], copyParams, padParams);
    }
    __aicore__ inline void CopyOutRow(AscendC::LocalTensor<half>& src, AscendC::GlobalTensor<half>& dst, uint64_t off) {
        if (this->hiddenSize == this->alignedHidden) {
            AscendC::DataCopyParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint16_t>(this->hiddenSize * sizeof(half) / 32);
            params.srcStride = 0;
            params.dstStride = 0;
            AscendC::DataCopy(dst[off], src, params);
            return;
        }
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1; copyParams.blockLen = static_cast<uint32_t>(this->hiddenSize * sizeof(half));
        copyParams.srcStride = 0; copyParams.dstStride = 0;
        AscendC::DataCopyPad(dst[off], src, copyParams);
    }
    __aicore__ inline void CopyOutChunk(AscendC::LocalTensor<half>& src, AscendC::GlobalTensor<half>& dst, uint64_t off, int32_t n) {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1; copyParams.blockLen = static_cast<uint32_t>(n * sizeof(half));
        copyParams.srcStride = 0; copyParams.dstStride = 0;
        AscendC::DataCopyPad(dst[off], src, copyParams);
    }
    __aicore__ inline void ReduceNormal(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src, const int totalElements) {
        constexpr int elemsPerBlock = 32 / sizeof(float);
        int currentLen = totalElements;
        AscendC::SetMaskCount();
        while (currentLen > (elemsPerBlock * 8)) {
            int blockCount = (currentLen + elemsPerBlock - 1) / elemsPerBlock;
            int repeat = (blockCount + 7) / 8;
            AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(currentLen);
            AscendC::BlockReduceSum<float, false>(src, src, repeat, AscendC::MASK_PLACEHOLDER, 1, 1, 8);
            currentLen = blockCount;
        }
        AscendC::SetVectorMask<float, AscendC::MaskMode::COUNTER>(currentLen);
        AscendC::WholeReduceSum<float, false>(dst, src, AscendC::MASK_PLACEHOLDER, 1, 1, 1, 8);
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }
private:
    AscendC::TPipe* pipe;
    int32_t blockIdx, batchSize, hiddenSize, alignedHidden, alignNum, tileElems;
    int64_t startRow, endRow;
    float eps;
    bool useNR;
    AscendC::GlobalTensor<half> xGm, residualGm, weightGm, yGm, residualOutGm;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueX, inQueRes;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueY, outQueResOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weightHalfBuf, weightFp32Buf, resoFp32Buf, sqBuf, meanBuf, nrBuf, scalarBuf, reduceTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resOutHalfBuf0, resOutHalfBuf1, yHalfBuf0, yHalfBuf1;
};
extern "C" __global__ __aicore__ void fused_add_rms_norm(GM_ADDR x, GM_ADDR residual, GM_ADDR weight, GM_ADDR y, GM_ADDR residual_out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    KernelFusedAddRmsNorm op;
    op.Init(x, residual, weight, y, residual_out, tilingData, &pipe);
    op.Process();
}
