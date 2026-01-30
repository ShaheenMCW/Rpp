/*
 * Standalone Resize Nearest Neighbor Implementation
 * Extracted from RPP (ROCm Performance Primitives) for benchmarking
 * Modified to handle single images instead of batches
 */

#include <cstdio>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <omp.h>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif

// Include half.hpp for Rpp16f support
#if __has_include(<half/half.hpp>)
    #include <half/half.hpp>
#else
    #include <half.hpp>
#endif
using halfhpp = half_float::half;

// ==================== Type Definitions ====================
typedef halfhpp Rpp16f;
typedef unsigned char       Rpp8u;
typedef signed char         Rpp8s;
typedef unsigned short      Rpp16u;
typedef short               Rpp16s;
typedef unsigned int        Rpp32u;
typedef int                 Rpp32s;
typedef unsigned long long  Rpp64u;
typedef long long           Rpp64s;
typedef float               Rpp32f;
typedef double              Rpp64f;
typedef void*               RppPtr_t;
typedef size_t              RppSize_t;

typedef enum
{
    RPP_SUCCESS = 0,
    RPP_ERROR = -1,
} RppStatus_t;

typedef RppStatus_t RppStatus;

typedef enum
{
    XYWH = 0,
    LTRB = 1,
} RpptRoiType;

typedef enum
{
    NCHW = 0,
    NHWC = 1,
} RpptLayout;

typedef struct
{
    Rpp32u nStride;
    Rpp32u cStride;
    Rpp32u hStride;
    Rpp32u wStride;
} RpptStrides;

typedef struct
{
    RppSize_t numDims;
    Rpp32u offsetInBytes;
    Rpp32u n, c, h, w;
    RpptStrides strides;
    RpptLayout layout;
} RpptDesc, *RpptDescPtr;

typedef struct
{
    Rpp32u x;
    Rpp32u y;
} RpptXY;

typedef struct
{
    RpptXY lt;
    RpptXY rb;
} RpptLTRBROI;

typedef struct
{
    RpptXY xy;
    Rpp32u roiWidth;
    Rpp32u roiHeight;
} RpptXYWHROI;

typedef struct
{
    RpptXYWHROI xywhROI;
    RpptLTRBROI ltrbROI;
} RpptROI;

typedef RpptROI *RpptROIPtr;

typedef struct
{
    Rpp32u width;
    Rpp32u height;
} RpptImagePatch;

typedef RpptImagePatch *RpptImagePatchPtr;

typedef struct
{
    RpptLayout layout;
    Rpp32u bufferMultiplier;
} RppLayoutParams;
// ==================== Handle ====================
namespace rpp {
    struct Handle {
        Handle(size_t nBatchSize = 1, unsigned int numThreads = 0)
            : batchSize(nBatchSize),
              threads(numThreads ? numThreads : omp_get_max_threads()) {}
        
        size_t GetBatchSize() const { return batchSize; }
        unsigned int GetNumThreads() const { return threads; }
        void SetBatchSize(size_t bSize) { batchSize = bSize; }
        
    private:
        size_t batchSize;
        unsigned int threads;
    };
}

// ==================== SIMD Constants ====================
const __m128 xmm_p0 = _mm_setzero_ps();
const __m128 xmm_p3 = _mm_set1_ps(3.0f);
const __m128 xmm_p4 = _mm_set1_ps(4.0f);
const __m128 xmm_pDstLocInit = _mm_setr_ps(0, 1, 2, 3);

const __m128i xmm_px0 = _mm_set1_epi32(0);
const __m128i xmm_char_maskR = _mm_setr_epi8(0, 3, 6, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
const __m128i xmm_char_maskG = _mm_setr_epi8(1, 4, 7, 10, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
const __m128i xmm_char_maskB = _mm_setr_epi8(2, 5, 8, 11, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
const __m128i xmm_pkd_mask = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0x80, 0x80, 0x80, 0x80);
const __m128i xmm_store4_pkd_pixels = _mm_setr_epi8(0, 1, 8, 2, 3, 9, 4, 5, 10, 6, 7, 11, 0x80, 0x80, 0x80, 0x80);

// ==================== Helper Functions ====================
inline void rpp_storeu_si32(void *__p, __m128i __b)
{
#if defined(__GNUC__) || defined(__clang__)
    struct __storeu_si32
    {
        int __v;
    } __attribute__((__packed__, __may_alias__));
    ((struct __storeu_si32*)__p)->__v = _mm_cvtsi128_si32(__b);
#else
    *((int*)__p) = _mm_cvtsi128_si32(__b);
#endif
}

inline void compute_roi_validation_host(RpptROIPtr roiPtrInput, RpptROIPtr roiPtr, RpptROIPtr roiPtrDefault, RpptRoiType roiType)
{
    if (roiPtrInput == NULL)
    {
        *roiPtr = *roiPtrDefault;
    }
    else
    {
        RpptROI roiImage;
        RpptROIPtr roiPtrImage = &roiImage;
        
        if (roiType == RpptRoiType::LTRB)
        {
            roiPtrImage->xywhROI.xy.x = roiPtrInput->ltrbROI.lt.x;
            roiPtrImage->xywhROI.xy.y = roiPtrInput->ltrbROI.lt.y;
            roiPtrImage->xywhROI.roiWidth = roiPtrInput->ltrbROI.rb.x - roiPtrInput->ltrbROI.lt.x + 1;
            roiPtrImage->xywhROI.roiHeight = roiPtrInput->ltrbROI.rb.y - roiPtrInput->ltrbROI.lt.y + 1;
        }
        else if (roiType == RpptRoiType::XYWH)
        {
            roiPtrImage = roiPtrInput;
        }
        
        // Boundary check
        roiPtr->xywhROI.xy.x = std::max(roiPtrDefault->xywhROI.xy.x, roiPtrImage->xywhROI.xy.x);
        roiPtr->xywhROI.xy.y = std::max(roiPtrDefault->xywhROI.xy.y, roiPtrImage->xywhROI.xy.y);
        roiPtr->xywhROI.roiWidth = std::min(roiPtrDefault->xywhROI.roiWidth - roiPtrImage->xywhROI.xy.x, roiPtrImage->xywhROI.roiWidth);
        roiPtr->xywhROI.roiHeight = std::min(roiPtrDefault->xywhROI.roiHeight - roiPtrImage->xywhROI.xy.y, roiPtrImage->xywhROI.roiHeight);
    }
}

inline void compute_dst_size_cap_host(RpptImagePatchPtr dstImgSize, RpptDescPtr dstDescPtr)
{
    dstImgSize->width = std::min(dstImgSize->width, dstDescPtr->w);
    dstImgSize->height = std::min(dstImgSize->height, dstDescPtr->h);
}

inline void compute_resize_nn_src_loc(Rpp32s dstLocation, Rpp32f scale, Rpp32u limit, Rpp32s &srcLoc, Rpp32f offset = 0, Rpp32u srcStride = 1)
{
    Rpp32f srcLocation = ((Rpp32f) dstLocation) * scale + offset;
    Rpp32s srcLocationFloor = std::floor(srcLocation);
    srcLoc = ((srcLocationFloor > (Rpp32s)limit) ? (Rpp32s)limit : srcLocationFloor) * srcStride;
}

inline void compute_resize_nn_src_loc_sse(__m128 &pDstLoc, __m128 &pScale, __m128 &pLimit, Rpp32s *srcLoc, __m128 pOffset = xmm_p0, bool hasRGBChannels = false)
{
    __m128 pLoc = _mm_fmadd_ps(pDstLoc, pScale, pOffset);
    pDstLoc = _mm_add_ps(pDstLoc, xmm_p4);
    __m128 pLocFloor = _mm_floor_ps(pLoc);
    pLocFloor = _mm_max_ps(_mm_min_ps(pLocFloor, pLimit), xmm_p0);
    if(hasRGBChannels)
        pLocFloor = _mm_mul_ps(pLocFloor, xmm_p3);
    __m128i pxLocFloor = _mm_cvtps_epi32(pLocFloor);
    _mm_storeu_si128((__m128i*) srcLoc, pxLocFloor);
}

// ==================== SIMD Load Functions ====================
inline void rpp_resize_nn_load_u8pkd3(Rpp8u *srcRowPtrsForInterp, Rpp32s *loc, __m128i &p)
{
    __m128i px[4];
    px[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[0]));
    px[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[1]));
    px[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[2]));
    px[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[3]));
    px[0] = _mm_unpacklo_epi64(_mm_unpacklo_epi32(px[0], px[1]), _mm_unpacklo_epi32(px[2], px[3]));
    p = _mm_shuffle_epi8(px[0], xmm_pkd_mask);
}

inline void rpp_resize_nn_load_u8pln1(Rpp8u *srcRowPtrsForInterp, Rpp32s *loc, __m128i &p)
{
    __m128i px[4];
    px[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[0]));
    px[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[1]));
    px[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[2]));
    px[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[3]));
    px[0] = _mm_unpacklo_epi8(px[0], px[2]);
    px[1] = _mm_unpacklo_epi8(px[1], px[3]);
    p = _mm_unpacklo_epi8(px[0], px[1]);
}

inline void rpp_resize_nn_load_i8pkd3(Rpp8s *srcRowPtrsForInterp, Rpp32s *loc, __m128i &p)
{
    __m128i px[4];
    px[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[0]));
    px[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[1]));
    px[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[2]));
    px[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[3]));
    px[0] = _mm_unpacklo_epi64(_mm_unpacklo_epi32(px[0], px[1]), _mm_unpacklo_epi32(px[2], px[3]));
    p = _mm_shuffle_epi8(px[0], xmm_pkd_mask);
}

inline void rpp_resize_nn_load_i8pln1(Rpp8s *srcRowPtrsForInterp, Rpp32s *loc, __m128i &p)
{
    __m128i px[4];
    px[0] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[0]));
    px[1] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[1]));
    px[2] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[2]));
    px[3] = _mm_loadu_si128((__m128i *)(srcRowPtrsForInterp + loc[3]));
    px[0] = _mm_unpacklo_epi8(px[0], px[2]);
    px[1] = _mm_unpacklo_epi8(px[1], px[3]);
    p = _mm_unpacklo_epi8(px[0], px[1]);
}

inline void rpp_resize_nn_load_f32pkd3_to_f32pln3(Rpp32f *srcRowPtrsForInterp, Rpp32s *loc, __m128 *p)
{
    p[0] = _mm_loadu_ps(srcRowPtrsForInterp + loc[0]);
    p[1] = _mm_loadu_ps(srcRowPtrsForInterp + loc[1]);
    p[2] = _mm_loadu_ps(srcRowPtrsForInterp + loc[2]);
    __m128 pTemp = _mm_loadu_ps(srcRowPtrsForInterp + loc[3]);
    _MM_TRANSPOSE4_PS(p[0], p[1], p[2], pTemp);
}

inline void rpp_resize_nn_load_f32pln1(Rpp32f *srcRowPtrsForInterp, Rpp32s *loc, __m128 &p)
{
    __m128 pTemp[4];
    pTemp[0] = _mm_loadu_ps(srcRowPtrsForInterp + loc[0]);
    pTemp[1] = _mm_loadu_ps(srcRowPtrsForInterp + loc[1]);
    pTemp[2] = _mm_loadu_ps(srcRowPtrsForInterp + loc[2]);
    pTemp[3] = _mm_loadu_ps(srcRowPtrsForInterp + loc[3]);
    pTemp[0] = _mm_unpacklo_ps(pTemp[0], pTemp[2]);
    pTemp[1] = _mm_unpacklo_ps(pTemp[1], pTemp[3]);
    p = _mm_unpacklo_ps(pTemp[0], pTemp[1]);
}

inline void rpp_load4_f32_to_f32(Rpp32f *srcPtr, __m128 *p)
{
    *p = _mm_loadu_ps(srcPtr);
}

// ==================== SIMD Store Functions ====================
inline void rpp_store12_u8pkd3_to_u8pln3(Rpp8u* dstPtrR, Rpp8u* dstPtrG, Rpp8u* dstPtrB, __m128i &p)
{
    rpp_storeu_si32((__m128i *)(dstPtrR), _mm_shuffle_epi8(p, xmm_char_maskR));
    rpp_storeu_si32((__m128i *)(dstPtrG), _mm_shuffle_epi8(p, xmm_char_maskG));
    rpp_storeu_si32((__m128i *)(dstPtrB), _mm_shuffle_epi8(p, xmm_char_maskB));
}

inline void rpp_store12_u8_to_u8(Rpp8u* dstPtr, __m128i &p)
{
    _mm_storeu_si128((__m128i *)(dstPtr), p);
}

inline void rpp_store12_u8pln3_to_u8pkd3(Rpp8u* dstPtr, __m128i *p)
{
    __m128i px[4];
    px[0] = _mm_unpacklo_epi8(p[0], p[1]);
    px[1] = _mm_unpacklo_epi64(px[0], p[2]);
    _mm_storeu_si128((__m128i *)(dstPtr), _mm_shuffle_epi8(px[1], xmm_store4_pkd_pixels));
}

inline void rpp_store12_i8pkd3_to_i8pln3(Rpp8s* dstPtrR, Rpp8s* dstPtrG, Rpp8s* dstPtrB, __m128i &p)
{
    rpp_storeu_si32((__m128i *)(dstPtrR), _mm_shuffle_epi8(p, xmm_char_maskR));
    rpp_storeu_si32((__m128i *)(dstPtrG), _mm_shuffle_epi8(p, xmm_char_maskG));
    rpp_storeu_si32((__m128i *)(dstPtrB), _mm_shuffle_epi8(p, xmm_char_maskB));
}

inline void rpp_store12_i8_to_i8(Rpp8s* dstPtr, __m128i &p)
{
    _mm_storeu_si128((__m128i *)(dstPtr), p);
}

inline void rpp_store12_i8pln3_to_i8pkd3(Rpp8s* dstPtr, __m128i *p)
{
    __m128i px[4];
    px[0] = _mm_unpacklo_epi8(p[0], p[1]);
    px[1] = _mm_unpacklo_epi64(px[0], p[2]);
    _mm_storeu_si128((__m128i *)(dstPtr), _mm_shuffle_epi8(px[1], xmm_store4_pkd_pixels));
}

inline void rpp_store12_f32pln3_to_f32pln3(Rpp32f *dstPtrR, Rpp32f *dstPtrG, Rpp32f *dstPtrB, __m128 *p)
{
    _mm_storeu_ps(dstPtrR, p[0]);
    _mm_storeu_ps(dstPtrG, p[1]);
    _mm_storeu_ps(dstPtrB, p[2]);
}

inline void rpp_store12_f32pln3_to_f32pkd3(Rpp32f *dstPtr, __m128 *p)
{
    __m128 pTemp[3];
    pTemp[0] = _mm_unpacklo_ps(p[0], p[1]);
    pTemp[1] = _mm_unpacklo_ps(p[2], p[2]);
    pTemp[2] = _mm_unpackhi_ps(p[0], p[1]);
    _mm_storeu_ps(dstPtr, _mm_movelh_ps(pTemp[0], pTemp[1]));
    _mm_storeu_ps(dstPtr + 4, _mm_movehl_ps(pTemp[1], pTemp[0]));
    _mm_storeu_ps(dstPtr + 8, _mm_movelh_ps(pTemp[2], p[2]));
}

inline void rpp_store4_f32_to_f32(Rpp32f *dstPtr, __m128 p)
{
    _mm_storeu_ps(dstPtr, p);
}

// Define rpp_simd_load and rpp_simd_store macros
#define rpp_simd_load(function, ...) function(__VA_ARGS__)
#define rpp_simd_store(function, ...) function(__VA_ARGS__)



// The 4 functions (raw copy with fixes applied)
RppStatus resize_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8u *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr dstImgSize,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams srcLayoutParams,
                                      rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f hOffset = hRatio * 0.5f;
        Rpp32f wOffset = wRatio * 0.5f;
        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;

        Rpp8u *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~3;   // Align dst width to process 4 dst pixels per iteration
        __m128 pWRatio = _mm_set1_ps(wRatio);
        __m128 pWidthLimit = _mm_set1_ps((float)widthLimit);
        __m128 pWOffset = _mm_set1_ps(wOffset);
        __m128 pDstLoc;
        Rpp32s srcLocationColumnArray[4] = {0};     // Since 4 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcRowPtr;
            srcRowPtr = srcPtrChannel;
            Rpp8u *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8u *dstPtrTempR, *dstPtrTempG, *dstPtrTempB, *srcPtrTemp;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset, true);
                    rpp_simd_load(rpp_resize_nn_load_u8pkd3, srcPtrTemp, srcLocationColumnArray, pRow);
                    rpp_simd_store(rpp_store12_u8pkd3_to_u8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTempR++ = (Rpp8u)*(srcPtrTemp + srcLocationColumn);
                    *dstPtrTempG++ = (Rpp8u)*(srcPtrTemp + srcLocationColumn + 1);
                    *dstPtrTempB++ = (Rpp8u)*(srcPtrTemp + srcLocationColumn + 2);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8u *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8u * dstPtrTemp, *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTempR = srcPtrRowR + srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempG = srcPtrRowG + srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempB = srcPtrRowB + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow[3];
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrTempR, srcLocationColumnArray, pRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrTempG, srcLocationColumnArray, pRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrTempB, srcLocationColumnArray, pRow[2]);
                    rpp_simd_store(rpp_store12_u8pln3_to_u8pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTempR + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTempG + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTempB + srcLocationColumn);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8u *srcRowPtr;
            srcRowPtr = srcPtrChannel;
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8u *dstPtrTemp, *srcPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset, true);
                    rpp_simd_load(rpp_resize_nn_load_u8pkd3, srcPtrTemp, srcLocationColumnArray, pRow);
                    rpp_simd_store(rpp_store12_u8_to_u8, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTemp + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTemp + srcLocationColumn + 1);
                    *dstPtrTemp++ = (Rpp8u)*(srcPtrTemp + srcLocationColumn + 2);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8u *srcRowPtr;
            srcRowPtr = srcPtrChannel;
            Rpp8u *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8u *dstPtrTemp, *srcPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pRow;
                        rpp_simd_load(rpp_resize_nn_load_u8pln1, srcPtrTempChn, srcLocationColumnArray, pRow);
                        rpp_simd_store(rpp_storeu_si32, dstPtrTempChn, pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp8u *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp8u)*(srcPtrTempChn + srcLocationColumn);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus resize_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptImagePatchPtr dstImgSize,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams srcLayoutParams,
                                        rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f hOffset = hRatio * 0.5f;
        Rpp32f wOffset = wRatio * 0.5f;
        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;

        Rpp32f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~3;   // Align dst width to process 4 dst pixels per iteration
        __m128 pWRatio = _mm_set1_ps(wRatio);
        __m128 pWidthLimit = _mm_set1_ps((float)widthLimit);
        __m128 pWOffset = _mm_set1_ps(wOffset);
        __m128 pDstLoc;
        Rpp32s srcLocationColumnArray[4] = {0};     // Since 4 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcRowPtr;
            srcRowPtr = srcPtrChannel;
            Rpp32f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp32f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB, *srcPtrTemp;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pRow[3];
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset, true);
                    rpp_simd_load(rpp_resize_nn_load_f32pkd3_to_f32pln3, srcPtrTemp, srcLocationColumnArray, pRow);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTempR++ = (Rpp32f)*(srcPtrTemp + srcLocationColumn);
                    *dstPtrTempG++ = (Rpp32f)*(srcPtrTemp + srcLocationColumn + 1);
                    *dstPtrTempB++ = (Rpp32f)*(srcPtrTemp + srcLocationColumn + 2);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp32f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp32f * dstPtrTemp, *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTempR = srcPtrRowR + srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempG = srcPtrRowG + srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempB = srcPtrRowB + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pRow[3];
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrTempR, srcLocationColumnArray, pRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrTempG, srcLocationColumnArray, pRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrTempB, srcLocationColumnArray, pRow[2]);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    *dstPtrTemp++ = (Rpp32f)*(srcPtrTempR + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp32f)*(srcPtrTempG + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp32f)*(srcPtrTempB + srcLocationColumn);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp32f *srcRowPtr;
            srcRowPtr = srcPtrChannel;
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp32f *dstPtrTemp, *srcPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128 pRow[3];
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset, true);
                    rpp_simd_load(rpp_resize_nn_load_f32pkd3_to_f32pln3, srcPtrTemp, srcLocationColumnArray, pRow);
                    rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTemp++ = (Rpp32f)*(srcPtrTemp + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp32f)*(srcPtrTemp + srcLocationColumn + 1);
                    *dstPtrTemp++ = (Rpp32f)*(srcPtrTemp + srcLocationColumn + 2);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp32f *srcRowPtr;
            srcRowPtr = srcPtrChannel;
            Rpp32f *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp32f *dstPtrTemp, *srcPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp32f *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128 pRow;
                        rpp_simd_load(rpp_resize_nn_load_f32pln1, srcPtrTempChn, srcLocationColumnArray, pRow);
                        rpp_simd_store(rpp_store4_f32_to_f32, dstPtrTempChn, pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp32f *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp32f)*(srcPtrTempChn + srcLocationColumn);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus resize_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8s *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr dstImgSize,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams srcLayoutParams,
                                      rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f hOffset = hRatio * 0.5f;
        Rpp32f wOffset = wRatio * 0.5f;
        Rpp32s vectorIncrementPerChannel = 4;
        Rpp32s vectorIncrementPkd = 12;

        Rpp8s *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~3;   // Align dst width to process 4 dst pixels per iteration
        __m128 pWRatio = _mm_set1_ps(wRatio);
        __m128 pWidthLimit = _mm_set1_ps((float)widthLimit);
        __m128 pWOffset = _mm_set1_ps(wOffset);
        __m128 pDstLoc;
        Rpp32s srcLocationColumnArray[4] = {0};     // Since 4 dst pixels are processed per iteration
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with fused output-layout toggle (NHWC -> NCHW)
        if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcRowPtr;
            srcRowPtr = srcPtrChannel;
            Rpp8s *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8s *dstPtrTempR, *dstPtrTempG, *dstPtrTempB, *srcPtrTemp;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset, true);
                    rpp_simd_load(rpp_resize_nn_load_i8pkd3, srcPtrTemp, srcLocationColumnArray, pRow);
                    rpp_simd_store(rpp_store12_i8pkd3_to_i8pln3, dstPtrTempR, dstPtrTempG, dstPtrTempB, pRow);
                    dstPtrTempR += vectorIncrementPerChannel;
                    dstPtrTempG += vectorIncrementPerChannel;
                    dstPtrTempB += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTempR++ = (Rpp8s)*(srcPtrTemp + srcLocationColumn);
                    *dstPtrTempG++ = (Rpp8s)*(srcPtrTemp + srcLocationColumn + 1);
                    *dstPtrTempB++ = (Rpp8s)*(srcPtrTemp + srcLocationColumn + 2);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;
            Rpp8s *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8s * dstPtrTemp, *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTempR = srcPtrRowR + srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempG = srcPtrRowG + srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempB = srcPtrRowB + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow[3];
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset);
                    rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrTempR, srcLocationColumnArray, pRow[0]);
                    rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrTempG, srcLocationColumnArray, pRow[1]);
                    rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrTempB, srcLocationColumnArray, pRow[2]);
                    rpp_simd_store(rpp_store12_i8pln3_to_i8pkd3, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTempR + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTempG + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTempB + srcLocationColumn);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NHWC -> NHWC)
        else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
        {
            Rpp8s *srcRowPtr;
            srcRowPtr = srcPtrChannel;
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8s *dstPtrTemp, *srcPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    __m128i pRow;
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset, true);
                    rpp_simd_load(rpp_resize_nn_load_i8pkd3, srcPtrTemp, srcLocationColumnArray, pRow);
                    rpp_simd_store(rpp_store12_i8_to_i8, dstPtrTemp, pRow);
                    dstPtrTemp += vectorIncrementPkd;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset, srcDescPtr->strides.wStride);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTemp + srcLocationColumn);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTemp + srcLocationColumn + 1);
                    *dstPtrTemp++ = (Rpp8s)*(srcPtrTemp + srcLocationColumn + 2);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }

        // Resize with fused output-layout toggle (NCHW -> NCHW)
        else if ((srcDescPtr->layout == RpptLayout::NCHW) && (dstDescPtr->layout == RpptLayout::NCHW))
        {
            Rpp8s *srcRowPtr;
            srcRowPtr = srcPtrChannel;
            Rpp8s *dstPtrRow;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp8s *dstPtrTemp, *srcPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcRowPtr + srcLocationRow * srcDescPtr->strides.hStride;
                pDstLoc = xmm_pDstLocInit;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
                {
                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        __m128i pRow;
                        rpp_simd_load(rpp_resize_nn_load_i8pln1, srcPtrTempChn, srcLocationColumnArray, pRow);
                        rpp_simd_store(rpp_storeu_si32, dstPtrTempChn, pRow);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp += vectorIncrementPerChannel;
                }
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    Rpp8s *dstPtrTempChn, *srcPtrTempChn;
                    srcPtrTempChn = srcPtrTemp;
                    dstPtrTempChn = dstPtrTemp;
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    for(int c = 0; c < srcDescPtr->c; c++)
                    {
                        *dstPtrTempChn = (Rpp8s)*(srcPtrTempChn + srcLocationColumn);
                        srcPtrTempChn += srcDescPtr->strides.cStride;
                        dstPtrTempChn += dstDescPtr->strides.cStride;
                    }
                    dstPtrTemp++;
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus resize_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp16f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptImagePatchPtr dstImgSize,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams srcLayoutParams,
                                        rpp::Handle& handle)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    Rpp32u numThreads = handle.GetNumThreads();

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(numThreads)
    for(int batchCount = 0; batchCount < dstDescPtr->n; batchCount++)
    {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        compute_dst_size_cap_host(&dstImgSize[batchCount], dstDescPtr);
        Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize[batchCount].width));
        Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize[batchCount].height));
        Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
        Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
        Rpp32f hOffset = hRatio * 0.5f;
        Rpp32f wOffset = wRatio * 0.5f;

        Rpp16f *srcPtrChannel, *dstPtrChannel, *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;
        srcPtrChannel = srcPtrImage + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
        dstPtrChannel = dstPtrImage;

        Rpp32u alignedLength = dstImgSize[batchCount].width & ~3;
        Rpp32s srcLocationColumnArray[4] = {0};
        Rpp32s srcLocationRow, srcLocationColumn;

        // Resize with 3 channel inputs and outputs (assume NCHW for c==3, as per original)
        if (srcDescPtr->c == 3 && srcDescPtr->layout == RpptLayout::NCHW && dstDescPtr->layout == RpptLayout::NCHW)
        {
            Rpp16f *srcPtrRowR, *srcPtrRowG, *srcPtrRowB;
            srcPtrRowR = srcPtrChannel;
            srcPtrRowG = srcPtrRowR + srcDescPtr->strides.cStride;
            srcPtrRowB = srcPtrRowG + srcDescPtr->strides.cStride;
            Rpp16f *dstPtrRowR, *dstPtrRowG, *dstPtrRowB;
            dstPtrRowR = dstPtrChannel;
            dstPtrRowG = dstPtrRowR + dstDescPtr->strides.cStride;
            dstPtrRowB = dstPtrRowG + dstDescPtr->strides.cStride;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp16f *dstPtrTempR, *dstPtrTempG, *dstPtrTempB, *srcPtrTempR, *srcPtrTempG, *srcPtrTempB;
                dstPtrTempR = dstPtrRowR;
                dstPtrTempG = dstPtrRowG;
                dstPtrTempB = dstPtrRowB;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTempR = srcPtrRowR + srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempG = srcPtrRowG + srcLocationRow * srcDescPtr->strides.hStride;
                srcPtrTempB = srcPtrRowB + srcLocationRow * srcDescPtr->strides.hStride;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    *dstPtrTempR++ = (Rpp16f)*(srcPtrTempR + srcLocationColumn);
                    *dstPtrTempG++ = (Rpp16f)*(srcPtrTempG + srcLocationColumn);
                    *dstPtrTempB++ = (Rpp16f)*(srcPtrTempB + srcLocationColumn);
                }
                dstPtrRowR += dstDescPtr->strides.hStride;
                dstPtrRowG += dstDescPtr->strides.hStride;
                dstPtrRowB += dstDescPtr->strides.hStride;
            }
        }

        // Resize with single channel inputs and outputs
        else
        {
            Rpp16f *srcPtrRow, *dstPtrRow;
            srcPtrRow = srcPtrChannel;
            dstPtrRow = dstPtrChannel;

            for(int i = 0; i < dstImgSize[batchCount].height; i++)
            {
                Rpp16f *srcPtrTemp, *dstPtrTemp;
                dstPtrTemp = dstPtrRow;
                compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
                srcPtrTemp = srcPtrRow + srcLocationRow * srcDescPtr->strides.hStride;

                int vectorLoopCount = 0;
                for (; vectorLoopCount < dstImgSize[batchCount].width; vectorLoopCount++)
                {
                    compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                    *dstPtrTemp++ = (Rpp16f)*(srcPtrTemp + srcLocationColumn);
                }
                dstPtrRow += dstDescPtr->strides.hStride;
            }
        }
    }

    return RPP_SUCCESS;
}

// Add this after the resize functions
int main() {
    // Small 2x2 RGB image (NHWC): pixel1 [1,2,3], pixel2 [4,5,6], pixel3 [7,8,9], pixel4 [10,11,12]
    Rpp8u srcData[2*2*3] = {1,2,3, 4,5,6, 7,8,9, 10,11,12};

    // Source desc (NHWC, batch=1)
    RpptDesc srcDesc;
    srcDesc.numDims = 4;
    srcDesc.offsetInBytes = 0;
    srcDesc.n = 1;
    srcDesc.c = 3;
    srcDesc.h = 2;
    srcDesc.w = 2;
    srcDesc.layout = NHWC;
    srcDesc.strides.nStride = 1*3*2*2;
    srcDesc.strides.cStride = 1;
    srcDesc.strides.hStride = 2*3;
    srcDesc.strides.wStride = 3;

    // Destination desc (NCHW, 1x1)
    Rpp8u dstData[3];  // 1 pixel, 3 channels
    RpptDesc dstDesc = srcDesc;
    dstDesc.h = 1;
    dstDesc.w = 1;
    dstDesc.layout = NCHW;
    dstDesc.strides.hStride = 1*3;
    dstDesc.strides.wStride = 1;

    // Dst size
    RpptImagePatch dstSize;
    dstSize.height = 1;
    dstSize.width = 1;

    // ROI (full image)
    RpptROI roiPtrSrc;
    roiPtrSrc.xywhROI.xy.x = 0;
    roiPtrSrc.xywhROI.xy.y = 0;
    roiPtrSrc.xywhROI.roiWidth = 2;
    roiPtrSrc.xywhROI.roiHeight = 2;

    // Layout params (for source)
    RppLayoutParams srcParams;
    srcParams.layout = NHWC;
    srcParams.bufferMultiplier = 3;  // for PKD3

    // Handle
    rpp::Handle handle(1, 1);

    // Call u8 function
    RppStatus status = resize_nn_u8_u8_host_tensor(
        srcData, &srcDesc, dstData, &dstDesc, &dstSize, &roiPtrSrc, XYWH, srcParams, handle
    );

    if (status == RPP_SUCCESS) {
        printf("Resize successful!\n");
        printf("Output pixel (NCHW): R=%d, G=%d, B=%d\n", dstData[0], dstData[1], dstData[2]);
    } else {
        printf("Resize failed!\n");
    }

    // Expected: Depending on offset, might pick pixel at src_loc ~0.5*2=1, so pixel3 or 4
    return 0;
}