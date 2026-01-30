/*
 * Standalone Resize Nearest Neighbor Implementation - Single Image Version
 * Extracted from RPP (ROCm Performance Primitives) for benchmarking
 * Modified to handle single images instead of batches
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <cmath>

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
    p[0] = _mm_loadu_ps(srcRowPtrsForInterp + loc[0]);  // LOC0 load [R01|G01|B01|R02] - Need RGB 01
    p[1] = _mm_loadu_ps(srcRowPtrsForInterp + loc[1]);  // LOC1 load [R11|G11|B11|R12] - Need RGB 11
    p[2] = _mm_loadu_ps(srcRowPtrsForInterp + loc[2]);  // LOC2 load [R21|G21|B21|R22] - Need RGB 21
    __m128 pTemp = _mm_loadu_ps(srcRowPtrsForInterp + loc[3]);  // LOC2 load [R31|G31|B31|R32]  - Need RGB 31
    _MM_TRANSPOSE4_PS(p[0], p[1], p[2], pTemp); // Transpose to obtain RGB in each vector
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
    _MM_TRANSPOSE4_PS(p[0], p[1], p[2], p[3]);
    _mm_storeu_ps(dstPtr, p[0]);
    _mm_storeu_ps(dstPtr + 3, p[1]);
    _mm_storeu_ps(dstPtr + 6, p[2]);
    _mm_storeu_ps(dstPtr + 9, p[3]);
}

inline void rpp_store4_f32_to_f32(Rpp32f *dstPtr, __m128 p)
{
    _mm_storeu_ps(dstPtr, p);
}

// Define rpp_simd_load and rpp_simd_store macros
#define rpp_simd_load(function, ...) function(__VA_ARGS__)
#define rpp_simd_store(function, ...) function(__VA_ARGS__)

// ==================== Single Image Resize Functions ====================

RppStatus resize_nn_u8_u8_host_tensor(Rpp8u *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8u *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr dstImgSize,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    RpptROI roi;
    RpptROIPtr roiPtrInput = roiTensorPtrSrc;
    compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

    compute_dst_size_cap_host(dstImgSize, dstDescPtr);
    Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize->width));
    Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize->height));
    Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
    Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
    Rpp32f hOffset = hRatio * 0.5f;
    Rpp32f wOffset = wRatio * 0.5f;
    Rpp32s vectorIncrementPerChannel = 4;
    Rpp32s vectorIncrementPkd = 12;

    Rpp8u *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtr + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
    dstPtrChannel = dstPtr;

    Rpp32u alignedLength = dstImgSize->width & ~3;
    __m128 pWRatio = _mm_set1_ps(wRatio);
    __m128 pWidthLimit = _mm_set1_ps((float)widthLimit);
    __m128 pWOffset = _mm_set1_ps(wOffset);
    __m128 pDstLoc;
    Rpp32s srcLocationColumnArray[4] = {0};
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

        for(int i = 0; i < dstImgSize->height; i++)
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
            for (; vectorLoopCount < dstImgSize->width; vectorLoopCount++)
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

        for(int i = 0; i < dstImgSize->height; i++)
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
            for (; vectorLoopCount < dstImgSize->width; vectorLoopCount++)
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

        for(int i = 0; i < dstImgSize->height; i++)
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
            for (; vectorLoopCount < dstImgSize->width; vectorLoopCount++)
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

        for(int i = 0; i < dstImgSize->height; i++)
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
            for (; vectorLoopCount < dstImgSize->width; vectorLoopCount++)
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

    return RPP_SUCCESS;
}

RppStatus resize_nn_f32_f32_host_tensor(Rpp32f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp32f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptImagePatchPtr dstImgSize,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    RpptROI roi;
    RpptROIPtr roiPtrInput = roiTensorPtrSrc;
    compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

    compute_dst_size_cap_host(dstImgSize, dstDescPtr);
    Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize->width));
    Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize->height));
    Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
    Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
    Rpp32f hOffset = hRatio * 0.5f;
    Rpp32f wOffset = wRatio * 0.5f;
    Rpp32s vectorIncrementPerChannel = 4;
    Rpp32s vectorIncrementPkd = 12;

    Rpp32f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtr + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
    dstPtrChannel = dstPtr;

    Rpp32u alignedLength = dstImgSize->width & ~3;
    __m128 pWRatio = _mm_set1_ps(wRatio);
    __m128 pWidthLimit = _mm_set1_ps((float)widthLimit);
    __m128 pWOffset = _mm_set1_ps(wOffset);
    __m128 pDstLoc;
    Rpp32s srcLocationColumnArray[4] = {0};
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

        for(int i = 0; i < dstImgSize->height; i++)
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
            for (; vectorLoopCount < dstImgSize->width; vectorLoopCount++)
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

        for(int i = 0; i < dstImgSize->height; i++)
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
            for (; vectorLoopCount < dstImgSize->width; vectorLoopCount++)
            {
                compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                *dstPtrTemp++ = (Rpp32f)*(srcPtrTempR + srcLocationColumn);
                *dstPtrTemp++ = (Rpp32f)*(srcPtrTempG + srcLocationColumn);
                *dstPtrTemp++ = (Rpp32f)*(srcPtrTempB + srcLocationColumn);
            }
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    }

    // Resize with fused output-layout toggle (NHWC -> NHWC) - FIXED VERSION
    else if ((srcDescPtr->c == 3) && (srcDescPtr->layout == RpptLayout::NHWC) && (dstDescPtr->layout == RpptLayout::NHWC))
    {
        Rpp32f *srcRowPtr;
        srcRowPtr = srcPtrChannel;
        Rpp32f *dstPtrRow;
        dstPtrRow = dstPtrChannel;

        for(int i = 0; i < dstImgSize->height; i++)
        {
            Rpp32f *dstPtrTemp, *srcPtrTemp;
            dstPtrTemp = dstPtrRow;
            compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
            srcPtrTemp = srcRowPtr + srcLocationRow * srcDescPtr->strides.hStride;
            pDstLoc = xmm_pDstLocInit;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < alignedLength; vectorLoopCount += vectorIncrementPerChannel)
            {
                __m128 pRow[3];   // ← keep [3] exactly like original
                compute_resize_nn_src_loc_sse(pDstLoc, pWRatio, pWidthLimit, srcLocationColumnArray, pWOffset, true);
                rpp_simd_load(rpp_resize_nn_load_f32pkd3_to_f32pln3, srcPtrTemp, srcLocationColumnArray, pRow);
                rpp_simd_store(rpp_store12_f32pln3_to_f32pkd3, dstPtrTemp, pRow);   // ← safe, as explained
                dstPtrTemp += vectorIncrementPkd;
            }
            for (; vectorLoopCount < dstImgSize->width; vectorLoopCount++)
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

        for(int i = 0; i < dstImgSize->height; i++)
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
            for (; vectorLoopCount < dstImgSize->width; vectorLoopCount++)
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

    return RPP_SUCCESS;
}

RppStatus resize_nn_i8_i8_host_tensor(Rpp8s *srcPtr,
                                      RpptDescPtr srcDescPtr,
                                      Rpp8s *dstPtr,
                                      RpptDescPtr dstDescPtr,
                                      RpptImagePatchPtr dstImgSize,
                                      RpptROIPtr roiTensorPtrSrc,
                                      RpptRoiType roiType,
                                      RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    RpptROI roi;
    RpptROIPtr roiPtrInput = roiTensorPtrSrc;
    compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

    compute_dst_size_cap_host(dstImgSize, dstDescPtr);
    Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize->width));
    Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize->height));
    Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
    Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
    Rpp32f hOffset = hRatio * 0.5f;
    Rpp32f wOffset = wRatio * 0.5f;
    Rpp32s vectorIncrementPerChannel = 4;
    Rpp32s vectorIncrementPkd = 12;

    Rpp8s *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtr + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
    dstPtrChannel = dstPtr;

    Rpp32u alignedLength = dstImgSize->width & ~3;
    __m128 pWRatio = _mm_set1_ps(wRatio);
    __m128 pWidthLimit = _mm_set1_ps((float)widthLimit);
    __m128 pWOffset = _mm_set1_ps(wOffset);
    __m128 pDstLoc;
    Rpp32s srcLocationColumnArray[4] = {0};
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

        for(int i = 0; i < dstImgSize->height; i++)
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
            for (; vectorLoopCount < dstImgSize->width; vectorLoopCount++)
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

        for(int i = 0; i < dstImgSize->height; i++)
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
            for (; vectorLoopCount < dstImgSize->width; vectorLoopCount++)
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

        for(int i = 0; i < dstImgSize->height; i++)
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
            for (; vectorLoopCount < dstImgSize->width; vectorLoopCount++)
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

        for(int i = 0; i < dstImgSize->height; i++)
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
            for (; vectorLoopCount < dstImgSize->width; vectorLoopCount++)
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

    return RPP_SUCCESS;
}

RppStatus resize_nn_f16_f16_host_tensor(Rpp16f *srcPtr,
                                        RpptDescPtr srcDescPtr,
                                        Rpp16f *dstPtr,
                                        RpptDescPtr dstDescPtr,
                                        RpptImagePatchPtr dstImgSize,
                                        RpptROIPtr roiTensorPtrSrc,
                                        RpptRoiType roiType,
                                        RppLayoutParams srcLayoutParams)
{
    RpptROI roiDefault = {0, 0, (Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h};
    RpptROI roi;
    RpptROIPtr roiPtrInput = roiTensorPtrSrc;
    compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

    compute_dst_size_cap_host(dstImgSize, dstDescPtr);
    Rpp32f wRatio = ((Rpp32f)(roi.xywhROI.roiWidth)) / ((Rpp32f)(dstImgSize->width));
    Rpp32f hRatio = ((Rpp32f)(roi.xywhROI.roiHeight)) / ((Rpp32f)(dstImgSize->height));
    Rpp32u heightLimit = roi.xywhROI.roiHeight - 1;
    Rpp32u widthLimit = roi.xywhROI.roiWidth - 1;
    Rpp32f hOffset = hRatio * 0.5f;
    Rpp32f wOffset = wRatio * 0.5f;

    Rpp16f *srcPtrChannel, *dstPtrChannel;
    srcPtrChannel = srcPtr + (roi.xywhROI.xy.y * srcDescPtr->strides.hStride) + (roi.xywhROI.xy.x * srcLayoutParams.bufferMultiplier);
    dstPtrChannel = dstPtr;

    Rpp32u alignedLength = dstImgSize->width & ~3;
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

        for(int i = 0; i < dstImgSize->height; i++)
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
            for (; vectorLoopCount < dstImgSize->width; vectorLoopCount++)
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

        for(int i = 0; i < dstImgSize->height; i++)
        {
            Rpp16f *srcPtrTemp, *dstPtrTemp;
            dstPtrTemp = dstPtrRow;
            compute_resize_nn_src_loc(i, hRatio, heightLimit, srcLocationRow, hOffset);
            srcPtrTemp = srcPtrRow + srcLocationRow * srcDescPtr->strides.hStride;

            int vectorLoopCount = 0;
            for (; vectorLoopCount < dstImgSize->width; vectorLoopCount++)
            {
                compute_resize_nn_src_loc(vectorLoopCount, wRatio, widthLimit, srcLocationColumn, wOffset);
                *dstPtrTemp++ = (Rpp16f)*(srcPtrTemp + srcLocationColumn);
            }
            dstPtrRow += dstDescPtr->strides.hStride;
        }
    }

    return RPP_SUCCESS;
}

int main()
{
    // === Configuration - change these paths as needed ===
    const char* input_image_path = "eight_images_mixed_src1/img300x300.jpg";   // forward slash works too on Windows    const char* output_folder      = "./output/";
          // your test image (can be PNG/JPG/BMP)
    const char* output_folder      = "D:/shaheen/RPP/output/";

    // Desired output sizes (you can change these)
    int target_width_down  = 100;
    int target_height_down = 100;

    int target_width_up    = 1024;
    int target_height_up   = 1024;

    // === Load image (NHWC RGB uint8 by default) ===
    int width, height, channels;
    unsigned char* img_data = stbi_load(input_image_path, &width, &height, &channels, 3); // force 3 channels

    if (!img_data) {
        printf("Error: cannot load image %s\n", input_image_path);
        return 1;
    }

    printf("Loaded image: %s  %d x %d   channels: %d\n", input_image_path, width, height, channels);

    // We assume RGB (3 channels) NHWC layout for input
    RpptDesc srcDesc;
    srcDesc.n = 1;
    srcDesc.c = 3;
    srcDesc.h = height;
    srcDesc.w = width;
    srcDesc.layout = RpptLayout::NHWC;
    srcDesc.strides.nStride   = height * width * 3;
    srcDesc.strides.cStride   = 1;                    // not used in NHWC
    srcDesc.strides.hStride   = width * 3;
    srcDesc.strides.wStride   = 3;

    RppLayoutParams layoutParams;
    layoutParams.layout = RpptLayout::NHWC;
    layoutParams.bufferMultiplier = 3;

    RpptROI roi;
    roi.xywhROI.xy.x       = 0;
    roi.xywhROI.xy.y       = 0;
    roi.xywhROI.roiWidth   = width;
    roi.xywhROI.roiHeight  = height;

    // =============================================================
    // Test 1: unsigned char (u8)  - NHWC → NHWC
    // =============================================================
    {
        unsigned char* dst_u8_down = new unsigned char[target_height_down * target_width_down * 3];
        unsigned char* dst_u8_up   = new unsigned char[target_height_up   * target_width_up   * 3];

        RpptDesc dstDesc_down = srcDesc;
        dstDesc_down.h = target_height_down;
        dstDesc_down.w = target_width_down;
        dstDesc_down.strides.hStride = target_width_down * 3;
        dstDesc_down.strides.nStride = target_height_down * target_width_down * 3;

        RpptImagePatch dstSize_down = { (Rpp32u)target_width_down, (Rpp32u)target_height_down };

        resize_nn_u8_u8_host_tensor(img_data, &srcDesc, dst_u8_down, &dstDesc_down,
                                    &dstSize_down, &roi, XYWH, layoutParams);

        RpptDesc dstDesc_up = srcDesc;
        dstDesc_up.h = target_height_up;
        dstDesc_up.w = target_width_up;
        dstDesc_up.strides.hStride = target_width_up * 3;
        dstDesc_up.strides.nStride = target_height_up * target_width_up * 3;

        RpptImagePatch dstSize_up = { (Rpp32u)target_width_up, (Rpp32u)target_height_up };

        resize_nn_u8_u8_host_tensor(img_data, &srcDesc, dst_u8_up, &dstDesc_up,
                                    &dstSize_up, &roi, XYWH, layoutParams);

        char path_down[256], path_up[256];
        snprintf(path_down, sizeof(path_down), "%su8_down.png", output_folder);
        snprintf(path_up,   sizeof(path_up),   "%su8_up.png",   output_folder);

        stbi_write_png(path_down, target_width_down,  target_height_down,  3, dst_u8_down, target_width_down * 3);
        stbi_write_png(path_up,   target_width_up,    target_height_up,    3, dst_u8_up,   target_width_up   * 3);

        printf("u8: saved %s and %s\n", path_down, path_up);

        delete[] dst_u8_down;
        delete[] dst_u8_up;
    }

    // =============================================================
    // Test 2: float (f32)  - NHWC → NHWC
    // =============================================================
    {
        float* src_f32 = new float[width * height * 3];
        for (int i = 0; i < width * height * 3; i++) {
            src_f32[i] = img_data[i] / 255.0f;
        }

        float* dst_f32_down = new float[target_height_down * target_width_down * 3];
        float* dst_f32_up   = new float[target_height_up   * target_width_up   * 3];

        RpptDesc srcDesc_f32 = srcDesc;
        srcDesc_f32.strides.wStride = 3;
        srcDesc_f32.strides.hStride = width * 3;

        RpptDesc dstDesc_down_f32 = srcDesc_f32;
        dstDesc_down_f32.h = target_height_down;
        dstDesc_down_f32.w = target_width_down;
        dstDesc_down_f32.strides.hStride = target_width_down * 3;
        dstDesc_down_f32.strides.nStride = target_height_down * target_width_down * 3;

        RpptImagePatch dstSize_down = { (Rpp32u)target_width_down, (Rpp32u)target_height_down };

        resize_nn_f32_f32_host_tensor(src_f32, &srcDesc_f32, dst_f32_down, &dstDesc_down_f32,
                                      &dstSize_down, &roi, XYWH, layoutParams);

        RpptDesc dstDesc_up_f32 = srcDesc_f32;
        dstDesc_up_f32.h = target_height_up;
        dstDesc_up_f32.w = target_width_up;
        dstDesc_up_f32.strides.hStride = target_width_up * 3;
        dstDesc_up_f32.strides.nStride = target_height_up * target_width_up * 3;

        RpptImagePatch dstSize_up = { (Rpp32u)target_width_up, (Rpp32u)target_height_up };

        resize_nn_f32_f32_host_tensor(src_f32, &srcDesc_f32, dst_f32_up, &dstDesc_up_f32,
                                      &dstSize_up, &roi, XYWH, layoutParams);

        // Save as PNG (multiply back to 0-255)
        unsigned char* tmp_down = new unsigned char[target_height_down * target_width_down * 3];
        unsigned char* tmp_up   = new unsigned char[target_height_up   * target_width_up   * 3];

        for (int i = 0; i < target_height_down * target_width_down * 3; i++)
            tmp_down[i] = (unsigned char)std::round(std::clamp(dst_f32_down[i], 0.0f, 1.0f) * 255.0f);

        for (int i = 0; i < target_height_up * target_width_up * 3; i++)
            tmp_up[i] = (unsigned char)std::round(std::clamp(dst_f32_up[i], 0.0f, 1.0f) * 255.0f);

        char path_f32_down[256], path_f32_up[256];
        snprintf(path_f32_down, sizeof(path_f32_down), "%sf32_down.png", output_folder);
        snprintf(path_f32_up,   sizeof(path_f32_up),   "%sf32_up.png",   output_folder);

        stbi_write_png(path_f32_down, target_width_down,  target_height_down,  3, tmp_down, target_width_down * 3);
        stbi_write_png(path_f32_up,   target_width_up,    target_height_up,    3, tmp_up,   target_width_up   * 3);

        printf("f32: saved %s and %s\n", path_f32_down, path_f32_up);

        delete[] src_f32;
        delete[] dst_f32_down;
        delete[] dst_f32_up;
        delete[] tmp_down;
        delete[] tmp_up;
    }

    // =============================================================
    // Test 3: signed char (i8)  - NHWC → NHWC   (map 0-255 → -128..127)
    // =============================================================
    {
        signed char* src_i8 = new signed char[width * height * 3];
        for (int i = 0; i < width * height * 3; i++) {
            src_i8[i] = (signed char)(img_data[i] - 128);
        }

        signed char* dst_i8_down = new signed char[target_height_down * target_width_down * 3];
        signed char* dst_i8_up   = new signed char[target_height_up   * target_width_up   * 3];

        RpptDesc srcDesc_i8 = srcDesc;
        RpptDesc dstDesc_down_i8 = srcDesc;
        dstDesc_down_i8.h = target_height_down;
        dstDesc_down_i8.w = target_width_down;
        dstDesc_down_i8.strides.hStride = target_width_down * 3;

        RpptImagePatch dstSize_down = { (Rpp32u)target_width_down, (Rpp32u)target_height_down };

        resize_nn_i8_i8_host_tensor(src_i8, &srcDesc_i8, dst_i8_down, &dstDesc_down_i8,
                                    &dstSize_down, &roi, XYWH, layoutParams);

        RpptDesc dstDesc_up_i8 = srcDesc;
        dstDesc_up_i8.h = target_height_up;
        dstDesc_up_i8.w = target_width_up;
        dstDesc_up_i8.strides.hStride = target_width_up * 3;

        RpptImagePatch dstSize_up = { (Rpp32u)target_width_up, (Rpp32u)target_height_up };

        resize_nn_i8_i8_host_tensor(src_i8, &srcDesc_i8, dst_i8_up, &dstDesc_up_i8,
                                    &dstSize_up, &roi, XYWH, layoutParams);


        // Convert back to uint8 for saving
        unsigned char* tmp_i8_down = new unsigned char[target_height_down * target_width_down * 3];
        unsigned char* tmp_i8_up   = new unsigned char[target_height_up   * target_width_up   * 3];

        for (int i = 0; i < target_height_down * target_width_down * 3; i++)
            tmp_i8_down[i] = (unsigned char)std::clamp(dst_i8_down[i] + 128, 0, 255);

        for (int i = 0; i < target_height_up * target_width_up * 3; i++)
            tmp_i8_up[i] = (unsigned char)std::clamp(dst_i8_up[i] + 128, 0, 255);

        char path_i8_down[256], path_i8_up[256];
        snprintf(path_i8_down, sizeof(path_i8_down), "%si8_down.png", output_folder);
        snprintf(path_i8_up,   sizeof(path_i8_up),   "%si8_up.png",   output_folder);

        stbi_write_png(path_i8_down, target_width_down,  target_height_down,  3, tmp_i8_down, target_width_down * 3);
        stbi_write_png(path_i8_up,   target_width_up,    target_height_up,    3, tmp_i8_up,   target_width_up   * 3);

        printf("i8: saved %s and %s\n", path_i8_down, path_i8_up);

        delete[] src_i8;
        delete[] dst_i8_down;
        delete[] dst_i8_up;
        delete[] tmp_i8_down;
        delete[] tmp_i8_up;
    }

    // =============================================================
    // Test 4: half-float (f16)  - NHWC input → NCHW output
    // =============================================================
    {
        // Convert NHWC uint8 → NCHW half-float
        halfhpp* src_f16_nchw = new halfhpp[3 * height * width];
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int nhwc_idx = y * width * 3 + x * 3 + c;
                    int nchw_idx = c * height * width + y * width + x;
                    src_f16_nchw[nchw_idx] = halfhpp(img_data[nhwc_idx] / 255.0f);
                }
            }
        }

        // Output buffer (NCHW: C*H*W)
        int out_h = target_height_down;
        int out_w = target_width_down;
        halfhpp* dst_f16_down = new halfhpp[3 * out_h * out_w];

        // Source descriptor → NCHW
        RpptDesc srcDesc_f16_nchw;
        srcDesc_f16_nchw.n = 1;
        srcDesc_f16_nchw.c = 3;
        srcDesc_f16_nchw.h = height;
        srcDesc_f16_nchw.w = width;
        srcDesc_f16_nchw.layout = RpptLayout::NCHW;
        srcDesc_f16_nchw.strides.nStride   = 3 * height * width;
        srcDesc_f16_nchw.strides.cStride   = height * width;
        srcDesc_f16_nchw.strides.hStride   = width;
        srcDesc_f16_nchw.strides.wStride   = 1;

        // Destination descriptor → NCHW
        RpptDesc dstDesc_f16_down;
        dstDesc_f16_down.n = 1;
        dstDesc_f16_down.c = 3;
        dstDesc_f16_down.h = out_h;
        dstDesc_f16_down.w = out_w;
        dstDesc_f16_down.layout = RpptLayout::NCHW;
        dstDesc_f16_down.strides.nStride   = 3 * out_h * out_w;
        dstDesc_f16_down.strides.cStride   = out_h * out_w;
        dstDesc_f16_down.strides.hStride   = out_w;
        dstDesc_f16_down.strides.wStride   = 1;

        RpptImagePatch dstSize_down_f16 = { (Rpp32u)out_w, (Rpp32u)out_h };

        // Call the function
        resize_nn_f16_f16_host_tensor(src_f16_nchw, &srcDesc_f16_nchw, dst_f16_down, &dstDesc_f16_down,
                                    &dstSize_down_f16, &roi, XYWH, layoutParams);

        // Save as PNG (NCHW → NHWC conversion for stb_write)
        unsigned char* tmp_f16_down = new unsigned char[out_h * out_w * 3];
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < out_h; y++) {
                for (int x = 0; x < out_w; x++) {
                    int nchw_idx = c * out_h * out_w + y * out_w + x;
                    float val = float(dst_f16_down[nchw_idx]);
                    tmp_f16_down[y * out_w * 3 + x * 3 + c] = (unsigned char)std::round(std::clamp(val, 0.0f, 1.0f) * 255.0f);
                }
            }
        }

        char path_f16_down[256];
        snprintf(path_f16_down, sizeof(path_f16_down), "%sf16_down_nchw.png", output_folder);
        stbi_write_png(path_f16_down, out_w, out_h, 3, tmp_f16_down, out_w * 3);

        printf("f16 (NCHW output): saved %s\n", path_f16_down);

        delete[] src_f16_nchw;
        delete[] dst_f16_down;
        delete[] tmp_f16_down;

    // Add this after your existing F16 downscale test (around line 950)

    // =============================================================
    // F16 UPSCALE TEST
    // =============================================================
    {
        // Convert NHWC uint8 → NCHW half-float (same as before)
        halfhpp* src_f16_nchw = new halfhpp[3 * height * width];
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int nhwc_idx = y * width * 3 + x * 3 + c;
                    int nchw_idx = c * height * width + y * width + x;
                    src_f16_nchw[nchw_idx] = halfhpp(img_data[nhwc_idx] / 255.0f);
                }
            }
        }

        // Output buffer for UPSCALE (NCHW: C*H*W)
        int out_h_up = target_height_up;
        int out_w_up = target_width_up;
        halfhpp* dst_f16_up = new halfhpp[3 * out_h_up * out_w_up];

        // Source descriptor → NCHW (same as before)
        RpptDesc srcDesc_f16_nchw;
        srcDesc_f16_nchw.n = 1;
        srcDesc_f16_nchw.c = 3;
        srcDesc_f16_nchw.h = height;
        srcDesc_f16_nchw.w = width;
        srcDesc_f16_nchw.layout = RpptLayout::NCHW;
        srcDesc_f16_nchw.strides.nStride   = 3 * height * width;
        srcDesc_f16_nchw.strides.cStride   = height * width;
        srcDesc_f16_nchw.strides.hStride   = width;
        srcDesc_f16_nchw.strides.wStride   = 1;

        // Destination descriptor → NCHW for UPSCALE
        RpptDesc dstDesc_f16_up;
        dstDesc_f16_up.n = 1;
        dstDesc_f16_up.c = 3;
        dstDesc_f16_up.h = out_h_up;
        dstDesc_f16_up.w = out_w_up;
        dstDesc_f16_up.layout = RpptLayout::NCHW;
        dstDesc_f16_up.strides.nStride   = 3 * out_h_up * out_w_up;
        dstDesc_f16_up.strides.cStride   = out_h_up * out_w_up;
        dstDesc_f16_up.strides.hStride   = out_w_up;
        dstDesc_f16_up.strides.wStride   = 1;

        RpptImagePatch dstSize_up_f16 = { (Rpp32u)out_w_up, (Rpp32u)out_h_up };

        // Call the resize function for UPSCALE
        resize_nn_f16_f16_host_tensor(src_f16_nchw, &srcDesc_f16_nchw, dst_f16_up, &dstDesc_f16_up,
                                    &dstSize_up_f16, &roi, XYWH, layoutParams);

        // Save as PNG (NCHW → NHWC conversion for stb_write)
        unsigned char* tmp_f16_up = new unsigned char[out_h_up * out_w_up * 3];
        for (int c = 0; c < 3; c++) {
            for (int y = 0; y < out_h_up; y++) {
                for (int x = 0; x < out_w_up; x++) {
                    int nchw_idx = c * out_h_up * out_w_up + y * out_w_up + x;
                    float val = float(dst_f16_up[nchw_idx]);
                    tmp_f16_up[y * out_w_up * 3 + x * 3 + c] = (unsigned char)std::round(std::clamp(val, 0.0f, 1.0f) * 255.0f);
                }
            }
        }

        char path_f16_up[256];
        snprintf(path_f16_up, sizeof(path_f16_up), "%sf16_up_nchw.png", output_folder);
        stbi_write_png(path_f16_up, out_w_up, out_h_up, 3, tmp_f16_up, out_w_up * 3);

        printf("f16 (NCHW output UPSCALE): saved %s\n", path_f16_up);

        delete[] src_f16_nchw;
        delete[] dst_f16_up;
        delete[] tmp_f16_up;
    }    
    }
    
    // Cleanup original image
    stbi_image_free(img_data);

    printf("\nAll four functions tested. Check output folder.\n");
    return 0;
}