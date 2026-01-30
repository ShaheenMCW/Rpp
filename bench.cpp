/*
 * Resize Nearest Neighbor Benchmark - Single Image Version
 * Compares custom implementation vs OpenCV for a single image
 * Tests all 4 data types: U8, F32, I8, F16
 */

#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstring>

// OpenCV headers
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// SIMD headers
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#endif

// Half precision library
#if __has_include(<half/half.hpp>)
    #include <half/half.hpp>
#else
    #include <half.hpp>
#endif
using halfhpp = half_float::half;

using namespace std;
using namespace cv;
using namespace chrono;

// Configuration
#define NUM_ITERATIONS 100

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

// ==================== Benchmark Helper Functions ====================

double benchmarkCustomResize_U8(const Mat& srcImg, int targetWidth, int targetHeight, bool saveOutput = false)
{
    // Prepare buffers
    Rpp8u* dstBuffer = new Rpp8u[targetHeight * targetWidth * 3];
    
    // Setup descriptors
    RpptDesc srcDesc, dstDesc;
    srcDesc.n = 1; srcDesc.c = 3; srcDesc.h = srcImg.rows; srcDesc.w = srcImg.cols;
    srcDesc.layout = RpptLayout::NHWC;
    srcDesc.strides.nStride = srcImg.rows * srcImg.cols * 3;
    srcDesc.strides.hStride = srcImg.cols * 3;
    srcDesc.strides.wStride = 3;
    srcDesc.strides.cStride = 1;
    
    dstDesc = srcDesc;
    dstDesc.h = targetHeight; dstDesc.w = targetWidth;
    dstDesc.strides.nStride = targetHeight * targetWidth * 3;
    dstDesc.strides.hStride = targetWidth * 3;
    
    RpptROI roi;
    roi.xywhROI.xy.x = 0; roi.xywhROI.xy.y = 0;
    roi.xywhROI.roiWidth = srcImg.cols; roi.xywhROI.roiHeight = srcImg.rows;
    
    RpptImagePatch dstSize = {(Rpp32u)targetWidth, (Rpp32u)targetHeight};
    RppLayoutParams layoutParams;
    layoutParams.layout = RpptLayout::NHWC;
    layoutParams.bufferMultiplier = 3;
    
    // Warmup
    for(int i = 0; i < 10; i++)
        resize_nn_u8_u8_host_tensor(srcImg.data, &srcDesc, dstBuffer, &dstDesc, &dstSize, &roi, XYWH, layoutParams);
    
    // Benchmark
    auto start = high_resolution_clock::now();
    for(int i = 0; i < NUM_ITERATIONS; i++)
        resize_nn_u8_u8_host_tensor(srcImg.data, &srcDesc, dstBuffer, &dstDesc, &dstSize, &roi, XYWH, layoutParams);
    auto end = high_resolution_clock::now();
    
    // Save output if requested
    if(saveOutput) {
        Mat outputImg(targetHeight, targetWidth, CV_8UC3, dstBuffer);
        imwrite("custom_outputs/u8_" + to_string(targetWidth) + "x" + to_string(targetHeight) + ".jpg", outputImg);
    }
    
    delete[] dstBuffer;
    return duration_cast<microseconds>(end - start).count() / (double)NUM_ITERATIONS / 1000.0;
}

double benchmarkCustomResize_F32(const Mat& srcImg, int targetWidth, int targetHeight, bool saveOutput = false)
{
    // Convert to F32
    int srcSize = srcImg.rows * srcImg.cols * 3;
    Rpp32f* srcF32 = new Rpp32f[srcSize];
    for(int i = 0; i < srcSize; i++)
        srcF32[i] = srcImg.data[i] / 255.0f;
    
    Rpp32f* dstBuffer = new Rpp32f[targetHeight * targetWidth * 3];
    
    // Setup descriptors
    RpptDesc srcDesc, dstDesc;
    srcDesc.n = 1; srcDesc.c = 3; srcDesc.h = srcImg.rows; srcDesc.w = srcImg.cols;
    srcDesc.layout = RpptLayout::NHWC;
    srcDesc.strides.nStride = srcImg.rows * srcImg.cols * 3;
    srcDesc.strides.hStride = srcImg.cols * 3;
    srcDesc.strides.wStride = 3;
    srcDesc.strides.cStride = 1;
    
    dstDesc = srcDesc;
    dstDesc.h = targetHeight; dstDesc.w = targetWidth;
    dstDesc.strides.nStride = targetHeight * targetWidth * 3;
    dstDesc.strides.hStride = targetWidth * 3;
    
    RpptROI roi;
    roi.xywhROI.xy.x = 0; roi.xywhROI.xy.y = 0;
    roi.xywhROI.roiWidth = srcImg.cols; roi.xywhROI.roiHeight = srcImg.rows;
    
    RpptImagePatch dstSize = {(Rpp32u)targetWidth, (Rpp32u)targetHeight};
    RppLayoutParams layoutParams;
    layoutParams.layout = RpptLayout::NHWC;
    layoutParams.bufferMultiplier = 3;
    
    // Warmup
    for(int i = 0; i < 10; i++)
        resize_nn_f32_f32_host_tensor(srcF32, &srcDesc, dstBuffer, &dstDesc, &dstSize, &roi, XYWH, layoutParams);
    
    // Benchmark
    auto start = high_resolution_clock::now();
    for(int i = 0; i < NUM_ITERATIONS; i++)
        resize_nn_f32_f32_host_tensor(srcF32, &srcDesc, dstBuffer, &dstDesc, &dstSize, &roi, XYWH, layoutParams);
    auto end = high_resolution_clock::now();
    
    // Save output if requested
    if(saveOutput) {
        Mat outputImg(targetHeight, targetWidth, CV_8UC3);
        for(int i = 0; i < targetHeight * targetWidth * 3; i++)
            outputImg.data[i] = (uchar)(dstBuffer[i] * 255.0f);
        imwrite("custom_outputs/f32_" + to_string(targetWidth) + "x" + to_string(targetHeight) + ".jpg", outputImg);
    }
    
    delete[] srcF32;
    delete[] dstBuffer;
    return duration_cast<microseconds>(end - start).count() / (double)NUM_ITERATIONS / 1000.0;
}

double benchmarkCustomResize_I8(const Mat& srcImg, int targetWidth, int targetHeight, bool saveOutput = false)
{
    // Convert to I8
    int srcSize = srcImg.rows * srcImg.cols * 3;
    Rpp8s* srcI8 = new Rpp8s[srcSize];
    for(int i = 0; i < srcSize; i++)
        srcI8[i] = (Rpp8s)(srcImg.data[i] - 128);
    
    Rpp8s* dstBuffer = new Rpp8s[targetHeight * targetWidth * 3];
    
    // Setup descriptors
    RpptDesc srcDesc, dstDesc;
    srcDesc.n = 1; srcDesc.c = 3; srcDesc.h = srcImg.rows; srcDesc.w = srcImg.cols;
    srcDesc.layout = RpptLayout::NHWC;
    srcDesc.strides.nStride = srcImg.rows * srcImg.cols * 3;
    srcDesc.strides.hStride = srcImg.cols * 3;
    srcDesc.strides.wStride = 3;
    srcDesc.strides.cStride = 1;
    
    dstDesc = srcDesc;
    dstDesc.h = targetHeight; dstDesc.w = targetWidth;
    dstDesc.strides.nStride = targetHeight * targetWidth * 3;
    dstDesc.strides.hStride = targetWidth * 3;
    
    RpptROI roi;
    roi.xywhROI.xy.x = 0; roi.xywhROI.xy.y = 0;
    roi.xywhROI.roiWidth = srcImg.cols; roi.xywhROI.roiHeight = srcImg.rows;
    
    RpptImagePatch dstSize = {(Rpp32u)targetWidth, (Rpp32u)targetHeight};
    RppLayoutParams layoutParams;
    layoutParams.layout = RpptLayout::NHWC;
    layoutParams.bufferMultiplier = 3;
    
    // Warmup
    for(int i = 0; i < 10; i++)
        resize_nn_i8_i8_host_tensor(srcI8, &srcDesc, dstBuffer, &dstDesc, &dstSize, &roi, XYWH, layoutParams);
    
    // Benchmark
    auto start = high_resolution_clock::now();
    for(int i = 0; i < NUM_ITERATIONS; i++)
        resize_nn_i8_i8_host_tensor(srcI8, &srcDesc, dstBuffer, &dstDesc, &dstSize, &roi, XYWH, layoutParams);
    auto end = high_resolution_clock::now();
    
    // Save output if requested
    if(saveOutput) {
        Mat outputImg(targetHeight, targetWidth, CV_8UC3);
        for(int i = 0; i < targetHeight * targetWidth * 3; i++)
            outputImg.data[i] = (uchar)(dstBuffer[i] + 128);
        imwrite("custom_outputs/i8_" + to_string(targetWidth) + "x" + to_string(targetHeight) + ".jpg", outputImg);
    }
    
    delete[] srcI8;
    delete[] dstBuffer;
    return duration_cast<microseconds>(end - start).count() / (double)NUM_ITERATIONS / 1000.0;
}

double benchmarkCustomResize_F16(const Mat& srcImg, int targetWidth, int targetHeight, bool saveOutput = false)
{
    // Convert to F16 NCHW
    int srcSize = srcImg.rows * srcImg.cols;
    Rpp16f* srcF16 = new Rpp16f[3 * srcSize];
    for(int c = 0; c < 3; c++) {
        for(int y = 0; y < srcImg.rows; y++) {
            for(int x = 0; x < srcImg.cols; x++) {
                int nhwc_idx = y * srcImg.cols * 3 + x * 3 + c;
                int nchw_idx = c * srcSize + y * srcImg.cols + x;
                srcF16[nchw_idx] = Rpp16f(srcImg.data[nhwc_idx] / 255.0f);
            }
        }
    }
    
    Rpp16f* dstBuffer = new Rpp16f[3 * targetHeight * targetWidth];
    
    // Setup descriptors for NCHW
    RpptDesc srcDesc, dstDesc;
    srcDesc.n = 1; srcDesc.c = 3; srcDesc.h = srcImg.rows; srcDesc.w = srcImg.cols;
    srcDesc.layout = RpptLayout::NCHW;
    srcDesc.strides.nStride = 3 * srcImg.rows * srcImg.cols;
    srcDesc.strides.cStride = srcImg.rows * srcImg.cols;
    srcDesc.strides.hStride = srcImg.cols;
    srcDesc.strides.wStride = 1;
    
    dstDesc = srcDesc;
    dstDesc.h = targetHeight; dstDesc.w = targetWidth;
    dstDesc.strides.nStride = 3 * targetHeight * targetWidth;
    dstDesc.strides.cStride = targetHeight * targetWidth;
    dstDesc.strides.hStride = targetWidth;
    
    RpptROI roi;
    roi.xywhROI.xy.x = 0; roi.xywhROI.xy.y = 0;
    roi.xywhROI.roiWidth = srcImg.cols; roi.xywhROI.roiHeight = srcImg.rows;
    
    RpptImagePatch dstSize = {(Rpp32u)targetWidth, (Rpp32u)targetHeight};
    RppLayoutParams layoutParams;
    layoutParams.layout = RpptLayout::NCHW;
    layoutParams.bufferMultiplier = 1;
    
    // Warmup
    for(int i = 0; i < 10; i++)
        resize_nn_f16_f16_host_tensor(srcF16, &srcDesc, dstBuffer, &dstDesc, &dstSize, &roi, XYWH, layoutParams);
    
    // Benchmark
    auto start = high_resolution_clock::now();
    for(int i = 0; i < NUM_ITERATIONS; i++)
        resize_nn_f16_f16_host_tensor(srcF16, &srcDesc, dstBuffer, &dstDesc, &dstSize, &roi, XYWH, layoutParams);
    auto end = high_resolution_clock::now();
    
    // Save output if requested
    if(saveOutput) {
        Mat outputImg(targetHeight, targetWidth, CV_8UC3);
        int dstSize_pixels = targetHeight * targetWidth;
        for(int c = 0; c < 3; c++) {
            for(int y = 0; y < targetHeight; y++) {
                for(int x = 0; x < targetWidth; x++) {
                    int nchw_idx = c * dstSize_pixels + y * targetWidth + x;
                    int nhwc_idx = y * targetWidth * 3 + x * 3 + c;
                    outputImg.data[nhwc_idx] = (uchar)(float(dstBuffer[nchw_idx]) * 255.0f);
                }
            }
        }
        imwrite("custom_outputs/f16_" + to_string(targetWidth) + "x" + to_string(targetHeight) + ".jpg", outputImg);
    }
    
    delete[] srcF16;
    delete[] dstBuffer;
    return duration_cast<microseconds>(end - start).count() / (double)NUM_ITERATIONS / 1000.0;
}

double benchmarkOpenCV_Resize(const Mat& srcImg, int targetWidth, int targetHeight, bool saveOutput = false)
{
    Mat dstImg;
    
    // Warmup
    for(int i = 0; i < 10; i++)
        resize(srcImg, dstImg, Size(targetWidth, targetHeight), 0, 0, INTER_NEAREST);
    
    // Benchmark
    auto start = high_resolution_clock::now();
    for(int i = 0; i < NUM_ITERATIONS; i++)
        resize(srcImg, dstImg, Size(targetWidth, targetHeight), 0, 0, INTER_NEAREST);
    auto end = high_resolution_clock::now();
    
    // Save output if requested
    if(saveOutput) {
        imwrite("opencv_outputs/opencv_" + to_string(targetWidth) + "x" + to_string(targetHeight) + ".jpg", dstImg);
    }
    
    return duration_cast<microseconds>(end - start).count() / (double)NUM_ITERATIONS / 1000.0;
}

// ==================== Main ====================
int main(int argc, char** argv)
{
    // IMMEDIATE OUTPUT TEST
    printf("=== PROGRAM STARTED ===\n");
    fflush(stdout);
    
    printf("Number of arguments: %d\n", argc);
    for(int i = 0; i < argc; i++) {
        printf("Arg[%d]: %s\n", i, argv[i]);
    }
    fflush(stdout);
    
    // Force console output to flush immediately
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    
    printf("=== CHECKING ARGUMENTS ===\n");
    fflush(stdout);
    
    // Parse command line arguments
    if(argc < 4) {
        fprintf(stderr, "ERROR: Not enough arguments!\n");
        fprintf(stderr, "Usage: %s <image_path> <target_width> <target_height>\n", argv[0]);
        fprintf(stderr, "Example: %s input.jpg 144 144\n", argv[0]);
        fprintf(stderr, "\nPress Enter to exit...\n");
        fflush(stderr);
        getchar();
        return -1;
    }
    
    string imagePath = argv[1];
    int targetWidth = atoi(argv[2]);
    int targetHeight = atoi(argv[3]);
    
    printf("=== ARGUMENTS PARSED ===\n");
    printf("Image path: %s\n", imagePath.c_str());
    printf("Target size: %d x %d\n", targetWidth, targetHeight);
    fflush(stdout);
    
    printf("=== CREATING DIRECTORIES ===\n");
    fflush(stdout);
    
    // Create output directories
    system("if not exist opencv_outputs mkdir opencv_outputs");
    system("if not exist custom_outputs mkdir custom_outputs");
    
    printf("Directories created\n");
    fflush(stdout);
    
    printf("\n==========================================================\n");
    printf("   Resize Nearest Neighbor Benchmark\n");
    printf("   Custom Implementation vs OpenCV\n");
    printf("==========================================================\n\n");
    fflush(stdout);
    
    // Load image
    printf("=== LOADING IMAGE ===\n");
    printf("Path: %s\n", imagePath.c_str());
    fflush(stdout);
    
    Mat srcImg;
    try {
        srcImg = imread(imagePath, IMREAD_COLOR);
        printf("imread() returned, checking if empty...\n");
        fflush(stdout);
    } catch (const std::exception& e) {
        fprintf(stderr, "EXCEPTION during imread: %s\n", e.what());
        fflush(stderr);
        getchar();
        return -1;
    }
    
    if(srcImg.empty()) {
        fprintf(stderr, "\nERROR: Cannot load image: %s\n", imagePath.c_str());
        fprintf(stderr, "Possible reasons:\n");
        fprintf(stderr, "1. File does not exist\n");
        fprintf(stderr, "2. File is not a valid image\n");
        fprintf(stderr, "3. OpenCV DLLs are missing\n");
        fprintf(stderr, "\nPress Enter to exit...\n");
        fflush(stderr);
        getchar();
        return -1;
    }
    
    printf("=== IMAGE LOADED SUCCESSFULLY ===\n");
    printf("Image size: %d x %d\n", srcImg.cols, srcImg.rows);
    printf("Image channels: %d\n", srcImg.channels());
    fflush(stdout);
    
    printf("\nInput Image:     %s\n", imagePath.c_str());
    printf("Input Size:      %d x %d\n", srcImg.cols, srcImg.rows);
    printf("Target Size:     %d x %d\n", targetWidth, targetHeight);
    printf("Iterations:      %d\n\n", NUM_ITERATIONS);
    fflush(stdout);
    
    string operation = (targetWidth > srcImg.cols || targetHeight > srcImg.rows) ? "UPSCALE" : "DOWNSCALE";
    printf("Operation:       %s\n\n", operation.c_str());
    fflush(stdout);
    
    printf("==========================================================\n");
    printf("   Running Benchmarks...\n");
    printf("==========================================================\n\n");
    fflush(stdout);
    
    // Run benchmarks and save outputs
    double timeOpenCV = 0, timeU8 = 0, timeF32 = 0, timeI8 = 0, timeF16 = 0;
    
    try {
        printf("[1/5] Starting OpenCV Resize...\n");
        fflush(stdout);
        timeOpenCV = benchmarkOpenCV_Resize(srcImg, targetWidth, targetHeight, true);
        printf("OpenCV Done! (%f ms)\n", timeOpenCV);
        fflush(stdout);
        
        printf("[2/5] Starting Custom U8 Resize...\n");
        fflush(stdout);
        timeU8 = benchmarkCustomResize_U8(srcImg, targetWidth, targetHeight, true);
        printf("U8 Done! (%f ms)\n", timeU8);
        fflush(stdout);
        
        printf("[3/5] Starting Custom F32 Resize...\n");
        fflush(stdout);
        timeF32 = benchmarkCustomResize_F32(srcImg, targetWidth, targetHeight, true);
        printf("F32 Done! (%f ms)\n", timeF32);
        fflush(stdout);
        
        printf("[4/5] Starting Custom I8 Resize...\n");
        fflush(stdout);
        timeI8 = benchmarkCustomResize_I8(srcImg, targetWidth, targetHeight, true);
        printf("I8 Done! (%f ms)\n", timeI8);
        fflush(stdout);
        
        printf("[5/5] Starting Custom F16 Resize...\n");
        fflush(stdout);
        timeF16 = benchmarkCustomResize_F16(srcImg, targetWidth, targetHeight, true);
        printf("F16 Done! (%f ms)\n\n", timeF16);
        fflush(stdout);
        
        // Print results table
        printf("==========================================================\n");
        printf("   BENCHMARK RESULTS\n");
        printf("==========================================================\n\n");
        
        printf("+------------------+--------------+--------------+\n");
        printf("| Implementation   | Time (ms)    | Speedup      |\n");
        printf("+------------------+--------------+--------------+\n");
        
        printf("| %-16s | %12.4f | %12s |\n", "OpenCV", timeOpenCV, "baseline");
        printf("| %-16s | %12.4f | %11.2fx |\n", "Custom U8", timeU8, timeOpenCV/timeU8);
        printf("| %-16s | %12.4f | %11.2fx |\n", "Custom F32", timeF32, timeOpenCV/timeF32);
        printf("| %-16s | %12.4f | %11.2fx |\n", "Custom I8", timeI8, timeOpenCV/timeI8);
        printf("| %-16s | %12.4f | %11.2fx |\n", "Custom F16", timeF16, timeOpenCV/timeF16);
        
        printf("+------------------+--------------+--------------+\n\n");
        fflush(stdout);
        
        // Print output locations
        printf("==========================================================\n");
        printf("   OUTPUT FILES\n");
        printf("==========================================================\n\n");
        
        printf("OpenCV outputs:  opencv_outputs\\opencv_%dx%d.jpg\n", targetWidth, targetHeight);
        printf("Custom outputs:  custom_outputs\\u8_%dx%d.jpg\n", targetWidth, targetHeight);
        printf("                 custom_outputs\\f32_%dx%d.jpg\n", targetWidth, targetHeight);
        printf("                 custom_outputs\\i8_%dx%d.jpg\n", targetWidth, targetHeight);
        printf("                 custom_outputs\\f16_%dx%d.jpg\n\n", targetWidth, targetHeight);
        
        printf("==========================================================\n");
        printf("Benchmark completed successfully!\n");
        printf("==========================================================\n");
        fflush(stdout);
        
    } catch (const std::exception& e) {
        fprintf(stderr, "\nEXCEPTION during benchmark: %s\n", e.what());
        fprintf(stderr, "Press Enter to exit...\n");
        fflush(stderr);
        getchar();
        return -1;
    } catch (...) {
        fprintf(stderr, "\nUNKNOWN EXCEPTION during benchmark!\n");
        fprintf(stderr, "Press Enter to exit...\n");
        fflush(stderr);
        getchar();
        return -1;
    }
    
    printf("\n=== PROGRAM COMPLETED ===\n");
    printf("Press Enter to exit...\n");
    fflush(stdout);
    getchar();
    
    return 0;
}
