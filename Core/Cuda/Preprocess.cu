#include <CuProc.h>
#include <iostream>


// ---------------------
// #### RESIZE IMAGE ####
// ---------------------
bool imageResize_8u_C3R(void *src, int srcWidth, int srcHeight, void *dst, int dstWidth, int dstHeight)
{
    NppiSize oSrcSize;
    oSrcSize.width = srcWidth;
    oSrcSize.height = srcHeight;
    int nSrcStep = srcWidth * 3;

    NppiRect oSrcROI;
    oSrcROI.x = 0;
    oSrcROI.y = 0;
    oSrcROI.width = srcWidth;
    oSrcROI.height = srcHeight;

    int nDstStep = dstWidth * 3;
    NppiRect oDstROI;
    oDstROI.x = 0;
    oDstROI.y = 0;
    oDstROI.width = dstWidth;
    oDstROI.height = dstHeight;

    // Scale Factor
    double nXFactor = double(dstWidth) / (oSrcROI.width);
    double nYFactor = double(dstHeight) / (oSrcROI.height);

    // Scaled X/Y  Shift
    double nXShift = - oSrcROI.x * nXFactor ;
    double nYShift = - oSrcROI.y * nYFactor;
    int eInterpolation = NPPI_INTER_LINEAR;
    if (nXFactor >= 1.f || nYFactor >= 1.f)
        eInterpolation = NPPI_INTER_LINEAR;

    NppStatus ret = nppiResizeSqrPixel_8u_C3R((const Npp8u *)src, oSrcSize, nSrcStep, oSrcROI, (Npp8u *)dst,
                         nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation );
    if(ret != NPP_SUCCESS) {
        std::cout << "Resize failed with code: " << ret << std::endl;
        return false;
    }

    return true;
}


void imageResize_32f_C3R(void *src, int srcWidth, int srcHeight, void *dst, int dstWidth, int dstHeight)
{
    NppiSize oSrcSize;
    oSrcSize.width = srcWidth;
    oSrcSize.height = srcHeight;
    int nSrcStep = srcWidth * 3 * sizeof(float);

    NppiRect oSrcROI;
    oSrcROI.x = 0;
    oSrcROI.y = 0;
    oSrcROI.width = srcWidth;
    oSrcROI.height = srcHeight;

    int nDstStep = dstWidth * 3 * sizeof(float);
    NppiRect oDstROI;
    oDstROI.x = 0;
    oDstROI.y = 0;
    oDstROI.width = dstWidth;
    oDstROI.height = dstHeight;
    double nXFactor = double(dstWidth) / (oSrcROI.width);
    double nYFactor = double(dstHeight) / (oSrcROI.height);
    double nXShift = 0;
    double nYShift = 0;
    int eInterpolation = NPPI_INTER_SUPER;
    if (nXFactor >= 1.f || nYFactor >= 1.f)
        eInterpolation = NPPI_INTER_LANCZOS;

    NppStatus ret = nppiResizeSqrPixel_32f_C3R((const Npp32f *)src, oSrcSize, nSrcStep, oSrcROI, (Npp32f *)dst,
                         nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation );
    if(ret != NPP_SUCCESS) {
        std::cout << "Resize failed with code: " << ret << "\n" << std::endl;
    }
}

// ---------------------
// #### BGR2RGB ####
// ---------------------
__global__ void convertBGR2RGBfloatKernel(uchar3 *src, float3 *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    uchar3 color = src[y * width + x];
    dst[y * width + x] = make_float3(color.z, color.y, color.x);
}

void convertBGR2RGBfloat(void *src, void *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    convertBGR2RGBfloatKernel<<<grids, blocks>>>((uchar3 *)src, (float3 *)dst, width, height);
}

// ---------------------
// #### NORMALIZATION ####
// ---------------------
__global__ void imageNormalizationPositiveKernel(float3 *ptr, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    float3 color = ptr[y * width + x];
    color.x = color.x / 255.0;
    color.y = color.y / 255.0;
    color.z = color.z / 255.0;

    ptr[y * width + x] = make_float3(color.x, color.y, color.z);
}

void imageNormalizationPositive(void *ptr, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    imageNormalizationPositiveKernel<<<grids, blocks>>>((float3 *)ptr, width, height);
}

__global__ void imageNormalizationKernel(float3 *ptr, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    float3 color = ptr[y * width + x];
    color.x = (color.x - 127.5) * 0.0078125;
    color.y = (color.y - 127.5) * 0.0078125;
    color.z = (color.z - 127.5) * 0.0078125;

    ptr[y * width + x] = make_float3(color.x, color.y, color.z);
}

void imageNormalization(void *ptr, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    imageNormalizationKernel<<<grids, blocks>>>((float3 *)ptr, width, height);
}

// ---------------------
// #### SPLIT ####
// ---------------------
__global__ void imageSplitKernel(float3 *ptr, float *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    float3 color = ptr[y * width + x];

    dst[y * width + x] = color.x;
    dst[y * width + x + width * height] = color.y;
    dst[y * width + x + width * height * 2] = color.z;
}

void imageSplit(const void *src, float *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    imageSplitKernel<<<grids, blocks>>>((float3 *)src, (float *)dst, width, height);
}

__global__ void imageSplit_8UC3Kernel(uchar3 *ptr, unsigned char *dst1, unsigned char *dst2, unsigned char *dst3, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    uchar3 color = ptr[y * width + x];

    dst1[y * width + x] = color.x;
    dst2[y * width + x] = color.y;
    dst3[y * width + x] = color.z;
}

void imageSplit_8UC3(const void *src, unsigned char *dst1, unsigned char *dst2, unsigned char *dst3, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    imageSplit_8UC3Kernel<<<grids, blocks>>>((uchar3 *)src, (unsigned char *)dst1, (unsigned char *)dst2, (unsigned char *)dst3, width, height);
}

__global__ void imageCombine_8UC3Kernel(unsigned char *src1, unsigned char *src2, unsigned char *src3, uchar3 *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    uchar3 color;
    color.x = src1[y * width + x];
    color.y = src2[y * width + x];
    color.z = src3[y * width + x];

    dst[y * width + x] = color;
}

void imageCombine_8UC3(const void *src1, const void *src2, const void *src3, void *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    imageCombine_8UC3Kernel<<<grids, blocks>>>((unsigned char *)src1, (unsigned char *)src2, (unsigned char *)src3, (uchar3 *)dst, width, height);
}


// Npp32s *histDevice = 0;
// Npp8u *pDeviceBuffer;

// Npp32s  *lutDevice  = 0;
// Npp32s  *lvlsDevice = 0;

// const int binCount = 256;
// // levels array has one more element
// const int levelCount = binCount + 1;




