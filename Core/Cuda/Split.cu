#include "CuProc.h"


//-----------------------------------------------------------------------------------
// Image Split HWC -> CHW
//-----------------------------------------------------------------------------------
__global__ void SplitKernel(float3* srcImage, float* dstImage, int width, int height)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if( x >= width || y >= height )
		return;

    const int pixel = y * width + x;
    const int area = width * height;
    float3 color = srcImage[pixel];

    dstImage[pixel] = color.x;
    dstImage[pixel + area] = color.y;
    dstImage[pixel + area * 2] = color.z;
}

cudaError_t cudaSplit(void* input, float* output, size_t width, size_t height)
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y));

	SplitKernel<<<gridDim, blockDim>>>((float3*)input, output, width, height);
	
	return CUDA(cudaGetLastError());
}