#include "CuProc.h"

//-----------------------------------------------------------------------------------
// gpuNormalize
//-----------------------------------------------------------------------------------
__global__ void gpuNormalizeKernel(float3* input, float3* output, int width, int height, float scaling_factor)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width )
		return; 

	if( y >= height )
		return;

    const int pixel = y * width + x;
	const float3 px = input[pixel];

	#define rescale(v) (v * scaling_factor)

	output[pixel] = make_float3(rescale(px.x),
							    rescale(px.y),
							    rescale(px.z));
}

cudaError_t cudaNormalize(void* input, void* output,
						    size_t  width,  size_t height,
                            const float scaling_factor)
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0  )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(32,8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuNormalizeKernel<<<gridDim, blockDim>>>((float3*)input, (float3*)output, width, height, scaling_factor);

	return CUDA(cudaGetLastError());
}
