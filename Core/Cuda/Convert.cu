#include <CuProc.h>


//-----------------------------------------------------------------------------------
// RGB (uint8) <-> BGR (uint8)
//-----------------------------------------------------------------------------------
__global__ void BGR8ToRGB8(uchar3* srcImage, uchar3* dstImage, int width, int height)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if( x >= width )
		return; 

	if( y >= height )
		return;

	const int pixel = y * width + x;

	const uchar3 px = srcImage[pixel];
	
	dstImage[pixel] = make_uchar3(px.z, px.y, px.x);
}

cudaError_t cudaBGR8ToRGB8( void* input, void* output, size_t width, size_t height )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(32,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	BGR8ToRGB8<<<gridDim, blockDim>>>((uchar3*)input, (uchar3*)output, width, height);
	
	return cudaGetLastError();
}


//-----------------------------------------------------------------------------------
// BGR (uint8) <-> RGB (float)
//-----------------------------------------------------------------------------------
__global__ void BGR8ToRGB32(uchar3* srcImage, float3* dstImage, int width, int height)
{
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	const int pixel = y * width + x;

	if( x >= width || y >= height )
		return;

	const uchar3 px = srcImage[pixel];

	const float b = (float) px.x;
	const float g = (float) px.y;
	const float r = (float) px.z;
	
	dstImage[pixel] = make_float3(r, g, b);
}

cudaError_t cudaBGR8ToRGB32(void* input, void* output, size_t width, size_t height)
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(32, 8, 1);
	const dim3 gridDim(iDivUp(width, blockDim.x), iDivUp(height, blockDim.y), 1);

	BGR8ToRGB32<<<gridDim, blockDim>>>((uchar3*)input, (float3*)output, width, height);
	
	return CUDA(cudaGetLastError());
}


//-----------------------------------------------------------------------------------
// Main function to test with nvcc
//-----------------------------------------------------------------------------------
// int main()
// {

//     int height = 256;
//     int width = 256;
//     int SIZE = width * height * 3;
    
//     unsigned char host_src[SIZE];
//     // unsigned char host_dst[SIZE];

//     // init src image
//     for(int i = 0; i < SIZE; i++){
//         host_src[i] = i%255;
//     }

// 	void *gpu_frame, *converted;

// 	cudaMalloc(&gpu_frame, SIZE);
// 	cudaMemcpy(gpu_frame , host_src , SIZE, cudaMemcpyHostToDevice);
// 	cudaMalloc(&converted, SIZE * sizeof(float));
// 	cudaError_t error = CUDA(cudaBGR8ToRGB32(gpu_frame, converted, width, height));
// 	// cudaError_t error = CUDA(cudaBGR8ToRGB8(gpu_frame, converted, width, height));

// 	cudaFree(gpu_frame);
// 	cudaFree(converted);
// 	return 0;
// }