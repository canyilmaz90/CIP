#include <iostream>
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_geometry_transforms.h>
#include <nppi_support_functions.h>


inline __device__ __host__ int iDivUp( int a, int b )  		{ return (a % b != 0) ? (a / b + 1) : (a / b); }


#define CUDA(x) cudaCheckError((x), __FILE__, __LINE__)
inline cudaError_t cudaCheckError(cudaError_t err, const char* file, int line, bool abort=true)
{
   if (err != cudaSuccess) 
   {
       std::cout << "CudaError: " << cudaGetErrorString(err) << std::endl;
       std::cout << file << ": " << line << std::endl;
       if (abort) exit(err);
   }

   return err;
}


cudaError_t cudaResize(void* input, int inputWidth, int inputHeight, void* output, int outputWidth, int outputHeight);
cudaError_t cudaBGR8ToRGB8(void* input, void* output, size_t width, size_t height);
cudaError_t cudaBGR8ToRGB32(void* input, void* output, size_t width, size_t height);
cudaError_t cudaNormalize(void* input, void* output, size_t  width,  size_t height, const float scaling_factor);
cudaError_t cudaSplit(void* input, float* output, size_t width, size_t height);


bool imageResize_8u_C3R(void *src, int srcWidth, int srcHeight, void *dst, int dstWidth, int dstHeight);
void imageResize_32f_C3R(void *src, int srcWidth, int srcHeight, void *dst, int dstWidth, int dstHeight);

