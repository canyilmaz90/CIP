#include <CuProc.h>


// __device__ float lerp1d(int a, int b, float w)
// {
//     if(b>a){
//         return a + w*(b-a);
//     }
//     else{
//         return b + w*(a-b);
//     }
// }

// __device__ float lerp2d(int f00, int f01, int f10, int f11,
//                         float centroid_h, float centroid_w )
// {
//     centroid_w = (1 + lroundf(centroid_w) - centroid_w)/2;
//     centroid_h = (1 + lroundf(centroid_h) - centroid_h)/2;
    
//     float r0, r1, r;
//     r0 = lerp1d(f00,f01,centroid_w);
//     r1 = lerp1d(f10,f11,centroid_w);

//     r = lerp1d(r0, r1, centroid_h); //+ 0.00001
//     // printf("re: %f, %f | %f, %f | %f, %f | %f | %d, %d, %d, %d \n", centroid_x , centroid_y, centroid_x_re, centroid_y_re, r0, r1, r, f00, f01, f10, f11);
//     return r;
// }


// __global__ void ResizeKernel(unsigned char* src_img, unsigned char* dst_img, 
//     const int src_h, const int src_w, 
//     const int dst_h, const int dst_w,
//     const float stride_h, const float stride_w)
// {
//     /* 
//     Input: 
//         src_img - NHWC
//         channel C, default = 3 
    
//     Output:
//         dst_img - NHWC

//     */

//     // int const N = gridDim.y; // batch size
//     int const n = blockIdx.y; // batch number
//     int const C = gridDim.z; // channel 
//     int const c = blockIdx.z; // channel number
//     long idx = n * blockDim.x * gridDim.x * C + 
//               threadIdx.x * gridDim.x * C +
//               blockIdx.x * C+
//               c;
    
//     // some overhead threads in each image process
//     // when thread idx in one image exceed one image size return;
//     if (idx%(blockDim.x * gridDim.x * C) >= dst_h* dst_w * C)
//     {
//         return;
//     } 

//     /*
//     Now implementation : 
//     ( (1024 * int(dstSize/3/1024)+1) - (src_h * src_w) )* N
//     = overhead * N times
    
//     to do: put the batch into gridDim.x
//     dim3 dimGrid(int(dstSize*batch/3/1024)+1,1,3);

//     */

//     int H = dst_h;
//     int W = dst_w;

//     int img_coor = idx % (dst_h*dst_w*C); //coordinate of one image, not idx of batch image
//     int h = img_coor / (W*C); // dst idx 
//     int w = img_coor % (W*C)/C; // dst idx

//     float centroid_h, centroid_w;  
//     centroid_h = stride_h * (h + 0.5); // h w c -> x, y, z : 1080 , 1920 , 3
//     centroid_w = stride_w * (w + 0.5); // 

//     // unsigned long = 4,294,967,295 , up to (1080p,RGB)*600 imgs
//     long f00,f01,f10,f11;

//     int src_h_idx = lroundf(centroid_h)-1;
//     int src_w_idx = lroundf(centroid_w)-1;
//     if (src_h_idx<0){src_h_idx=0;}
//     if (src_w_idx<0){src_w_idx=0;}
//     // printf("h:%d w:%d\n",src_h_idx,src_w_idx);
//     // printf("src_h_idx:%d , h: %d | src_w_idx:%d , w: %d\n",src_h_idx,h,src_w_idx,w);

//     // idx = NHWC = n*(HWC) + h*(WC) + w*C + c;
//     f00 = n * src_h * src_w * C + 
//           src_h_idx * src_w * C + 
//           src_w_idx * C +
//           c;
//     f01 = n * src_h * src_w * C +
//           src_h_idx * src_w * C +
//           (src_w_idx+1) * C +
//           c;
//     f10 = n * src_h * src_w * C +
//           (src_h_idx+1) * src_w * C +
//           src_w_idx * C +
//           c;
//     f11 = n * src_h * src_w * C + 
//           (src_h_idx+1) * src_w * C +
//           (src_w_idx+1) * C +
//           c;
//     int rs;   
//     if (int(f10/ (src_h * src_w * C)) > n ){
//         centroid_w = (1 + lroundf(centroid_w) - centroid_w)/2;
//         rs = lroundf(lerp1d(f00,f01,centroid_w));
//     }else{
//         rs = lroundf(lerp2d(src_img[f00], src_img[f01], src_img[f10], src_img[f11], 
//             centroid_h, centroid_w));
//     }
    
//     long dst_idx =  n * (H * W * C) + 
//                     h * (W * C) +
//                     w * C +
//                     c;

//     dst_img[dst_idx] = (unsigned char)rs;
// }

// cudaError_t cudaResize(void *input, int inputWidth, int inputHeight, void *output, int outputWidth, int outputHeight)
// {
// 	if( !input || !output )
// 		return CUDA(cudaErrorInvalidDevicePointer);

// 	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
// 		return CUDA(cudaErrorInvalidValue);
    
    
//     const dim3 blockDim(32, 8, 1);
// 	const dim3 gridDim(iDivUp(outputWidth, blockDim.x), iDivUp(outputHeight, blockDim.y), 1);
    
//     float stride_w = (float)inputWidth / outputWidth;
//     float stride_h = (float)inputHeight / outputHeight;

//     ResizeKernel<<<gridDim, blockDim>>>((unsigned char *)input, (unsigned char *)output,
//                                         inputHeight, inputWidth,
//                                         outputHeight, outputWidth,
//                                         stride_h, stride_w);

//     return CUDA(cudaGetLastError());
// }


//-----------------------------------------------------------------------------------
// simple resize
//-----------------------------------------------------------------------------------
__global__ void gpuResize(float2 scale, uchar3* input, int iWidth, uchar3* output, int oWidth, int oHeight)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const uchar3 px = input[ dy * iWidth + dx ];

	output[y*oWidth+x] = px;
}

// launchResize
cudaError_t cudaResize(void* input, int inputWidth, int inputHeight, void* output, int outputWidth, int outputHeight)
{
	if( !input || !output )
		return CUDA(cudaErrorInvalidDevicePointer);

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return CUDA(cudaErrorInvalidValue);

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							     float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuResize<<<gridDim, blockDim>>>(scale, (uchar3*)input, inputWidth, (uchar3*)output, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}

//-----------------------------------------------------------------------------------
// Main function to test with nvcc,   , 0, stream
//-----------------------------------------------------------------------------------
// int main()
// {
//     int srcWidth = 1920;
//     int srcHeight = 1080;
//     int srcSIZE = srcWidth * srcHeight * 3;

//     int dstWidth = 256;
//     int dstHeight = 256;
//     int dstSIZE = dstWidth * dstHeight * 3;
    
//     unsigned char host_src[srcSIZE];

//     // init src image
//     for(int i = 0; i < srcSIZE; i++){
//         host_src[i] = i%255;
//     }

// 	void *gpu_frame, *resized;

// 	cudaMalloc(&gpu_frame, srcSIZE);
// 	cudaMemcpy(gpu_frame , host_src , srcSIZE, cudaMemcpyHostToDevice);
// 	cudaMalloc(&resized, dstSIZE);
//     // cudaMemset(resized, 0, dstSIZE);
// 	cudaError_t error = CUDA(cudaResize(gpu_frame, srcWidth, srcHeight, resized, dstWidth, dstHeight));

// 	cudaFree(gpu_frame);
// 	cudaFree(resized);
// 	return 0;
// }