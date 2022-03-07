#pragma once

#include "Types.h"

using namespace TRT::Types;


size_t getSizeByDim(const nvinfer1::Dims& dims);
std::vector< std::string > getClassNames(const std::string& names_file);

int compare_detection_scores(const void* a, const void* b);
float box_iou(detection a, detection b);
void nms(detection* dets, int &total, float nms_thresh);
detection calculateBox(detection a, float imageWidth, float imageHeight);
detection checkBorders(detection det);


/**
 * @brief Check Cuda codes and address the line that breaks if sth goes wrong
 * 
 */
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define CUDA_FREE(x) 		if(x != NULL) { cudaFree(x); x = NULL; }


/**
 * @brief A unique pointer for TensorRT objects, which also contains the destroyer
 * 
 */
struct TRTDestroy
{
    template <class T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};
template <class T>
using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>;