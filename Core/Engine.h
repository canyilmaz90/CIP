#pragma once

#include "Common.h"
#include "GLogger.h"
#include "ModelUtils.h"

using namespace TRT::Types;

namespace TRT
{
    class Engine
    {
    public:
        Engine();
        ~Engine();
        void init(std::string engineFile, gLogger gLogger, int batch_size);
        virtual sharedV<shared<Image>> run(sharedV<shared<Image>> inputBatch) = 0;

    protected:
        virtual void preprocess() = 0;
        virtual void postprocess() = 0;
    private:
        void deserializeTRTEngine(std::string engineFile, gLogger gLogger);
        void checkImplicitBatch(int batch_size);
        void getBindings();
    
    protected:
        TRTUniquePtr<nvinfer1::IRuntime> runtime;
        TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
        TRTUniquePtr<nvinfer1::IExecutionContext> context{nullptr};
        cudaStream_t stream;
        
        std::vector<void*> buffers;
        std::vector<nvinfer1::Dims> input_dims;
        std::vector<nvinfer1::Dims> output_dims;
    };
}