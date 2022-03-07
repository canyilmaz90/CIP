#include "Engine.h"

using namespace TRT;

Engine::Engine()
{
}

Engine::~Engine()
{
}

void Engine::init(std::string engineFile, gLogger gLogger, int batch_size)
{
    try
    {
        deserializeTRTEngine(engineFile, gLogger);
        checkImplicitBatch(batch_size);
        getBindings();
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    catch (std::exception &e)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, e.what());
    }   
}

void Engine::deserializeTRTEngine(std::string engineFile, gLogger gLogger)
{
    try
    {
        std::cout << "[INFO] Loading Engine:" << engineFile << std::endl;
        std::vector<char> TRTModelStream;
        size_t size{0};

        std::ifstream file(engineFile, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            TRTModelStream.resize(size);
            file.read(TRTModelStream.data(), size);
            file.close();
        }
        
        runtime = TRTUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        assert(runtime != nullptr);
        engine = TRTUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(TRTModelStream.data(), size, nullptr));
        assert(engine != nullptr);
        context = TRTUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        assert(context != nullptr);
    }
    catch (std::exception &e)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, e.what());
    }
        
}

void Engine::checkImplicitBatch(int batch_size)
{
    nvinfer1::Dims indims = context->getBindingDimensions(0);
    if (indims.d[0] == -1)
    {
        indims.d[0] = batch_size;
        context->setBindingDimensions(0, indims);
    }
    
}

void Engine::getBindings()
{
    buffers.resize(engine->getNbBindings()); // buffers for input and output data
    for (size_t i = 0; i < engine->getNbBindings(); ++i)
    {
        auto binding_size = getSizeByDim(context->getBindingDimensions(i)) * sizeof(float);
        CUDA_CHECK(cudaMalloc(&buffers[i], binding_size));
        if (engine->bindingIsInput(i))
        {
            CUDA_CHECK(cudaMemset(buffers[i], 0, binding_size));
            input_dims.emplace_back(context->getBindingDimensions(i));
        }
        else
        {
            output_dims.emplace_back(context->getBindingDimensions(i));
        }
    }
    if (input_dims.empty() || output_dims.empty())
    {
        std::cerr << "Expect at least one input and one output for network\n";
    }
}

