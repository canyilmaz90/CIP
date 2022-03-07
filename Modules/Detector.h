#pragma once

#include "ModuleTools.h"

using namespace TRT::Types;

namespace TRT
{
    class Detector
    {
    public:
        Detector();
        ~Detector();

        void initializeModuleTools(nlohmann::json config,
                                   gLogger gLogger,
                                   sharedV<sharedQ<shared<Image>>> outQueues,
                                   sharedQ<shared<Info>> infoQueue);
    
    private:
        void initDistributor(sharedV<sharedQ<shared<Image>>> outQueues,
                            sharedQ<shared<Info>> infoQueue);
        void initInferenceModel(nlohmann::json config, gLogger gLogger);
        void initBatchMaker();
    
    private:
        shared<BatchMaker> _BatchMaker;
        shared<InferenceModel> _Model;
        shared<DetectionDistributor> _Distributor;

        ImageType _InputType;
        ImageType _OutputType;
        InfoType _InfoType;

        int _BatchSize;
    };
}