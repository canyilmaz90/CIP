#include "InferenceModel.h"

using namespace TRT;

InferenceModel::InferenceModel(nlohmann::json config, gLogger gLogger)
{
    this->setConfig(config);
    _Engine = std::make_shared<YOLO>(config, gLogger);

    sharedQ<sharedV<shared<Image>>> Q = make_Q<sharedV<shared<Image>>>();
    this->setInputQueue(Q);
}

void InferenceModel::run()
{
    while(_IsRunning)
    {
        try
        {
            _InputBatch = _InputQueue->pop();
            sharedV<shared<Image>> Output = _Engine->run(_InputBatch);
            _OutputQueue->push(Output);
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
    }      
}

