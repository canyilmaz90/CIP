#include "Detector.h"

using namespace TRT;

void Detector::initDistributor(sharedV<sharedQ<shared<Image>>> outQueues,
                            sharedQ<shared<Info>> infoQueue)
{
    _Distributor = std::make_shared<DetectionDistributor>();
    _Distributor->setOutputQueues(outQueues);
    _Distributor->setInfoOutputQueue(infoQueue);
    _Distributor->start();
}

void Detector::initInferenceModel(nlohmann::json config, gLogger gLogger)
{
    _Model = std::make_shared<InferenceModel>(config, gLogger);
    sharedQ<sharedV<shared<Image>>> QDistributorInput = _Distributor->getInputQueue();
    _Model->setOutputQueue(QDistributorInput);
    _Model->start();
}

void Detector::initBatchMaker()
{
    _BatchMaker = std::make_shared<BatchMaker>();
    sharedQ<sharedV<shared<Image>>> QModelInput = _Model->getInputQueue();
    _BatchMaker->setOutputQueue(QModelInput);
    _BatchMaker->start();
}

void Detector::initializeModuleTools(nlohmann::json config,
                                   gLogger gLogger,
                                   sharedV<sharedQ<shared<Image>>> outQueues,
                                   sharedQ<shared<Info>> infoQueue)
{
    this->initDistributor(outQueues, infoQueue);
    this->initInferenceModel(config, gLogger);
    this->initBatchMaker();
}
