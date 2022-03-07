#pragma once

#include "Engine.h"
#include "CuProc.h"

using namespace TRT::Types;

namespace TRT
{
    class YOLO : public Engine
    {
    public:
        YOLO(nlohmann::json config, gLogger gLogger);
        ~YOLO();
        void setInputBatch(sharedV<shared<Image>> inputBatch);
        sharedV<shared<Image>> run(sharedV<shared<Image>> inputBatch) override;
    
    private:
        void preprocess() override;
        void postprocess() override;
    
    private:
        sharedV<shared<Image>> _InputBatch;
        std::vector<std::string> _classes;
        float _conf;
        float _nms;
    };
}
    