#pragma once

#include "Common.h"

using namespace TRT::Types;

namespace TRT
{
    class BatchMaker : public Process<shared<Image>, sharedV<shared<Image>>>
    {
    public:
        BatchMaker();
        ~BatchMaker();

        bool isTimedOut();
        void timeoutOperation();
        void setBatchSize(int batch_size);
        void setStart();
        void pushBatch();
        void routine();
        void run();


    private:
        int _BatchSize;
        sharedV<shared<Image>> _Batch;
        timestamp _Start;
        double _TimeOut;
    };
}