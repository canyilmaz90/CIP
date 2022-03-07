#include "BatchMaker.h"

using namespace TRT;

BatchMaker::BatchMaker()
{
    sharedQ<shared<Image>> Q = make_Q<shared<Image>>();
    this->setInputQueue(Q);
}

void BatchMaker::setBatchSize(int batch_size)
{
    _BatchSize = batch_size;
}

void BatchMaker::pushBatch()
{
    sharedV<shared<Image>> batch(_Batch);
    _OutputQueue->push(batch);
    _Batch->clear();
}

bool BatchMaker::isTimedOut()
{
    return (calcTimeInterval(_Start, getCurrentTime()) >= _TimeOut);
}

void BatchMaker::timeoutOperation()
{
    if (_Batch->size() > 0)
    {
        pushBatch();
    }
    else
    {
        setStart();
    }
}

void BatchMaker::routine()
{
    if (_InputQueue->size() == 0)
    {
        if (isTimedOut())
        {
            timeoutOperation();
        }
    }
    else
    {
        shared<Image> image = std::make_shared<Image>();
        image = _InputQueue->pop();
        _Batch->push_back(image);
        if (_Batch->size() == 1)
        {
            setStart();
        }
        else if (_Batch->size() == _BatchSize)
        {
            pushBatch();
        }
    }
}

void BatchMaker::run()
{
    while(_IsRunning)
    {
        try
        {
            routine();
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
    }
}