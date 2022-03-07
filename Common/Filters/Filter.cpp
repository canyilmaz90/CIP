#include "Filter.h"

using namespace TRT;

Filter::Filter()
{
    _IsRunning.exchange(false);
}
Filter::~Filter()
{

}

void Filter::setConfig(nlohmann::json config)
{
    _Config = config;
}

void Filter::start()
{
    _IsRunning.exchange(true);
    _Thread = std::thread([this] {run(); });
}

void Filter::stop()
{
    _IsRunning.exchange(false);
}

void Filter::waitUntilFinished()
{
    while (_IsRunning)
    {
        SleepHere(100);
    }
}
