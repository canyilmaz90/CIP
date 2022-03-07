#include "FrameInfo.h"
#include "Utils.h"

using namespace TRT;

FrameInfo::FrameInfo()
{
    this->setTimeStamp();
    this->setType(InfoType::FRAME);
}
FrameInfo::~FrameInfo() {}

void FrameInfo::setTimeStamp()
{
    this->_TimeStamp = getCurrentTime();
}

timestamp FrameInfo::time_stamp() const
{
    return this->_TimeStamp;
}

void FrameInfo::setResolution(resolution& res)
{
    this->_Resolution = res;
}

resolution FrameInfo::_resolution() const
{
    return _Resolution;
}