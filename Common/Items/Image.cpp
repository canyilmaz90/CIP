#include "Image.h"
#include "Utils.h"

using namespace TRT;

void Image::setMat(const cv::Mat &mat)
{
    _MatObject = mat.clone();
}

cv::Mat Image::mat() const
{
    return _MatObject;
}

void Image::setDetections(detection* dets)
{
    _Detections = dets;
}

detection* Image::detections() const
{
    return _Detections;
}

void Image::setNumberOfDetections(int num)
{
    _NumberOfDetections = num;
}

int Image::num_of_detections() const
{
    return _NumberOfDetections;
}

void Image::setInfo(const Info &info)
{
    this->_Info = std::make_unique<Info>(info);
}

void Image::setInfo(const FrameInfo &info)
{
    this->_Info = std::make_unique<FrameInfo>(info);
}

void Image::setInfo(const ObjectInfo &info)
{
    this->_Info = std::make_unique<ObjectInfo>(info);
}

Info Image::info() const
{
    return *this->_Info;
}
