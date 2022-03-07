#pragma once

#include "Utils.h"
#include "Info.h"

using namespace TRT::Types;

namespace TRT
{
    class Image
    {
    public:
        Image();
        Image(const cv::Mat &mat);
        ~Image();

        void setMat(const cv::Mat &mat);
        void setDetections(detection* dets);
        void setNumberOfDetections(int num);
        void setInfo(const Info &info);
        void setInfo(const FrameInfo &info);
        void setInfo(const ObjectInfo &info);

        cv::Mat mat() const;
        detection* detections() const;
        int num_of_detections() const;
        Info info() const;

        // Info settings
        void setIDVector(std::vector<unsigned int> id);
        void setIDVector(unsigned int frame_id);
        std::vector<unsigned int> id_vector();
        

        virtual void updateInfo();

    protected:
        unique<Info> _Info;
        
        cv::Mat _MatObject;
        void* data;

        detection* _Detections;
        int _NumberOfDetections;
    };

    using ImageBatch = sharedV<shared<Image>>;
}