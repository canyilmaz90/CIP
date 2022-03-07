#pragma once

#include "Common.h"

using namespace TRT::Types;

namespace TRT
{
    class DetectionDistributor : public Distributor<sharedV<shared<Image>>, shared<Image>, shared<Info>>
    {
    public:
        DetectionDistributor();
        ~DetectionDistributor();
        
    private:
        void run();
        static cv::Mat cropDetection(shared<Image> parent, Box box);
        void createChildImageFromParent(shared<Image> parent, shared<Object> child, unsigned int id);
        void collectInfo(shared<ObjectInfo> info, shared<Image> image);
        void sendImage(const sharedQ<shared<Image>> &queue);
        void sendInfo(const sharedQ<shared<Info>> &queue);
    
    private:
        ImageType outType;
        unsigned int _ModuleID;
    };
}