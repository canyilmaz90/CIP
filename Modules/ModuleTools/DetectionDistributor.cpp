#include "DetectionDistributor.h"

using namespace TRT;

DetectionDistributor::DetectionDistributor()
{
    sharedQ<sharedV<shared<Image>>> Q = make_Q<sharedV<shared<Image>>>();
    this->setInputQueue(Q);
}

void DetectionDistributor::run()
{
    while (_IsRunning)
    {
        try
        {
            sharedV<shared<Image>> batch = _InputQueue->pop();
            for (const shared<Image>& image : *batch)
            {
                shared<ProductiveModuleInfo> moduleInfo = std::make_shared<ProductiveModuleInfo>();
                moduleInfo->initReceivalTable(image->num_of_detections());
                for (unsigned int i = 0; i < image->num_of_detections(); ++i)
                {
                    shared<Object> cropImage = std::make_shared<Object>(outType);
                    createChildImageFromParent(image, cropImage, i);
                    shared<ObjectInfo> objectInfo = std::make_shared<ObjectInfo>();
                    collectInfo(objectInfo, cropImage);

                }
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        
    }
}

void DetectionDistributor::createChildImageFromParent(shared<Image> parent, shared<Object> child, unsigned int id)
{
    detection det = parent->detections()[id];
    Box box = det.bboxCorner;
    child->setMat(cropDetection(parent, box));
    shared<ObjectInfo> childInfo = std::make_shared<ObjectInfo>();
    std::vector<unsigned int> idV(parent->info()->id_vector());
    idV.insert(idV.end(), {_ModuleID, id});
    childInfo->setIDVector(idV);
    childInfo->setPosition(box);
    if (parent->type() != ImageType::FRAME)
    {
        Box parentPos = parent->info()->;
        child->setGlobalPosition(Box(box.x1 + parentPos.x1,
                                     box.y1 + parentPos.y1,
                                     box.x2 + parentPos.x1,
                                     box.y2 + parentPos.y1));
    }
    else
    {
        child->setGlobalPosition(box);
    }
    child->setClassID(det.class_id);
    child->setClassName(det.class_name);
    child->setDetectionConfidence(det.prob);
}

cv::Mat DetectionDistributor::cropDetection(shared<Image> image, Box box)
{
    cv::Mat mat = image->mat();
    cv::Rect rect((int)box.x1, (int)box.y1, (int)box.x2 - (int)box.x1, (int)box.y2 - (int)box.y1);
    cv::Mat crop = mat(rect);
    return crop.clone();
}
