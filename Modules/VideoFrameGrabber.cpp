#include "VideoFrameGrabber.h"

using namespace TRT;

VideoFrameGrabber::VideoFrameGrabber(nlohmann::json config)
{
    this->setConfig(config);
    _SourcePath = config["sourcePath"];
    _Cap = std::make_shared<cv::VideoCapture>(_SourcePath);
    if(!_Cap->isOpened())
    {
        std::cout << "Error opening video stream or file" << std::endl;
    }
}

VideoFrameGrabber::~VideoFrameGrabber()
{

}


void VideoFrameGrabber::run()
{
    unsigned int frameID = 0;
    while (_IsRunning)
    {
        try
        {
            cv::Mat frame;
            *_Cap >> frame;
            
            if (!frame.empty())
            {
                if (frameID == 0) _Resolution = resolution(frame.cols, frame.rows, frame.channels());
                shared<Image> image = std::make_shared<Image>(frame);
                #ifdef USE_NPP
                cudaMallocHost(&image->_MatObject.data, frame.cols*frame.rows*3);
                // memcpy(frame.data, frame.image.data, frame.image.cols*frame.image.rows*3);
                #endif
                FrameInfo finfo;
                finfo.setIDVector(frameID);
                finfo.setResolution(_Resolution);
                image->setInfo(finfo);
                shared<Info> info = std::make_shared<FrameInfo>(finfo);
                send<shared<Info>>(_InfoOutputQueue, info);
                std::for_each(_OutputQueues->begin(), _OutputQueues->end(), [&](sharedQ<shared<Image>> q)
                {
                    send<shared<Image>>(q, image);
                });
                frameID++;
            }
            else
            {
                std::cout << "Video stream has ended." << std::endl;
                this->stop();
            }
        }
        catch (...)
        {

        }
    }

}
