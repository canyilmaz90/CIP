#include "Common.h"

using namespace TRT::Types;

namespace TRT
{
    class VideoFrameGrabber : public Distributor<void, shared<Image>, shared<Info>>
    {
    public:
        VideoFrameGrabber(nlohmann::json config);
        ~VideoFrameGrabber();
    
    private:
        void run();

    private:
        std::string _SourcePath;
        shared<cv::VideoCapture> _Cap;

        resolution _Resolution;
    };
}