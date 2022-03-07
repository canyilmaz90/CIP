#pragma once

#include "InfoBase.h"

using namespace TRT::Types;

namespace TRT
{
    class FrameInfo : public Info
    {
    public:
        FrameInfo();
        ~FrameInfo();

        void setTimeStamp();
        void setResolution(resolution& res);

        timestamp time_stamp() const;
        resolution _resolution() const;

        void update(timestamp t, resolution r) override;

    protected:
        timestamp _TimeStamp;
        resolution _Resolution;

        std::vector<Info *> _Modules;
        std::vector<bool> _ReceivalTable;
    };
}