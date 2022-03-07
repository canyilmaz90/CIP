#pragma once

#include "Common.h"

using namespace TRT::Types;

namespace TRT
{
    class InfoCollector : public Process<shared<Info>, shared<Info>>
    {
    public:
        InfoCollector();
        ~InfoCollector();

    private:
        void insertNewInfo(shared<Info> info);
        void updateBranchReceivalStatus();
        bool isFrameAllReceived();

    private:
        shared<std::unordered_map<unsigned int, FrameInfo>> _InfoTree;
    };
}