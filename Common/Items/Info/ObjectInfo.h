#pragma once

#include "InfoBase.h"

using namespace TRT::Types;

namespace TRT
{
    class ObjectInfo : public Info
    {
    public:
        ObjectInfo();
        ~ObjectInfo();

        void setPosition(Box pos);
        void setGlobalPosition(Box pos);
        void setClassID(int id);
        void setClassName(std::string name);
        void setDetectionConfidence(float conf);
        
        Box position();
        Box global_position();
        int class_id();
        std::string class_name();
        float detection_confidence();

        void updateReceivalStatus() override;

    private:
        Box _Position;
        Box _GlobalPosition;
        int _ClassID;
        std::string _ClassName;
        float _DetectionConfidence;

        std::vector<shared<Info>> _Modules;
        std::vector<bool> _ReceivalTable;
        
    };
}
