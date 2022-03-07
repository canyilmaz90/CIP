#include "ObjectInfo.h"

using namespace TRT;

ObjectInfo::ObjectInfo()
{
    this->setType(InfoType::OBJECT);
}
ObjectInfo::~ObjectInfo() {}

void ObjectInfo::setPosition(Box pos)
{
    _Position = pos;
}

Box ObjectInfo::position()
{
    return _Position;
}

void ObjectInfo::setGlobalPosition(Box pos)
{
    _GlobalPosition = pos;
}

Box ObjectInfo::global_position()
{
    return _GlobalPosition;
}

void ObjectInfo::setClassID(int id)
{
    _ClassID = id;
}

int ObjectInfo::class_id()
{
    return _ClassID;
}

void ObjectInfo::setClassName(std::string name)
{
    _ClassName = name;
}

std::string ObjectInfo::class_name()
{
    return _ClassName;
}

void ObjectInfo::setDetectionConfidence(float x)
{
    _DetectionConfidence = x;
}

float ObjectInfo::detection_confidence()
{
    return _DetectionConfidence;
}
