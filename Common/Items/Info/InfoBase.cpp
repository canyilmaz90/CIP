#include "Info.h"
#include "Utils.h"

using namespace TRT;

Info::Info() {}
Info::~Info() {}


void Info::setIDVector(std::vector<unsigned int> id)
{
    _IDVector = id;
}

void Info::setIDVector(unsigned int frame_id)
{
    std::vector<unsigned int> id({frame_id});
    _IDVector = id;
}

std::vector<unsigned int> Info::id_vector() const
{
    return _IDVector;
}


void Info::setType(InfoType type)
{
    _Type = type;
}

InfoType Info::type() const
{
    return _Type;
}


void Info::setParent(Info* parent)
{
    this->_Parent = parent;
}

Info* Info::getParent() const 
{
    return this->_Parent;
}
