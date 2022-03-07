#include "ProductiveModuleInfo.h"

using namespace TRT;

ProductiveModuleInfo::ProductiveModuleInfo()
{
    
}

void ProductiveModuleInfo::initReceivalTable(int size)
{
    _ReceivalTable.resize(size);
    std::fill(_ReceivalTable.begin(), _ReceivalTable.end(), false);
}

void ProductiveModuleInfo::addObject(shared<ObjectInfo> object)
{
    _Objects.push_back(object);
}