#pragma once

#include "ModuleInfo.h"
#include "ObjectInfo.h"

using namespace TRT::Types;

namespace TRT
{
    class ProductiveModuleInfo : public ModuleInfo
    {
    public:
        ProductiveModuleInfo();
        ~ProductiveModuleInfo();

        void initReceivalTable(int size);
        void addObject(shared<ObjectInfo> object);

    private:
        std::vector<shared<ObjectInfo>> _Objects;
        std::vector<bool> _ReceivalTable;
        
    };
}
