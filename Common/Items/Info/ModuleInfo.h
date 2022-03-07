#pragma once

#include "InfoBase.h"
#include "ObjectInfo.h"

using namespace TRT::Types;

namespace TRT
{
    class ModuleInfo : public Info
    {
    public:
        ModuleInfo();
        ~ModuleInfo();

        void setModuleType(ModuleType mtype);
        ModuleType module_type() const;

    protected:
        ModuleType _ModuleType;
    };
}
