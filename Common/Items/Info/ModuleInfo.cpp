#include "ModuleInfo.h"

using namespace TRT;

ModuleInfo::ModuleInfo() {
    this->setType(InfoType::MODULE);
}

ModuleInfo::~ModuleInfo() {}

void ModuleInfo::setModuleType(ModuleType mtype) {
    this->_ModuleType = mtype;
}

ModuleType ModuleInfo::module_type() const {
    return this->_ModuleType;
}
