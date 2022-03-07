#pragma once

#include "Utils.h"

using namespace TRT::Types;

namespace TRT
{
    class Info
    {
    public:
        Info();
        ~Info();

        // General for all Info types
        void setIDVector(std::vector<unsigned int> id);
        void setIDVector(unsigned int frame_id);
        void setType(InfoType type);
        std::vector<unsigned int> id_vector() const;
        InfoType type() const;
    
        void setParent(Info* parent);
        Info* getParent() const;
        
        virtual void update();
        virtual void update(timestamp t, resolution r);
        virtual void update(InfoType type, detection* det);
        virtual void updateReceivalStatus();
        virtual void updateParentReceivalStatus();
    
    protected:
        InfoType _Type;
        std::vector<unsigned int> _IDVector;
        bool _ReceivalStatus;

        Info* _Parent;
    };
}