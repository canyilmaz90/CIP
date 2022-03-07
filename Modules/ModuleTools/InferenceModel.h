#include "Common.h"
#include "Core.h"

using namespace TRT::Types;

namespace TRT
{
    class InferenceModel : public Process<sharedV<shared<Image>>, sharedV<shared<Image>>>
    {
    public:
        InferenceModel(nlohmann::json config, gLogger gLogger);
        ~InferenceModel();
    
    private:
        void run();
    };
}