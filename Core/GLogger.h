#pragma once

#include "Types.h"

namespace TRT
{
    class gLogger : public nvinfer1::ILogger {
    public:
        void log(nvinfer1::ILogger::Severity severity, const char *msg) override {
            // suppress info-level messages
            switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "INTERNAL_ERROR: ";
                break;
            case Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            case Severity::kVERBOSE:
                std::cerr << "VERBOSE: ";
                break;
            default:
                std::cerr << "UNKNOWN: ";
                break;
            }
            std::cerr << msg << std::endl;
        }
    };
}
