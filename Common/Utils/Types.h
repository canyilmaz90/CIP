#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <iterator>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <unistd.h>
#include <atomic>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <limits>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#ifndef USE_NPP
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif


namespace TRT
{
    namespace Types
    {
        #define SleepHere(ms) usleep(ms*1000)
        #define MAX_DETECTION_NUM 100
        #define max_int std::numeric_limits<int>::max()
        #define max_unsigned_int std::numeric_limits<unsigned int>::max()

        using timestamp = std::chrono::steady_clock::time_point;

        /**
         * @brief simplified unique pointer declaration for queues
         * 
         * @tparam T 
         */
        template <class T>
        using unique = std::unique_ptr<T>;

        /**
         * @brief simplified shared pointer declaration for queues
         * 
         * @tparam T 
         */
        template <class T>
        using shared = std::shared_ptr<T>;

        /**
         * @brief simplified weak pointer declaration for queues
         * 
         * @tparam T 
         */
        template <class T>
        using weak = std::weak_ptr<T>;

        /**
         * @brief simplified shared pointer declaration for vectors
         * 
         * @tparam T 
         */
        template <class T>
        using sharedV = std::shared_ptr<std::vector<T>>;

        typedef struct Box
        {
            float x1, y1, x2, y2;
            Box() {};
            Box(float x1_, float y1_, float x2_, float y2_)
            {
                x1 = x1_;
                y1 = y1_;
                x2 = x2_;
                y2 = y2_;
            };
        } Box;

        typedef struct detection
        {
            Box bboxNormalized;
            Box bboxCorner;
            int class_id;
            std::string class_name;
            float prob;
        } detection;

        typedef enum
        {
            FRAME,
            OBJECT
        } ImageType;

        typedef enum
        {
            UNKNOWN,
            FRAME,
            MODULE,
            OBJECT,
            ATTRIBUTE
        } InfoType;

        typedef enum
        {
            PRODUCTIVE,
            ATTRIBUTE
        } ModuleType;

        typedef struct plate
        {
            Box pos;
            float score;
            std::string text;
        } plate;

        typedef struct vehicle
        {
            Box pos;
            float score;
            std::string type;
            std::string color;
            std::string brand;
            unsigned int numPlates;
            std::vector<plate> plates;
        } vehicle;

        typedef struct resolution
        {
            unsigned int width, height, channels;
            resolution();
            resolution(unsigned int _width, unsigned int _height, unsigned int _channels=3)
            {
                width = _width;
                height = _height;
                channels = _channels;
            }
        } resolution;
    }
}
