#include "YOLO.h"

using namespace TRT;

YOLO::YOLO(nlohmann::json config, gLogger gLogger)
{
    std::string engineFilePath = config["path"];
    int batch_size = config["batch"];
    _conf = config["conf_thresh"];
    _nms = config["nms_thresh"];
    std::string names_file = config["names"];
    _classes = getClassNames(names_file);
    this->init(engineFilePath, gLogger, batch_size);
}

YOLO::~YOLO()
{
}

void YOLO::setInputBatch(std::shared_ptr<std::vector<std::shared_ptr<Image>>> inputBatch)
{
    _InputBatch = inputBatch;
}


#ifdef USE_NPP
void YOLO::preprocess()
{
    auto modelWidth = input_dims[0].d[3];
    auto modelHeight = input_dims[0].d[2];
    auto channels = input_dims[0].d[1];
    const float factor = 1.0 / 255.0;
    // float* gpu_input = (float *) buffers[0];
    auto modelInputLength = modelHeight * modelWidth * channels;
    void *gpu_frame, *resized, *converted, *normalized;

    

    for (int id=0; id < _InputBatch->size(); id++){
        auto imWidth = _InputBatch->operator[](id)->_MatObject.cols;
        auto imHeight = _InputBatch->operator[](id)->_MatObject.rows;
        auto imChannels = _InputBatch->operator[](id)->_MatObject.channels();
        uchar* imData = _InputBatch->operator[](id)->_MatObject.data;

        auto imageLength = imWidth * imHeight * imChannels;
        CUDA_CHECK(cudaMalloc(&gpu_frame, imageLength));
        CUDA_CHECK(cudaMalloc(&resized, modelInputLength));
        CUDA_CHECK(cudaMalloc(&converted, modelInputLength * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&normalized, modelInputLength * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(gpu_frame, imData, imageLength, cudaMemcpyHostToDevice, stream));
        
        CUDA_CHECK(cudaMemset(resized, 0, modelInputLength));
        imageResize_8u_C3R(gpu_frame, imWidth, imHeight, resized, modelWidth, modelHeight);
        // std::cout << resized << std::endl;
        // CUDA_CHECK(cudaResize(gpu_frame, _InputBatch[id].image.cols, _InputBatch[id].image.rows, resized, modelWidth, modelHeight));
        CUDA_CHECK(cudaBGR8ToRGB32(resized, converted, modelWidth, modelHeight));
        CUDA_CHECK(cudaNormalize(converted, normalized, modelWidth, modelHeight, factor));
        CUDA_CHECK(cudaSplit(normalized, (float *)(buffers[0] + id * modelInputLength * sizeof(float)), modelWidth, modelHeight));
        
        CUDA_FREE(gpu_frame);
        CUDA_FREE(resized);
        CUDA_FREE(converted);
        CUDA_FREE(normalized);
    }
}
#else
void YOLO::preprocess()
{
    int index = 0;
    auto modelWidth = input_dims[0].d[3];
    auto modelHeight = input_dims[0].d[2];
    auto channels = input_dims[0].d[1];
    auto input_size = cv::Size(modelWidth, modelHeight);
    float* gpu_input = (float *) buffers[0];

    // cv::cuda::GpuMat gpu_frame, resized, converted;
    cv::cuda::GpuMat normalizedgpu;
    cv::Mat resized, converted, normalized;

    for (int id = 0; id < _InputBatch->size(); id++){
        if (_InputBatch->operator[](id)->mat().cols > 0)
        {
            // gpu_frame.upload(_InputBatch[id].image);
            // cv::cuda::resize(gpu_frame, resized, input_size, 0.0, 0.0, cv::INTER_LINEAR);
            // cv::cuda::cvtColor(resized, converted, cv::COLOR_BGR2RGB);
            // converted.convertTo(normalizedgpu, CV_32FC3, 1.f/255.f);

            cv::resize(_InputBatch->operator[](id)->mat(), resized, input_size, 0.0, 0.0, cv::INTER_LINEAR);
            cv::cvtColor(resized, converted, cv::COLOR_BGR2RGB);
            converted.convertTo(normalized, CV_32FC3, 1.f/255.f);
            normalizedgpu.upload(normalized);

            std::vector<cv::cuda::GpuMat> tensor;
            int channelLength = modelWidth * modelHeight;
            for (size_t i = 0; i < channels; ++i)
            {
                tensor.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + (index + i) * channelLength));
            }
            index += channels;
            cv::cuda::split(normalizedgpu, tensor);
        }
    }
}
#endif


void YOLO::postprocess()
{
    float *gpu_output_box = (float *) buffers[1];
    float *gpu_output_score = (float *) buffers[2];
    // copy results from GPU to CPU
    float* cpu_output_box = (float*) malloc(getSizeByDim(output_dims[0]) * sizeof(float));
    float* cpu_output_score = (float*) malloc(getSizeByDim(output_dims[1]) * sizeof(float));
    CUDA_CHECK(cudaMemcpy(cpu_output_box, gpu_output_box, getSizeByDim(output_dims[0]) * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu_output_score, gpu_output_score, getSizeByDim(output_dims[1]) * sizeof(float), cudaMemcpyDeviceToHost));

    detection det;
    int num_preds = output_dims[1].d[1];
    int num_class = output_dims[1].d[2];

    for (int i = 0; i < _InputBatch->size(); ++i)
    {
        detection* dets = (detection*)xcalloc(MAX_DETECTION_NUM, sizeof(detection));
        int d = 0;
        for (int j = 0; j < num_preds; ++j)
        {
            float prob = 0;
            int idx;
            for (int k = 0; k < num_class; ++k)
            {
                if (*(cpu_output_score + i*num_preds*num_class + j*num_class + k) > prob)
                {
                    prob = *(cpu_output_score + i*num_preds*num_class + j*num_class + k);
                    idx = k;
                }
            }

            if (prob > _conf)
            {
                det.bboxNormalized.x1 = *(cpu_output_box + i*num_preds*4 + j*4);
                det.bboxNormalized.y1 = *(cpu_output_box + i*num_preds*4 + j*4 + 1);
                det.bboxNormalized.x2 = *(cpu_output_box + i*num_preds*4 + j*4 + 2);
                det.bboxNormalized.y2 = *(cpu_output_box + i*num_preds*4 + j*4 + 3);
                det = checkBorders(det);
                det = calculateBox(det, (float) _InputBatch->operator[](i)->mat().cols, (float) _InputBatch->operator[](i)->mat().rows);
                det.class_id = idx;
                det.class_name = _classes[idx];
                det.prob = prob;
                dets[d] = det; // emplace_back(det);
                d++;
            }
        }
        qsort(dets, d, sizeof(detection), compare_detection_scores);
        nms(dets, d, _nms);
        _InputBatch->operator[](i)->setDetections(dets);
        _InputBatch->operator[](i)->setNumberOfDetections(d);
    }

    free(cpu_output_box);
    free(cpu_output_score);
}


std::shared_ptr<std::vector<std::shared_ptr<Image>>> YOLO::run(std::shared_ptr<std::vector<std::shared_ptr<Image>>> inputBatch)
{
    setInputBatch(inputBatch);

    CUDA_CHECK(cudaMemset(buffers[0], 0, getSizeByDim(input_dims[0]) * sizeof(float)));
    preprocess();
    context->enqueueV2(buffers.data(), stream, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    postprocess();

    return _InputBatch;
}