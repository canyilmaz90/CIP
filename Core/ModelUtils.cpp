#include "ModelUtils.h"


size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] > 0) size *= dims.d[i];
    }
    return size;
}

std::vector< std::string > getClassNames(const std::string& names_file)
{
    std::ifstream classes_file(names_file);
    std::vector< std::string > classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with classes names.n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}


int compare_detection_scores(const void* a, const void* b)
{
    detection da = *(detection*) a;
    detection db = *(detection*) b;
    if (db.prob < da.prob) return -1;
    else if (db.prob > da.prob) return 1;
    else return 0;
}

detection calculateBox(detection a, float imageWidth, float imageHeight)
{
    a.bboxCorner.x1 = round(imageWidth * a.bboxNormalized.x1);
    a.bboxCorner.y1 = round(imageHeight * a.bboxNormalized.y1);
    a.bboxCorner.x2 = round(imageWidth * a.bboxNormalized.x2);
    a.bboxCorner.y2 = round(imageHeight * a.bboxNormalized.y2);
    return a;
}

float box_iou(detection a, detection b)
{
    float left = std::max(a.bboxCorner.x1, b.bboxCorner.x1);
    float top = std::max(a.bboxCorner.y1, b.bboxCorner.y1);
    float right = std::min(a.bboxCorner.x2, b.bboxCorner.x2);
    float bottom = std::min(a.bboxCorner.y2, b.bboxCorner.y2);
    float width = std::max(right - left + 1, 0.f);
    float height = std::max(bottom - top + 1, 0.f);
    float interS = width * height;
    float Sa = (a.bboxCorner.x2 - a.bboxCorner.x1 + 1) * (a.bboxCorner.y2 - a.bboxCorner.y1 + 1);
    float Sb = (b.bboxCorner.x2 - b.bboxCorner.x1 + 1) * (b.bboxCorner.y2 - b.bboxCorner.y1 + 1);
    return interS / (Sa + Sb - interS);
}

void nms(detection* dets, int &total, float nms_thresh)
{
    if (total > 1)
    {
        int remain = total;
        for (int i = 0; i < total - 1; ++i)
        {
            if (dets[i].prob == 0)
            {
                continue;
            }
            for (int j = i + 1; j < total; ++j)
            {
                if (dets[i].class_id == dets[j].class_id)
                {
                    if (box_iou(dets[i], dets[j]) > nms_thresh)
                    {
                        dets[j].prob = 0;
                        remain--;
                    }
                }
            }
        }
        qsort(dets, total, sizeof(detection), compare_detection_scores);
        total = remain;
    }
}


detection checkBorders(detection det)
{
    if (det.bboxNormalized.x1 < 0) det.bboxNormalized.x1 = 0;
    if (det.bboxNormalized.y1 < 0) det.bboxNormalized.y1 = 0;
    if (det.bboxNormalized.x2 > 1) det.bboxNormalized.x2 = 1;
    if (det.bboxNormalized.y2 > 1) det.bboxNormalized.y2 = 1;

    return det;
}