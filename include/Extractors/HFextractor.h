#ifndef HFNETEXTRACTOR_H
#define HFNETEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
#include "Extractors/BaseExtractor.h"
#include "Extractors/HFNetTFModelV2.h"
#include "Extractors/HFNetVINOModel.h"

namespace ORB_SLAM3
{

class HFextractor : public BaseExtractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    HFextractor(int nfeatures, int nNMSRadius, float threshold,
                float scaleFactor, int nlevels, const std::vector<BaseModel*>& vpModels);

    ~HFextractor(){}

    // Compute the features and descriptors on an image.
    int operator()(const cv::Mat &_image, std::vector<cv::KeyPoint>& _keypoints,
                   cv::Mat &_localDescriptors, cv::Mat &_globalDescriptors) override;

public:

    int nfeatures;
    int nNMSRadius;
    float threshold;
    std::vector<BaseModel*> mvpModels;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    void ComputePyramid(cv::Mat image);
};

} //namespace ORB_SLAM

#endif
