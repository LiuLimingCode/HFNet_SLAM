#ifndef HFNETEXTRACTOR_H
#define HFNETEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
#include "Extractors/BaseExtractor.h"
#include "Extractors/HFNetTFModel.h"
#include "Extractors/HFNetVINOModel.h"

namespace ORB_SLAM3
{

class HFextractor : public BaseExtractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    HFextractor(int nfeatures, int nNMSRadius, float threshold,
                float scaleFactor, int nlevels, BaseModel* model);

    ~HFextractor(){}

    // Compute the features and descriptors on an image.
    int operator()(const cv::Mat &_image, std::vector<cv::KeyPoint>& _keypoints,
                   cv::Mat &_localDescriptors, cv::Mat &_globalDescriptors) override;

protected:

    int nfeatures;
    int nNMSRadius;
    float threshold;
    BaseModel* model;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;
};

} //namespace ORB_SLAM

#endif
