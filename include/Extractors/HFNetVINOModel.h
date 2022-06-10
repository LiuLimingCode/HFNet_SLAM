#ifndef HFNETVINOMODEL_H
#define HFNETVINOMODEL_H

#include <string>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Extractors/BaseModel.h"

namespace ORB_SLAM3
{

class HFNetVINOModel : public BaseModel
{
public:
    HFNetVINOModel(const std::string &strResamplerDir, const std::string &strModelDir)
    {
    }

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &descriptors,
                int nKeypointsNum = 1000, int nRadius = 4){return false;}

private:
};

} // namespace ORB_SLAM

#endif // HFNETVINOMODEL_H