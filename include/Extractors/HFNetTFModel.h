#ifndef HFNETTFMODEL_H
#define HFNETTFMODEL_H

#include <string>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Extractors/HFNetBaseModel.h"

#ifdef USE_TENSORFLOW
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/c/c_api.h"
#endif // USE_TENSORFLOW

namespace ORB_SLAM3
{

#ifdef USE_TENSORFLOW


class HFNetTFModel : public HFNetBaseModel
{
public:
    HFNetTFModel(const std::string &strResamplerDir, const std::string &strModelDir)
    {
        LoadResamplerOp(strResamplerDir);
        LoadHFNetTFModel(strModelDir);
    }

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &descriptors,
                int nKeypointsNum = 1000, int nRadius = 4);

private:
    bool LoadResamplerOp(const std::string &strResamplerDir);

    bool LoadHFNetTFModel(const std::string &strModelDir);

    void Mat2Tensor(const cv::Mat &image, tensorflow::Tensor *tensor);


    std::unique_ptr<tensorflow::Session> session;
    tensorflow::GraphDef graph;
};

#else // USE_TENSORFLOW

class HFNetTFModel : public HFNetBaseModel
{
public:
    HFNetTFModel()
    {
        std::cerr << "You must set USE_TENSORFLOW in CMakeLists.txt to enable tensorflow function." << std::endl;
    }
};

#endif // USE_TENSORFLOW

} // namespace ORB_SLAM

#endif // HFNETTFMODEL_H