#ifndef HFNETTFMODEL_H
#define HFNETTFMODEL_H

#include <string>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Extractors/BaseModel.h"

#ifdef USE_TENSORFLOW
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/c/c_api.h"
#endif // USE_TENSORFLOW

namespace ORB_SLAM3
{

#ifdef USE_TENSORFLOW


class HFNetTFModel : public BaseModel
{
public:
    HFNetTFModel(const std::string &strResamplerDir, const std::string &strModelDir)
    {
        bool bLoadedLib = LoadResamplerOp(strResamplerDir);
        bool bLoadedModel = LoadHFNetTFModel(strModelDir);

        mbVaild = bLoadedLib & bLoadedModel;
    }

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                int nKeypointsNum, int threshold, int nRadius) override;

    bool IsValid(void) override { return mbVaild; }

private:
    bool LoadResamplerOp(const std::string &strResamplerDir);

    bool LoadHFNetTFModel(const std::string &strModelDir);

    void Mat2Tensor(const cv::Mat &image, tensorflow::Tensor *tensor);


    std::unique_ptr<tensorflow::Session> mSession;
    tensorflow::GraphDef mGraph;
    bool mbVaild;
};

#else // USE_TENSORFLOW

class HFNetTFModel : public BaseModel
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