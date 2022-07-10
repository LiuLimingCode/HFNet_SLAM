#ifndef HFNETTFMODELV2_H
#define HFNETTFMODELV2_H

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


class HFNetTFModelV2 : public BaseModel
{
public:
    HFNetTFModelV2(const std::string &strModelDir, ModelDetectionMode mode, const cv::Vec4i inputShape);
    virtual ~HFNetTFModelV2(void) = default;

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                int nKeypointsNum, float threshold, int nRadius) override;

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                         int nKeypointsNum, float threshold, int nRadius) override;

    bool Detect(const cv::Mat &intermediate, cv::Mat &globalDescriptors) override;

    void PredictScaledResults(std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                              cv::Size scaleSize, int nKeypointsNum, float threshold, int nRadius) override;

    bool IsValid(void) override { return mbVaild; }

    ModelType Type(void) override { return kHFNetTFModel; }

    std::shared_ptr<tensorflow::Session> mSession;
    tensorflow::GraphDef mGraph;

protected:
    bool LoadHFNetTFModel(const std::string &strModelDir);

    bool Run(std::vector<tensorflow::Tensor> &vNetResults);

    void GetLocalFeaturesFromTensor(const tensorflow::Tensor &tScoreDense, const tensorflow::Tensor &tDescriptorsMap,
                                    std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, 
                                    int nKeypointsNum, float threshold, int nRadius);

    void GetGlobalDescriptorFromTensor(const tensorflow::Tensor &tDescriptors, cv::Mat &globalDescriptors);

    void Mat2Tensor(const cv::Mat &mat, tensorflow::Tensor *tensor);

    void Tensor2Mat(tensorflow::Tensor *tensor, cv::Mat &mat);

    void ResamplerTF(const tensorflow::Tensor &data, const tensorflow::Tensor &warp, cv::Mat &output);

    ModelDetectionMode mMode;
    std::string mstrTFModelPath;
    bool mbVaild = false;
    std::vector<tensorflow::Tensor> mvNetResults;
    std::vector<std::pair<std::string, tensorflow::Tensor>> mvInputTensors;
    std::vector<std::string> mvOutputTensorNames;
    tensorflow::TensorShape mInputShape;
};

#else // USE_TENSORFLOW

class HFNetTFModelV2 : public BaseModel
{
public:
    HFNetTFModelV2()
    {
        std::cerr << "You must set USE_TENSORFLOW in CMakeLists.txt to enable tensorflow function." << std::endl;
        exit(-1);
    }

    bool IsValid(void) override { return false; }
};

#endif // USE_TENSORFLOW

} // namespace ORB_SLAM

#endif // HFNETTFMODELV2_H