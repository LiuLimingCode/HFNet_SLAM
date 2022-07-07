#ifndef HFNETVINOMODEL_H
#define HFNETVINOMODEL_H

#include <string>
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Extractors/BaseModel.h"

#include "openvino/openvino.hpp"

namespace ORB_SLAM3
{

class HFNetVINOModel : public BaseModel
{
public:
    HFNetVINOModel(const std::string &strXmlPath, const std::string &strBinPath);
    virtual ~HFNetVINOModel(void) = default;

    void Compile(const cv::Vec4i inputSize, bool onlyDetectLocalFeatures);

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                int nKeypointsNum, float threshold, int nRadius) override;

    bool DetectOnlyLocal(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                         int nKeypointsNum, float threshold, int nRadius) override;

    void PredictScaledResults(std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                              cv::Size scaleSize, int nKeypointsNum, float threshold, int nRadius) override;

    bool IsValid(void) override { return mbVaild; }

    std::shared_ptr<ov::Model> mpModel;
    std::shared_ptr<ov::CompiledModel> mpExecutableNet;
    std::shared_ptr<ov::InferRequest> mInferRequest;

protected:
    bool LoadHFNetVINOModel(const std::string &strXmlPath, const std::string &strBinPath);

    void printInputAndOutputsInfo(const ov::Model& network);

    bool Run(const cv::Mat &image, std::vector<ov::Tensor> &vNetResults);

    void GetLocalFeaturesFromTensor(const ov::Tensor &tScoreDense, const ov::Tensor &tDescriptorsMap,
                                    std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, 
                                    int nKeypointsNum, float threshold, int nRadius);

    void GetGlobalDescriptorFromTensor(const ov::Tensor &tDescriptors, cv::Mat &globalDescriptors);

    void Mat2Tensor(const cv::Mat &image, ov::Tensor *tensor);

    void ResamplerOV(const ov::Tensor &data, const ov::Tensor &warp, cv::Mat &output);

    std::string mStrXmlPath;
    std::string mStrBinPath;
    bool mbVaild;
    std::vector<ov::Tensor> mvNetResults;
    static ov::Core core;
};

} // namespace ORB_SLAM

#endif // HFNETVINOMODEL_H