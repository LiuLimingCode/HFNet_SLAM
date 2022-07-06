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

class [[deprecated]] HFNetTFModel : public BaseModel
{
public:
    HFNetTFModel(const std::string &strResamplerDir, const std::string &strModelDir);
    virtual ~HFNetTFModel(void) = default;

    HFNetTFModel* clone(void);

    void WarmUp(const cv::Size warmUpSize, bool onlyDetectLocalFeatures);

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                int nKeypointsNum, float threshold, int nRadius) override;

    bool DetectOnlyLocal(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                         int nKeypointsNum, float threshold, int nRadius) override;

    void PredictScaledResults(std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                              cv::Size scaleSize, int nKeypointsNum, float threshold, int nRadius) override;

    bool IsValid(void) override { return mbVaild; }

    std::shared_ptr<tensorflow::Session> mSession;
    tensorflow::GraphDef mGraph;

private:
    bool LoadResamplerOp(const std::string &strResamplerDir);

    bool LoadHFNetTFModel(const std::string &strModelDir);

    void Mat2Tensor(const cv::Mat &image, tensorflow::Tensor *tensor);

    bool Run(const cv::Mat &image, std::vector<tensorflow::Tensor> &vNetResults, bool onlyDetectLocalFeatures,
             int nKeypointsNum, float threshold, int nRadius);

    void GetLocalFeaturesFromTensor(const tensorflow::Tensor &tKeyPoints, const tensorflow::Tensor &tDescriptorsMap, const tensorflow::Tensor &tScoreDense,
                                    std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors);

    void GetGlobalDescriptorFromTensor(const tensorflow::Tensor &tDescriptors, cv::Mat &globalDescriptors);

    std::string mStrModelPath;
    bool mbVaild;
    static bool mbLoadedResampler;
    std::vector<tensorflow::Tensor> mvNetResults;
};

#else // USE_TENSORFLOW

class [[deprecated]] HFNetTFModel : public BaseModel
{
public:
    HFNetTFModel()
    {
        std::cerr << "You must set USE_TENSORFLOW in CMakeLists.txt to enable tensorflow function." << std::endl;
        exit(-1);
    }

    bool IsValid(void) override { return false; }
};

#endif // USE_TENSORFLOW

} // namespace ORB_SLAM

#endif // HFNETTFMODEL_H