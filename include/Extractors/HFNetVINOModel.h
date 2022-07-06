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
    HFNetVINOModel(const std::string &strXmlPath, const std::string &strModelDir);
    virtual ~HFNetVINOModel(void) = default;

    HFNetVINOModel* clone(void);

    void WarmUp(const cv::Size warmUpSize, bool onlyDetectLocalFeatures);

    bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                int nKeypointsNum, float threshold, int nRadius) override;

    bool DetectOnlyLocal(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                         int nKeypointsNum, float threshold, int nRadius) override;

    void PredictScaledResults(std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                              cv::Size scaleSize, int nKeypointsNum, float threshold, int nRadius) override;

    bool IsValid(void) override { return mbVaild; }

    // std::shared_ptr<ov::Session> mSession;
    // ov::GraphDef mGraph;

protected:
    bool LoadHFNetTFModel(const std::string &strModelDir);

    bool Run(const cv::Mat &image, std::vector<ov::Tensor> &vNetResults, bool onlyDetectLocalFeatures);

    void GetLocalFeaturesFromTensor(const ov::Tensor &tScoreDense, const ov::Tensor &tDescriptorsMap,
                                    std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, 
                                    int nKeypointsNum, float threshold, int nRadius);

    void GetGlobalDescriptorFromTensor(const ov::Tensor &tDescriptors, cv::Mat &globalDescriptors);

    void Mat2Tensor(const cv::Mat &image, ov::Tensor *tensor);

    void ResamplerTF(const ov::Tensor &data, const ov::Tensor &warp, cv::Mat &output);

    std::string mStrXmlPath;
    std::string mStrModelDir;
    bool mbVaild;
    std::vector<ov::Tensor> mvNetResults;
    static ov::Core core;
};

} // namespace ORB_SLAM

#endif // HFNETVINOMODEL_H