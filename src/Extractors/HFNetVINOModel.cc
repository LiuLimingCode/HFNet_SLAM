#include "Extractors/HFNetVINOModel.h"

using namespace cv;
using namespace ov;
using namespace std;

namespace ORB_SLAM3
{
    
ov::Core HFNetVINOModel::core;

HFNetVINOModel::HFNetVINOModel(const std::string &strXmlPath, const std::string &strBinPath)
{
    mStrXmlPath = strXmlPath;
    mStrBinPath = strBinPath;
    LoadHFNetVINOModel(strXmlPath, strBinPath);

    mbVaild = false; // must call Compile before use
}

void HFNetVINOModel::Compile(const cv::Vec4i inputSize, bool onlyDetectLocalFeatures)
{
    ov::Shape inputShape{(size_t)inputSize(0), (size_t)inputSize(1), (size_t)inputSize(2), (size_t)inputSize(3)};
    const ov::Layout modelLayout{"NHWC"};

    mpModel->reshape({{mpModel->input().get_any_name(), inputShape}});

    ov::preprocess::PrePostProcessor ppp(mpModel);

    ppp.input()
        .tensor()
        .set_layout(modelLayout);
    ppp.input().model().set_layout("NHWC");
    // ppp.output(0).tensor().set_element_type(ov::element::f32);
    // ppp.output(1).tensor().set_element_type(ov::element::f32);
    mpModel = ppp.build();

    mpExecutableNet = make_shared<ov::CompiledModel>(core.compile_model(mpModel, "CPU"));

    mInferRequest = make_shared<ov::InferRequest>(mpExecutableNet->create_infer_request());

    mbVaild = true;
}

bool HFNetVINOModel::Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                            int nKeypointsNum, float threshold, int nRadius)
{
    globalDescriptors = cv::Mat();
    DetectOnlyLocal(image, vKeyPoints, localDescriptors, nKeypointsNum, threshold, nRadius);
    return true;
}

bool HFNetVINOModel::DetectOnlyLocal(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                            int nKeypointsNum, float threshold, int nRadius)
{
    if (!mbVaild) return false;

    vKeyPoints.clear();

    ov::Tensor inputTensor = mInferRequest->get_input_tensor();
    ov::Shape inputShape = inputTensor.get_shape();
    if (inputShape[2] != image.cols || inputShape[1] != image.rows || inputShape[3] != image.channels())
    {
        cerr << "The input shape in VINO model should be the same as the compile shape" << endl;
        return false;
    }

    Mat2Tensor(image, &inputTensor);
    
    mInferRequest->infer();

    ov::Tensor tscoreDense = mInferRequest->get_tensor(mpExecutableNet->output("pred/local_head/detector/Squeeze:0"));
    ov::Tensor tLocalDescriptorMap = mInferRequest->get_tensor(mpExecutableNet->output("local_descriptor_map"));

    const int width = tscoreDense.get_shape()[2], height = tscoreDense.get_shape()[1];
    const float scaleWidth = (tLocalDescriptorMap.get_shape()[2] - 1.f) / (float)(tscoreDense.get_shape()[2] - 1.f);
    const float scaleHeight = (tLocalDescriptorMap.get_shape()[1] - 1.f) / (float)(tscoreDense.get_shape()[1] - 1.f);

    auto vResScoresDense = tscoreDense.data<float>();
    cv::KeyPoint keypoint;
    std::vector<cv::KeyPoint> vKeyPointsGood;
    vKeyPointsGood.reserve(10 * nKeypointsNum);
    for (int col = 0; col < width; ++col)
    {
        for (int row = 0; row < height; ++row)
        {
            float score = vResScoresDense[row * width + col];
            if (score >= threshold)
            {
                keypoint.pt.x = col;
                keypoint.pt.y = row;
                keypoint.response = score;
                vKeyPointsGood.emplace_back(keypoint);
            }
        }
    }

    vKeyPoints = NMS(vKeyPointsGood, width, height, nRadius);

    if (vKeyPoints.size() > nKeypointsNum)
    {
        std::partial_sort(vKeyPoints.begin(), vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end(), [](const cv::KeyPoint& p1, const cv::KeyPoint& p2) {
            return p1.response > p2.response;
        });
        vKeyPoints.resize(nKeypointsNum);
    }

    localDescriptors = cv::Mat(vKeyPoints.size(), 256, CV_32F);
    ov::Tensor tWarp(ov::element::f32, {(size_t)vKeyPoints.size(), 2});
    auto pWarp = tWarp.data<float>();
    for (size_t temp = 0; temp < vKeyPoints.size(); ++temp)
    {
        pWarp[temp * 2 + 0] = scaleWidth * vKeyPoints[temp].pt.x;
        pWarp[temp * 2 + 1] = scaleHeight * vKeyPoints[temp].pt.y;
    }

    ResamplerOV(tLocalDescriptorMap, tWarp, localDescriptors);

    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }

    return true;
}

void HFNetVINOModel::PredictScaledResults(std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                              cv::Size scaleSize, int nKeypointsNum, float threshold, int nRadius)
{

}

bool HFNetVINOModel::LoadHFNetVINOModel(const std::string &strXmlPath, const std::string &strBinPath)
{
    mpModel = core.read_model(strXmlPath, strBinPath);
    return true;
}

void HFNetVINOModel::Mat2Tensor(const cv::Mat &image, ov::Tensor *tensor)
{
    cv::Mat imagePixel(image.rows, image.cols, CV_32F, tensor->data<float>());
    image.convertTo(imagePixel, CV_32F);
}

void HFNetVINOModel::ResamplerOV(const ov::Tensor &data, const ov::Tensor &warp, cv::Mat &output)
{
    const int batch_size = data.get_shape()[0];
    const int data_height = data.get_shape()[1];
    const int data_width = data.get_shape()[2];
    const int data_channels = data.get_shape()[3];

    output = cv::Mat(warp.get_shape()[0], data_channels, CV_32F);
    
    const int num_sampling_points = warp.get_shape()[0];
    if (num_sampling_points > 0)
    {
        Resampler(data.data<float>(), warp.data<float>(), output.ptr<float>(),
                  batch_size, data_height, data_width, 
                  data_channels, num_sampling_points);
    }
}

} // namespace ORB_SLAM3
