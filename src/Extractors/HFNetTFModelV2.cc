#include "Extractors/HFNetTFModelV2.h"

using namespace cv;
using namespace std;
using namespace tensorflow;

namespace ORB_SLAM3
{

#ifdef USE_TENSORFLOW

HFNetTFModelV2::HFNetTFModelV2(const std::string &strModelDir)
{
    mbVaild = LoadHFNetTFModel(strModelDir);
}

HFNetTFModelV2* HFNetTFModelV2::clone(void)
{
    if (!mbVaild) return nullptr;
    HFNetTFModelV2 *newModel = new HFNetTFModelV2(mStrModelPath);
    return newModel;
}

void HFNetTFModelV2::WarmUp(const cv::Size warmUpSize, bool onlyDetectLocalFeatures)
{
    // Warming up, the tensorflow model cost huge time at the first detection.
    // Therefore, give a fake image to waming up
    // The size of fake image should be the same as the real image.

    if (mbVaild && warmUpSize.width > 0 && warmUpSize.height > 0)
    {
        Mat fakeImg(warmUpSize, CV_8UC1);
        cv::randu(fakeImg, Scalar(0), Scalar(255));
        vector<tensorflow::Tensor> vNetResults;
        Run(fakeImg, vNetResults, onlyDetectLocalFeatures);
    }
}

bool HFNetTFModelV2::Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                            int nKeypointsNum, float threshold, int nRadius)
{
    Run(image, mvNetResults, false);
    GetGlobalDescriptorFromTensor(mvNetResults[2], globalDescriptors);
    GetLocalFeaturesFromTensor(mvNetResults[0], mvNetResults[1], vKeypoints, localDescriptors, nKeypointsNum, threshold, nRadius);
    return true;
}

bool HFNetTFModelV2::DetectOnlyLocal(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors,
                                     int nKeypointsNum, float threshold, int nRadius)
{
    Run(image, mvNetResults, true);
    GetLocalFeaturesFromTensor(mvNetResults[0], mvNetResults[1], vKeypoints, localDescriptors, nKeypointsNum, threshold, nRadius);
    return true;
}

void HFNetTFModelV2::PredictScaledResults(std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors,
                                          cv::Size scaleSize, int nKeypointsNum, float threshold, int nRadius)
{
    tensorflow::Tensor &tScoreDense = mvNetResults[0];
    tensorflow::Tensor tScaledScoreDense(DT_FLOAT, {1, scaleSize.height, scaleSize.width});
    cv::Mat src(tScoreDense.dim_size(1), tScoreDense.dim_size(2), CV_32F, tScoreDense.flat<float>().data());
    cv::Mat dst(tScaledScoreDense.dim_size(1), tScaledScoreDense.dim_size(2), CV_32F, tScaledScoreDense.flat<float>().data());
    cv::resize(src, dst, cv::Size(scaleSize.height, scaleSize.width), 0, 0, INTER_LINEAR);

    GetLocalFeaturesFromTensor(tScaledScoreDense, mvNetResults[1], vKeypoints, localDescriptors, nKeypointsNum, threshold, nRadius);
}

bool HFNetTFModelV2::Run(const cv::Mat &image, std::vector<tensorflow::Tensor> &vNetResults, bool onlyDetectLocalFeatures) 
{
    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);

    std::vector<string> outputTensorName = {"scores_dense:0", "local_descriptor_map:0"};
    if (!onlyDetectLocalFeatures) outputTensorName.emplace_back("global_descriptor:0");
    Status status = mSession->Run({{"image:0", tImage}}, outputTensorName, {}, &vNetResults);

    // auto t1 = chrono::steady_clock::now();
    // Scope root = Scope::NewRootScope();
    // auto tAxis = tensorflow::ops::Const(root, -1);
    // tensorflow::ops::ExpandDims expendOps(root, vNetResults[0], tAxis);
    // auto session = new tensorflow::ClientSession(root);
    // vector<tensorflow::Tensor> outputs;
    // session->Run({expendOps}, &outputs);
    // delete session;
    // vNetResults[0] = outputs[0];
    // auto t2 = chrono::steady_clock::now();
    // auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
    // cout << "cost time: " << t << endl;
    // cout << vNetResults[0].shape() << endl;

    return status.ok();
}

void HFNetTFModelV2::GetLocalFeaturesFromTensor(const tensorflow::Tensor &tScoreDense, const tensorflow::Tensor &tDescriptorsMap,
                                                std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors, 
                                                int nKeypointsNum, float threshold, int nRadius)
{
    auto vResScoresDense = tScoreDense.tensor<float, 3>(); // shape: [1 image.height image.width]
    auto vResLocalDescriptorMap = tDescriptorsMap.tensor<float, 4>();

    const int width = tScoreDense.shape().dim_size(2), height = tScoreDense.shape().dim_size(1);
    const float scaleWidth = (tDescriptorsMap.shape().dim_size(2) - 1.f) / (float)(tScoreDense.shape().dim_size(2) - 1.f);
    const float scaleHeight = (tDescriptorsMap.shape().dim_size(1) - 1.f) / (float)(tScoreDense.shape().dim_size(1) - 1.f);

    cv::KeyPoint keypoint;
    keypoint.angle = 0;
    keypoint.octave = 0;
    std::vector<cv::KeyPoint> vKeyPointsGood;
    vKeyPointsGood.reserve(10 * nKeypointsNum);
    for (int col = 0; col < width; ++col)
    {
        for (int row = 0; row < height; ++row)
        {
            float score = vResScoresDense(row * width + col);
            if (score >= threshold)
            {
                keypoint.pt.x = col;
                keypoint.pt.y = row;
                keypoint.response = score;
                vKeyPointsGood.emplace_back(keypoint);
            }
        }
    }

    vKeyPointsGood = NMS(vKeyPointsGood, width, height, 4);

    if (vKeyPointsGood.size() > nKeypointsNum)
    {
        vKeypoints = DistributeOctTree(vKeyPointsGood, 0, width, 0, height, nKeypointsNum);
    }
    else vKeypoints = vKeyPointsGood;

    localDescriptors = cv::Mat(vKeypoints.size(), 256, CV_32F);
    Tensor tWarp(DT_FLOAT, TensorShape({(int)vKeypoints.size(), 2}));
    auto pWarp = tWarp.tensor<float, 2>();
    for (size_t temp = 0; temp < vKeypoints.size(); ++temp)
    {
        pWarp(temp * 2 + 0) = scaleWidth * vKeypoints[temp].pt.x;
        pWarp(temp * 2 + 1) = scaleHeight * vKeypoints[temp].pt.y;
    }

    ResamplerTF(tDescriptorsMap, tWarp, localDescriptors);

    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }
}

void HFNetTFModelV2::GetGlobalDescriptorFromTensor(const tensorflow::Tensor &tDescriptors, cv::Mat &globalDescriptors)
{
    auto vResGlobalDescriptor = tDescriptors.tensor<float, 2>();
    globalDescriptors = cv::Mat(4096, 1, CV_32F);
    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDescriptor(temp);
    }
}

bool HFNetTFModelV2::LoadHFNetTFModel(const std::string &strModelDir)
{
    mStrModelPath = strModelDir;
    tensorflow::Status status;
    tensorflow::SessionOptions sessionOptions;
    tensorflow::RunOptions runOptions;
    tensorflow::SavedModelBundle bundle;

    status = LoadSavedModel(sessionOptions, runOptions, strModelDir, {tensorflow::kSavedModelTagServe}, &bundle);
    if(!status.ok()){
        std::cerr << "Failed to load HFNet model at path: " << strModelDir <<std::endl;
        return false;
    }

    mSession = std::move(bundle.session);
    status = mSession->Create(mGraph);
    if(!status.ok()){
        std::cerr << "Failed to create mGraph for HFNet" << std::endl;
        return false;
    }

    std::cout << "Sucessfully loaded HFNet model" << std::endl;
    return true;
}

void HFNetTFModelV2::Mat2Tensor(const cv::Mat &image, tensorflow::Tensor *tensor)
{
    float *p = tensor->flat<float>().data();
    cv::Mat imagepixel(image.rows, image.cols, CV_32F, p);
    image.convertTo(imagepixel, CV_32F);
}

void HFNetTFModelV2::ResamplerTF(const tensorflow::Tensor &data, const tensorflow::Tensor &warp, cv::Mat &output)
{
    const tensorflow::TensorShape& data_shape = data.shape();
    const int batch_size = data_shape.dim_size(0);
    const int data_height = data_shape.dim_size(1);
    const int data_width = data_shape.dim_size(2);
    const int data_channels = data_shape.dim_size(3);
    const tensorflow::TensorShape& warp_shape = warp.shape();

    tensorflow::TensorShape output_shape = warp.shape();
    // output_shape.set_dim(output_shape.dims() - 1, data_channels);
    // output = Tensor(DT_FLOAT, output_shape);
    output = cv::Mat(output_shape.dim_size(0), data_channels, CV_32F);
    
    const int num_sampling_points = warp.NumElements() / batch_size / 2;
    if (num_sampling_points > 0)
    {
        Resampler(data.flat<float>().data(), warp.flat<float>().data(), output.ptr<float>(),
                  batch_size, data_height, data_width, 
                  data_channels, num_sampling_points);
    }
}

#endif

} // namespace ORB_SLAM3