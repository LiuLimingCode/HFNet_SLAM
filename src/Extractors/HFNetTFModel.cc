/**
 * Result
 * 1. We do not need OctTree, because we have NMS
 * 2. Set NMS radius to zero is catastrophic, one is much better
 * 3. It is reasonable to set threshold to zero
 */
#include "Extractors/HFNetTFModel.h"
using namespace cv;
using namespace std;
using namespace tensorflow;

namespace ORB_SLAM3
{

#ifdef USE_TENSORFLOW

HFNetTFModel::HFNetTFModel(const std::string &strResamplerDir, const std::string &strModelDir, const cv::Size warmUpImageSize)
{
    bool bLoadedLib = LoadResamplerOp(strResamplerDir);
    bool bLoadedModel = LoadHFNetTFModel(strModelDir);

    mbVaild = bLoadedLib & bLoadedModel;

    // Warming up, the tensorflow model cost huge time at the first detection.
    // Therefore, give a fake image to waming up
    // The size of fake image should be the same as the real image.
    if (mbVaild && warmUpImageSize.width > 0 && warmUpImageSize.height > 0)
    {
        Mat fakeImg(warmUpImageSize, CV_8UC1);
        cv::randu(fakeImg, Scalar(0), Scalar(255));
        std::vector<cv::KeyPoint> vKeyPoint;
        cv::Mat localDescriptors, globalDescriptors;
        Detect(fakeImg, vKeyPoint, localDescriptors, globalDescriptors, 1000, 0, 4);
    }
}

bool HFNetTFModel::Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                          int nKeypointsNum, float threshold, int nRadius) 
{
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    
    vector<Tensor> outputs;
    Status status = mSession->Run({{"image:0", tImage},{"pred/simple_nms/radius", tRadius},{"pred/top_k_keypoints/k", tKeypointsNum}},
                                 {"keypoints", "local_descriptors", "scores", "global_descriptor"}, {}, &outputs);
    if (!status.ok()) return false;

    int nResNumber = outputs[0].shape().dim_size(1);

    auto vResKeypoints = outputs[0].tensor<int32, 3>();
    auto vResLocalDes = outputs[1].tensor<float, 3>();
    auto vResScores = outputs[2].tensor<float, 2>();
    auto vResGlobalDes = outputs[3].tensor<float, 2>();

    vKeypoints.clear();
    vKeypoints.reserve(nResNumber);
    KeyPoint kp;
    kp.angle = 0;
    kp.octave = 0;
    for(int index = 0; index < nResNumber; index++)
    {
        if (vResScores(index) < threshold) continue;
        kp.pt = Point2f(vResKeypoints(2 * index), vResKeypoints(2 * index + 1));
        kp.response = vResScores(index);
        vKeypoints.emplace_back(kp);
    }
    localDescriptors = cv::Mat(vKeypoints.size(), 256, CV_32F);
    for (size_t index = 0; index < vKeypoints.size(); ++index)
    {
        for (int temp = 0; temp < 256; ++temp)
        {
            localDescriptors.ptr<float>(index)[temp] = vResLocalDes(256 * index + temp); 
        }
    }
    globalDescriptors = cv::Mat(4096, 1, CV_32F);
    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDes(temp);
    }
    return true;
}

bool HFNetTFModel::LoadResamplerOp(const std::string &strResamplerDir)
{
    TF_Status *status = TF_NewStatus();
    TF_LoadLibrary(strResamplerDir.c_str(), status);
    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "TF_LoadLibrary() error with code: " << TF_GetCode(status) << std::endl;
        std::cerr << "Failed to load resampler.so in path: " << strResamplerDir << std::endl;
        return false;
    }
    std::cout << "Sucessfully loaded resampler.so" << std::endl;
    return true;
}

bool HFNetTFModel::LoadHFNetTFModel(const std::string &strModelDir)
{
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

void HFNetTFModel::Mat2Tensor(const cv::Mat &image, tensorflow::Tensor *tensor)
{
    float *p = tensor->flat<float>().data();
    cv::Mat imagepixel(image.rows, image.cols, CV_32F, p);
    image.convertTo(imagepixel, CV_32F);
}

#endif

}