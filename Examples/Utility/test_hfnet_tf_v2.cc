/**
 * To test the tensorflow api, and the base function of HFNet
 * 
 * Result:
 * 
Test with TUM-VI/dataset-corridor4_512_16 Extractor.nFeatures: 1250 Extractor.nNMSRadius: 4 Extractor.threshold: 0.01
======================================
Evaluate the run time perfomance in dataset: 

Only detect the local features: 
run costs: 4.73615 ± 0.375948
detect costs: 5.60672 ± 0.398877
run global costs: 0 ± 0
detect global costs: 0 ± 0

Detect the full features: 
run costs: 6.80395 ± 0.109005
detect costs: 7.69137 ± 0.145479
run global costs: 0 ± 0
detect global costs: 0 ± 0

Detect the full features with intermediate: 
run costs: 4.84162 ± 0.105707
detect costs: 5.86983 ± 0.152857
run global costs: 2.70804 ± 0.128351
detect global costs: 2.78483 ± 0.129311

Detect the full features with pModel [kImageToLocalAndGlobal]: 
run costs: 0 ± 0
detect costs: 7.75988 ± 0.151413
run global costs: 0 ± 0
detect global costs: 0 ± 0

Detect the local features with HFextractor [kImageToLocal]: 
run costs: 0 ± 0
detect costs: 5.6085 ± 0.134683
run global costs: 0 ± 0
detect global costs: 0 ± 0

Detect the local features with HFextractor [kImageToLocalAndIntermediate]: 
run costs: 0 ± 0
detect costs: 5.95985 ± 0.153924
run global costs: 0 ± 0
detect global costs: 2.82719 ± 0.0516818

 */
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>

#include "Settings.h"
#include "Frame.h"
#include "Extractors/HFNetTFModelV2.h"
#include "Examples/Utility/utility_common.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;
using namespace tensorflow;

Settings *settings;
HFNetTFModelV2 *pModelImageToLocalAndGlobal;
HFNetTFModelV2 *pModelImageToLocal;
HFNetTFModelV2 *pModelImageToLocalAndInter;
HFNetTFModelV2 *pModelInterToGlobal;

TicToc timerDetect;
TicToc timerRun;
TicToc timerDetectGlobal;
TicToc timerRunGlobal;

void ClearTimer()
{
    timerDetect.clearBuff();
    timerRun.clearBuff();
    timerDetectGlobal.clearBuff();
    timerRunGlobal.clearBuff();
}

void PrintTimer()
{
    cout << "run costs: " << timerRun.aveCost() << " ± " << timerRun.devCost() << endl;
    cout << "detect costs: " << timerDetect.aveCost() << " ± " << timerDetect.devCost() << endl;
    cout << "run global costs: " << timerRunGlobal.aveCost() << " ± " << timerRunGlobal.devCost() << endl;
    cout << "detect global costs: " << timerDetectGlobal.aveCost() << " ± " << timerDetectGlobal.devCost() << endl;
}

void Mat2Tensor(const cv::Mat &mat, tensorflow::Tensor *tensor)
{
    cv::Mat fromMat(mat.rows, mat.cols, CV_32FC(mat.channels()), tensor->flat<float>().data());
    mat.convertTo(fromMat, CV_32F);
}

void Tensor2Mat(tensorflow::Tensor *tensor, cv::Mat &mat)
{
    const cv::Mat fromTensor(cv::Size(tensor->shape().dim_size(1), tensor->shape().dim_size(2)), CV_32FC(tensor->shape().dim_size(3)), tensor->flat<float>().data());
    mat = fromTensor.clone();
}

void ResamplerTF(const tensorflow::Tensor &data, const tensorflow::Tensor &warp, cv::Mat &output)
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
        Resampler(data.flat<float>().data(),
          warp.flat<float>().data(), output.ptr<float>(), batch_size,
          data_height, data_width, data_channels, num_sampling_points);
    }
}

bool DetectImageToLocal(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                     int nKeypointsNum, float threshold, int nRadius)
{
    vKeyPoints.clear();
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    // cout << "Data copy costs: " << timer.Toc() << endl;
    
    timerRun.Tic();
    vector<Tensor> outputs;
    Status status = pModelImageToLocal->mSession->Run({{"image", tImage},{"pred/simple_nms/radius", tRadius},},
                                   {"scores_dense_nms", "local_descriptor_map"}, {}, &outputs);
    timerRun.Toc();
    if (!status.ok()) return false;

    auto vResScoresDense = outputs[0].tensor<float, 3>(); // shape: [1 image.height image.width]
    auto vResLocalDescriptorMap = outputs[1].tensor<float, 4>();

    const int width = outputs[0].shape().dim_size(2), height = outputs[0].shape().dim_size(1);
    const float scaleWidth = (outputs[1].shape().dim_size(2) - 1.f) / (float)(outputs[0].shape().dim_size(2) - 1.f);
    const float scaleHeight = (outputs[1].shape().dim_size(1) - 1.f) / (float)(outputs[0].shape().dim_size(1) - 1.f);

    cv::KeyPoint keypoint;
    keypoint.angle = 0;
    keypoint.octave = 0;
    vKeyPoints.clear();
    vKeyPoints.reserve(2 * nKeypointsNum);
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
                vKeyPoints.emplace_back(keypoint);
            }
        }
    }

    // vKeyPoints = NMS(vKeyPoints, width, height, nRadius);

    if (vKeyPoints.size() > nKeypointsNum)
    {
        // vKeyPoints = DistributeOctTree(vKeyPoints, 0, width, 0, height, nKeypointsNum);
        std::partial_sort(vKeyPoints.begin(), vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end(), [](const cv::KeyPoint& p1, const cv::KeyPoint& p2) {
            return p1.response > p2.response;
        });
        vKeyPoints.resize(nKeypointsNum);
    }
    
    localDescriptors = cv::Mat(vKeyPoints.size(), 256, CV_32F);
    Tensor tWarp(DT_FLOAT, TensorShape({(int)vKeyPoints.size(), 2}));
    auto pWarp = tWarp.tensor<float, 2>();
    for (size_t temp = 0; temp < vKeyPoints.size(); ++temp)
    {
        pWarp(temp * 2 + 0) = scaleWidth * vKeyPoints[temp].pt.x;
        pWarp(temp * 2 + 1) = scaleHeight * vKeyPoints[temp].pt.y;
    }

    ResamplerTF(outputs[1], tWarp, localDescriptors);

    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }
    
    return true;
}

bool DetectImageToLocalAndGlobal(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                int nKeypointsNum, float threshold, int nRadius)
{
    vKeyPoints.clear();
    TicToc timer;
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    static Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    // cout << "Data copy costs: " << timer.Toc() << endl;
    
    timerRun.Tic();
    vector<Tensor> outputs;
    Status status = pModelImageToLocalAndGlobal->mSession->Run({{"image", tImage},{"pred/simple_nms/radius", tRadius},},
                                   {"scores_dense_nms", "local_descriptor_map", "global_descriptor"}, {}, &outputs);
    timerRun.Toc();
    if (!status.ok()) return false;

    auto vResScoresDense = outputs[0].tensor<float, 3>(); // shape: [1 image.height image.width]
    auto vResLocalDescriptorMap = outputs[1].tensor<float, 4>();
    auto vResGlobalDescriptor = outputs[2].tensor<float, 2>();

    globalDescriptors = cv::Mat(4096, 1, CV_32F);
    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDescriptor(temp);
    }

    const int width = outputs[0].shape().dim_size(2), height = outputs[0].shape().dim_size(1);
    const float scaleWidth = (outputs[1].shape().dim_size(2) - 1.f) / (float)(outputs[0].shape().dim_size(2) - 1.f);
    const float scaleHeight = (outputs[1].shape().dim_size(1) - 1.f) / (float)(outputs[0].shape().dim_size(1) - 1.f);

    cv::KeyPoint keypoint;
    keypoint.angle = 0;
    keypoint.octave = 0;
    vKeyPoints.clear();
    vKeyPoints.reserve(2 * nKeypointsNum);
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
                vKeyPoints.emplace_back(keypoint);
            }
        }
    }

    // vKeyPoints = NMS(vKeyPoints, width, height, nRadius);

    if (vKeyPoints.size() > nKeypointsNum)
    {
        // vKeyPoints = DistributeOctTree(vKeyPoints, 0, width, 0, height, nKeypointsNum);
        std::partial_sort(vKeyPoints.begin(), vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end(), [](const cv::KeyPoint& p1, const cv::KeyPoint& p2) {
            return p1.response > p2.response;
        });
        vKeyPoints.resize(nKeypointsNum);
    }
    
    localDescriptors = cv::Mat(vKeyPoints.size(), 256, CV_32F);
    Tensor tWarp(DT_FLOAT, TensorShape({(int)vKeyPoints.size(), 2}));
    auto pWarp = tWarp.tensor<float, 2>();
    for (size_t temp = 0; temp < vKeyPoints.size(); ++temp)
    {
        pWarp(temp * 2 + 0) = scaleWidth * vKeyPoints[temp].pt.x;
        pWarp(temp * 2 + 1) = scaleHeight * vKeyPoints[temp].pt.y;
    }

    ResamplerTF(outputs[1], tWarp, localDescriptors);

    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }

    return true;
}

bool DetectImageToLocalAndInter(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &preGlobalDescriptors,
                int nKeypointsNum, float threshold, int nRadius)
{
    vKeyPoints.clear();
    TicToc timer;
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    // cout << "Data copy costs: " << timer.Toc() << endl;
    
    timerRun.Tic();
    vector<Tensor> outputs;
    Status status = pModelImageToLocalAndInter->mSession->Run({{"image", tImage},{"pred/simple_nms/radius", tRadius},},
                                   {"scores_dense_nms", "local_descriptor_map", "pred/MobilenetV2/expanded_conv_6/input:0"}, {}, &outputs);
    timerRun.Toc();
    if (!status.ok()) return false;

    auto vResScoresDense = outputs[0].tensor<float, 3>(); // shape: [1 image.height image.width]
    auto vResLocalDescriptorMap = outputs[1].tensor<float, 4>();

    Tensor2Mat(&outputs[2], preGlobalDescriptors);

    const int width = outputs[0].shape().dim_size(2), height = outputs[0].shape().dim_size(1);
    const float scaleWidth = (outputs[1].shape().dim_size(2) - 1.f) / (float)(outputs[0].shape().dim_size(2) - 1.f);
    const float scaleHeight = (outputs[1].shape().dim_size(1) - 1.f) / (float)(outputs[0].shape().dim_size(1) - 1.f);

    cv::KeyPoint keypoint;
    keypoint.angle = 0;
    keypoint.octave = 0;
    vKeyPoints.clear();
    vKeyPoints.reserve(2 * nKeypointsNum);
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
                vKeyPoints.emplace_back(keypoint);
            }
        }
    }

    // vKeyPoints = NMS(vKeyPoints, width, height, nRadius);

    if (vKeyPoints.size() > nKeypointsNum)
    {
        // vKeyPoints = DistributeOctTree(vKeyPoints, 0, width, 0, height, nKeypointsNum);
        std::partial_sort(vKeyPoints.begin(), vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end(), [](const cv::KeyPoint& p1, const cv::KeyPoint& p2) {
            return p1.response > p2.response;
        });
        vKeyPoints.resize(nKeypointsNum);
    }
    
    localDescriptors = cv::Mat(vKeyPoints.size(), 256, CV_32F);
    Tensor tWarp(DT_FLOAT, TensorShape({(int)vKeyPoints.size(), 2}));
    auto pWarp = tWarp.tensor<float, 2>();
    for (size_t temp = 0; temp < vKeyPoints.size(); ++temp)
    {
        pWarp(temp * 2 + 0) = scaleWidth * vKeyPoints[temp].pt.x;
        pWarp(temp * 2 + 1) = scaleHeight * vKeyPoints[temp].pt.y;
    }

    ResamplerTF(outputs[1], tWarp, localDescriptors);

    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }

    return true;
}

bool DetectInterToGlobal(const cv::Mat &preGlobalDescriptors, cv::Mat &globalDescriptors)
{
    Tensor tPreGlobalDescriptors(DT_FLOAT, TensorShape({1, preGlobalDescriptors.rows, preGlobalDescriptors.cols, preGlobalDescriptors.channels()}));
    Mat2Tensor(preGlobalDescriptors, &tPreGlobalDescriptors);

    timerRunGlobal.Tic();
    vector<Tensor> outputs;
    Status status = pModelInterToGlobal->mSession->Run({{"pred/MobilenetV2/expanded_conv_6/input:0", tPreGlobalDescriptors},},
                                          {"global_descriptor"}, {}, &outputs);
    timerRunGlobal.Toc();
    if (!status.ok()) return false;

    auto vResGlobalDescriptor = outputs[0].tensor<float, 2>();
    globalDescriptors = cv::Mat(4096, 1, CV_32F, outputs[0].flat<float>().data()).clone();

    return true;
}

// const string strDatasetPath("/media/llm/Datasets/EuRoC/MH_04_difficult/mav0/cam0/data/");
// const string strSettingsPath("Examples/Monocular-Inertial/EuRoC.yaml");
// const int dbStart = 420;
// const int dbEnd = 50;

const string strDatasetPath("/media/llm/Datasets/TUM-VI/dataset-corridor4_512_16/mav0/cam0/data/");
const string strSettingsPath("Examples/Monocular-Inertial/TUM-VI.yaml");
const int dbStart = 50;
const int dbEnd = 50;

const std::string strTFModelPath("/home/llm/ROS/HFNet_ORBSLAM3_v2/model/hfnet_tf_v2_NMS2/");

int main(int argc, char* argv[])
{
    settings = new Settings(strSettingsPath, 0);
    pModelImageToLocalAndGlobal = new HFNetTFModelV2(strTFModelPath, kImageToLocalAndGlobal, {1,settings->newImSize().height,settings->newImSize().width,1});
    pModelImageToLocal = new HFNetTFModelV2(strTFModelPath, kImageToLocal, {1,settings->newImSize().height,settings->newImSize().width,1});
    pModelImageToLocalAndInter = new HFNetTFModelV2(strTFModelPath, kImageToLocalAndIntermediate, {1,settings->newImSize().height,settings->newImSize().width,1});
    pModelInterToGlobal = new HFNetTFModelV2(strTFModelPath, kIntermediateToGlobal, {1,settings->newImSize().height/8,settings->newImSize().width/8,96});

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(dbStart, files.size() - dbEnd);

    cv::Mat image;
    vector<KeyPoint> vKeyPoints;
    cv::Mat localDescriptors, preGlobalDescriptors, globalDescriptors;
    
    // randomly detect an image and show the results
    char command = ' ';
    float threshold = 0.005;
    int nNMSRadius = 4;
    int select = 0;
    while(1)
    {
        if (command == 'q') break;
        else if (command == 's') select = std::max(select - 1, 0);
        else if (command == 'w') select += 1;
        else if (command == 'a') threshold = std::max(threshold - 0.005, 0.005);
        else if (command == 'd') threshold += 0.005;
        else if (command == 'z') nNMSRadius = std::max(nNMSRadius - 1, 0);
        else if (command == 'c') nNMSRadius += 1;
        else select = distribution(generator);
        cout << "command: " << command << endl;
        cout << "select: " << select << endl;
        cout << "threshold: " << threshold << endl;
        cout << "nNMSRadius: " << nNMSRadius << endl;

        image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        if (settings->needToResize())
            cv::resize(image, image, settings->newImSize());
        
        ClearTimer();
        timerDetect.Tic();
        DetectImageToLocalAndInter(image, vKeyPoints, localDescriptors, preGlobalDescriptors, 1000, threshold, nNMSRadius);
        // if (!pModelImageToLocalAndInter->Detect(image, vKeyPoints, localDescriptors, preGlobalDescriptors, settings->nFeatures(), settings->threshold(), settings->nNMSRadius()))
        //     cerr << "error while detecting!" << endl;
        timerDetect.Toc();
        timerDetectGlobal.Tic();
        DetectInterToGlobal(preGlobalDescriptors, globalDescriptors);
        // if (!pModelInterToGlobal->Detect(preGlobalDescriptors, globalDescriptors))
        //     cerr << "error while detecting!" << endl;
        timerDetectGlobal.Toc();

        cout << "Get features number: " << vKeyPoints.size() << endl;
        PrintTimer();

        cout << localDescriptors.ptr<float>()[0] << endl;
        cout << preGlobalDescriptors.row(50).col(50) << endl;
        cout << globalDescriptors.col(0).rowRange(100, 110) << endl;
        
        ShowKeypoints("press 'q' to exit", image, vKeyPoints);
        cout << endl;
        command = cv::waitKey();
    }
    cv::destroyAllWindows();

    cout << "======================================" << endl
         << "Evaluate the run time perfomance in dataset: " << endl;

    {
        cout << endl;
        ClearTimer();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            DetectImageToLocal(image, vKeyPoints, localDescriptors, settings->nFeatures(), settings->threshold(), settings->nNMSRadius());
            timerDetect.Toc();
        }
        cout << "Only detect the local features: " << endl;
        PrintTimer();
    }

    {
        cout << endl;
        ClearTimer();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            DetectImageToLocalAndGlobal(image, vKeyPoints, localDescriptors, preGlobalDescriptors, settings->nFeatures(), settings->threshold(), settings->nNMSRadius());
            timerDetect.Toc();
        }
        cout << "Detect the full features: " << endl;
        PrintTimer();
    }

    {
        cout << endl;
        ClearTimer();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            DetectImageToLocalAndInter(image, vKeyPoints, localDescriptors, preGlobalDescriptors, settings->nFeatures(), settings->threshold(), settings->nNMSRadius());
            timerDetect.Toc();
            timerDetectGlobal.Tic();
            DetectInterToGlobal(preGlobalDescriptors, globalDescriptors);
            timerDetectGlobal.Toc();
        }
        cout << "Detect the full features with intermediate: " << endl;
        PrintTimer();
    }

    {
        cout << endl;
        ClearTimer();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            if (!pModelImageToLocalAndGlobal->Detect(image, vKeyPoints, localDescriptors, globalDescriptors, settings->nFeatures(), settings->threshold(), settings->nNMSRadius()))
                cerr << "error while detecting!" << endl;
            timerDetect.Toc();
        }
        cout << "Detect the full features with pModel [kImageToLocalAndGlobal]: " << endl;
        PrintTimer();
    }

    {
        cout << endl;
        ClearTimer();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            if (!pModelImageToLocal->Detect(image, vKeyPoints, localDescriptors, settings->nFeatures(), settings->threshold(), settings->nNMSRadius()))
                cerr << "error while detecting!" << endl;
            timerDetect.Toc();
        }
        cout << "Detect the local features with HFextractor [kImageToLocal]: " << endl;
        PrintTimer();
    }

    {
        cout << endl;
        ClearTimer();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            if (!pModelImageToLocalAndInter->Detect(image, vKeyPoints, localDescriptors, preGlobalDescriptors, settings->nFeatures(), settings->threshold(), settings->nNMSRadius()))
                cerr << "error while detecting!" << endl;
            timerDetect.Toc();
            timerDetectGlobal.Tic();
            if (!pModelInterToGlobal->Detect(preGlobalDescriptors, globalDescriptors))
                cerr << "error while detecting!" << endl;
            timerDetectGlobal.Toc();
        }
        cout << "Detect the local features with HFextractor [kImageToLocalAndIntermediate]: " << endl;
        PrintTimer();
    }

    cout << endl << "Press 'ENTER' to exit" << endl;
    getchar();

    return 0;
}