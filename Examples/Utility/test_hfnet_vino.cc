/**
 * To test the VINO api, and the base function of HFNet
 * 
 * Result:
======================================
Evaluate the run time perfomance in dataset: 

Only detect the local keypoints: 
run costs: 13.7171 ± 1.63629
detect costs: 15.4973 ± 1.84119
run global costs: 0 ± 0
detect global costs: 0 ± 0

Detect the full features with intermediate: 
run costs: 14.7501 ± 0.2554
detect costs: 16.8641 ± 0.363727
run global costs: 8.9496 ± 0.124993
detect global costs: 9.101 ± 0.12592

Detect the full features: 
run costs: 22.6395 ± 0.300341
detect costs: 24.7721 ± 0.403894
run global costs: 0 ± 0
detect global costs: 0 ± 0

Detect the local features with HFextractor [kImageToLocal]: 
run costs: 0 ± 0
detect costs: 15.4288 ± 0.393785
run global costs: 0 ± 0
detect global costs: 0 ± 0

Detect the local features with HFextractor [kImageToLocalAndIntermediate]: 
run costs: 0 ± 0
detect costs: 16.8607 ± 0.314796
run global costs: 0 ± 0
detect global costs: 9.10962 ± 0.118789
 */
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>

#include "Settings.h"
#include "Frame.h"
#include "Extractors/HFNetVINOModel.h"
#include "Examples/Utility/utility_common.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;
using namespace ov;

Settings *settings;
HFNetVINOModel *pModelImageToLocalAndGlobal;
HFNetVINOModel *pModelImageToLocal;
HFNetVINOModel *pModelImageToLocalAndInter;
HFNetVINOModel *pModelInterToGlobal;

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

void Mat2Tensor(const cv::Mat &mat, ov::Tensor *tensor)
{
    cv::Mat fromMat(mat.rows, mat.cols, CV_32FC(mat.channels()), tensor->data<float>());
    mat.convertTo(fromMat, CV_32F);
}

void Tensor2Mat(ov::Tensor *tensor, cv::Mat &mat)
{
    const cv::Mat fromTensor(cv::Size(tensor->get_shape()[1], tensor->get_shape()[2]), CV_32FC(tensor->get_shape()[3]), tensor->data<float>());
    fromTensor.convertTo(mat, CV_32F);
}

void ResamplerOV(const ov::Tensor &data, const ov::Tensor &warp, cv::Mat &output)
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

bool DetectImageToLocal(HFNetVINOModel *pModel, const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                            int nKeypointsNum, float threshold, int nRadius)
{
    vKeyPoints.clear();

    ov::Tensor inputTensor = pModel->mInferRequest->get_input_tensor();
    ov::Shape inputShape = inputTensor.get_shape();
    if (inputShape[2] != image.cols || inputShape[1] != image.rows || inputShape[3] != image.channels())
    {
        cerr << "The input shape in VINO model should be the same as the compile shape" << endl;
        return false;
    }

    Mat2Tensor(image, &inputTensor);
    
    timerRun.Tic();
    pModel->mInferRequest->infer();
    timerRun.Toc();

    ov::Tensor tscoreDense = pModel->mInferRequest->get_tensor("pred/local_head/detector/Squeeze:0");
    ov::Tensor tLocalDescriptorMap = pModel->mInferRequest->get_tensor("local_descriptor_map");

    const int width = tscoreDense.get_shape()[2], height = tscoreDense.get_shape()[1];
    const float scaleWidth = (tLocalDescriptorMap.get_shape()[2] - 1.f) / (float)(tscoreDense.get_shape()[2] - 1.f);
    const float scaleHeight = (tLocalDescriptorMap.get_shape()[1] - 1.f) / (float)(tscoreDense.get_shape()[1] - 1.f);

    auto vResScoresDense = tscoreDense.data<float>();
    cv::KeyPoint keypoint;
    vKeyPoints.reserve(2 * nKeypointsNum);
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
                vKeyPoints.emplace_back(keypoint);
            }
        }
    }

    vKeyPoints = NMS(vKeyPoints, width, height, nRadius);

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

bool DetectImageToLocalAndGlobal(HFNetVINOModel *pModel, const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                            int nKeypointsNum, float threshold, int nRadius)
{
    vKeyPoints.clear();

    ov::Tensor inputTensor = pModel->mInferRequest->get_input_tensor();
    ov::Shape inputShape = inputTensor.get_shape();
    if (inputShape[2] != image.cols || inputShape[1] != image.rows || inputShape[3] != image.channels())
    {
        cerr << "The input shape in VINO model should be the same as the compile shape" << endl;
        return false;
    }

    Mat2Tensor(image, &inputTensor);
    
    timerRun.Tic();
    pModel->mInferRequest->infer();
    timerRun.Toc();

    ov::Tensor tscoreDense = pModel->mInferRequest->get_tensor("pred/local_head/detector/Squeeze:0");
    ov::Tensor tLocalDescriptorMap = pModel->mInferRequest->get_tensor("local_descriptor_map");
    ov::Tensor tGlobalDescriptor = pModel->mInferRequest->get_tensor("global_descriptor");

    auto vResGlobalDescriptor = tGlobalDescriptor.data<float>();
    globalDescriptors = cv::Mat(4096, 1, CV_32F);
    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDescriptor[temp];
    }

    const int width = tscoreDense.get_shape()[2], height = tscoreDense.get_shape()[1];
    const float scaleWidth = (tLocalDescriptorMap.get_shape()[2] - 1.f) / (float)(tscoreDense.get_shape()[2] - 1.f);
    const float scaleHeight = (tLocalDescriptorMap.get_shape()[1] - 1.f) / (float)(tscoreDense.get_shape()[1] - 1.f);

    auto vResScoresDense = tscoreDense.data<float>();
    cv::KeyPoint keypoint;
    vKeyPoints.reserve(2 * nKeypointsNum);
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
                vKeyPoints.emplace_back(keypoint);
            }
        }
    }

    vKeyPoints = NMS(vKeyPoints, width, height, nRadius);

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

bool DetectImageToLocalAndInter(HFNetVINOModel *pModel, const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors, cv::Mat &preGlobalDescriptors,
                int nKeypointsNum, float threshold, int nRadius)
{
    vKeyPoints.clear();

    ov::Tensor inputTensor = pModel->mInferRequest->get_input_tensor();
    ov::Shape inputShape = inputTensor.get_shape();
    if (inputShape[2] != image.cols || inputShape[1] != image.rows || inputShape[3] != image.channels())
    {
        cerr << "The input shape in VINO model should be the same as the compile shape" << endl;
        return false;
    }

    Mat2Tensor(image, &inputTensor);
    
    timerRun.Tic();
    pModel->mInferRequest->infer();
    timerRun.Toc();

    ov::Tensor tscoreDense = pModel->mInferRequest->get_tensor("pred/local_head/detector/Squeeze:0");
    ov::Tensor tLocalDescriptorMap = pModel->mInferRequest->get_tensor("local_descriptor_map");
    ov::Tensor tIntermediate = pModel->mInferRequest->get_tensor("pred/MobilenetV2/expanded_conv_6/input:0");

    Tensor2Mat(&tIntermediate, preGlobalDescriptors);

    const int width = tscoreDense.get_shape()[2], height = tscoreDense.get_shape()[1];
    const float scaleWidth = (tLocalDescriptorMap.get_shape()[2] - 1.f) / (float)(tscoreDense.get_shape()[2] - 1.f);
    const float scaleHeight = (tLocalDescriptorMap.get_shape()[1] - 1.f) / (float)(tscoreDense.get_shape()[1] - 1.f);

    auto vResScoresDense = tscoreDense.data<float>();
    cv::KeyPoint keypoint;
    vKeyPoints.reserve(2 * nKeypointsNum);
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
                vKeyPoints.emplace_back(keypoint);
            }
        }
    }

    vKeyPoints = NMS(vKeyPoints, width, height, nRadius);

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

bool DetectInterToGlobal(HFNetVINOModel *pModel, const cv::Mat &preGlobalDescriptors, cv::Mat &globalDescriptors)
{
    ov::Tensor inputTensor = pModel->mInferRequest->get_input_tensor();
    ov::Shape inputShape = inputTensor.get_shape();
    if (inputShape[2] != preGlobalDescriptors.cols || inputShape[1] != preGlobalDescriptors.rows || inputShape[3] != preGlobalDescriptors.channels())
    {
        cerr << "The input shape in VINO model should be the same as the compile shape" << endl;
        return false;
    }

    Mat2Tensor(preGlobalDescriptors, &inputTensor);

    timerRunGlobal.Tic();
    pModel->mInferRequest->infer();
    timerRunGlobal.Toc();

    ov::Tensor tGlobalDescriptor = pModel->mInferRequest->get_tensor("global_descriptor");

    auto vResGlobalDescriptor = tGlobalDescriptor.data<float>();
    globalDescriptors = cv::Mat::zeros(4096, 1, CV_32F);
    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDescriptor[temp];
    }

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

const std::string strLocalModelPath("/home/llm/ROS/HFNet_ORBSLAM3_v2/model/hfnet_vino_local_f32/");
const std::string strGlobalModelPath("/home/llm/ROS/HFNet_ORBSLAM3_v2/model/hfnet_vino_global_f32/");
const std::string strFullModelPath("/home/llm/ROS/HFNet_ORBSLAM3_v2/model/hfnet_vino_full_f32/");

int main(int argc, char* argv[])
{
    settings = new Settings(strSettingsPath, 0);
    pModelImageToLocal = new HFNetVINOModel(strLocalModelPath + "/saved_model.xml", strLocalModelPath + "/saved_model.bin", kImageToLocal, {1, settings->newImSize().height, settings->newImSize().width, 1});
    pModelImageToLocalAndInter = new HFNetVINOModel(strLocalModelPath + "/saved_model.xml", strLocalModelPath + "/saved_model.bin", kImageToLocalAndIntermediate, {1, settings->newImSize().height, settings->newImSize().width, 1});
    pModelInterToGlobal = new HFNetVINOModel(strGlobalModelPath + "/saved_model.xml", strGlobalModelPath + "/saved_model.bin", kIntermediateToGlobal, {1, settings->newImSize().height/8, settings->newImSize().width/8, 96});
    pModelImageToLocalAndGlobal = new HFNetVINOModel(strFullModelPath + "/saved_model.xml", strFullModelPath + "/saved_model.bin", kImageToLocalAndGlobal, {1, settings->newImSize().height, settings->newImSize().width, 1});

    pModelImageToLocal->PrintInputAndOutputsInfo();
    pModelInterToGlobal->PrintInputAndOutputsInfo();

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(dbStart, files.size() - dbEnd);

    cv::Mat image;
    vector<KeyPoint> vKeyPoints;
    cv::Mat localDescriptors, globalDescriptors, preGlobalDescriptors;
    
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
        // DetectImageToLocalAndInter(pModelImageToLocalAndInter, image, vKeyPoints, localDescriptors, preGlobalDescriptors, settings->nFeatures(), settings->threshold(), settings->nNMSRadius()); 
        if (!pModelImageToLocalAndInter->Detect(image, vKeyPoints, localDescriptors, preGlobalDescriptors, settings->nFeatures(), settings->threshold(), settings->nNMSRadius()))
            cerr << "error while detecting!" << endl;
        timerDetect.Toc();
        timerDetectGlobal.Tic();
        // DetectInterToGlobal(pModelInterToGlobal, preGlobalDescriptors, globalDescriptors);
        if (!pModelInterToGlobal->Detect(preGlobalDescriptors, globalDescriptors))
            cerr << "error while detecting!" << endl;
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
            DetectImageToLocal(pModelImageToLocal, image, vKeyPoints, localDescriptors, settings->nFeatures(), settings->threshold(), settings->nNMSRadius());
            timerDetect.Toc();
        }
        cout << "Only detect the local keypoints: " << endl;
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
            DetectImageToLocalAndInter(pModelImageToLocalAndInter, image, vKeyPoints, localDescriptors, preGlobalDescriptors, settings->nFeatures(), settings->threshold(), settings->nNMSRadius());
            timerDetect.Toc();
            timerDetectGlobal.Tic();
            DetectInterToGlobal(pModelInterToGlobal, preGlobalDescriptors, globalDescriptors);
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
            DetectImageToLocalAndGlobal(pModelImageToLocalAndGlobal, image, vKeyPoints, localDescriptors, globalDescriptors, settings->nFeatures(), settings->threshold(), settings->nNMSRadius());
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
