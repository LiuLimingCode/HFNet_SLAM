/**
 * To test the tensorflow api, and the base function of HFNet
 * 
 * Result:

 */
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>

#include "Settings.h"
#include "Frame.h"
#include "Extractors/HFNetTFModelV2.h"
#include "Extractors/HFextractor.h"
#include "Examples/Utility/utility_common.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;
using namespace tensorflow;

Settings *settings;
HFNetTFModelV2 *pModel;
TicToc timerDetect;
TicToc timerRun;

void Mat2Tensor(const cv::Mat &image, tensorflow::Tensor *tensor)
{
    float *p = tensor->flat<float>().data();
    cv::Mat imagepixel(image.rows, image.cols, CV_32F, p);
    image.convertTo(imagepixel, CV_32F);
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

bool DetectOnlyLocal(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors,
                     int nKeypointsNum, int nRadius, float threshold)
{
    vKeypoints.clear();
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    static Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    // cout << "Data copy costs: " << timer.Toc() << endl;
    
    timerRun.Tic();
    vector<Tensor> outputs;
    Status status = pModel->mSession->Run({{"image:0", tImage},{"pred/simple_nms/radius", tRadius},},
                                   {"scores_dense:0", "local_descriptor_map:0"}, {}, &outputs);
    timerRun.Toc();
    if (!status.ok()) return false;

    auto vResScoresDense = outputs[0].tensor<float, 3>(); // shape: [1 image.height image.width]
    auto vResLocalDescriptorMap = outputs[1].tensor<float, 4>();

    const int width = outputs[0].shape().dim_size(2), height = outputs[0].shape().dim_size(1);
    const float scaleWidth = (outputs[1].shape().dim_size(2) - 1.f) / (float)(outputs[0].shape().dim_size(2) - 1.f);
    const float scaleHeight = (outputs[1].shape().dim_size(1) - 1.f) / (float)(outputs[0].shape().dim_size(1) - 1.f);

    // timer.Tic();
    cv::KeyPoint keypoint;
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
    // cout << "Threshold costs: " << timer.Toc() << endl;

    vKeyPointsGood = NMS(vKeyPointsGood, width, height, nRadius);

    // timer.Tic();
    if (vKeyPointsGood.size() > nKeypointsNum)
    {
        vKeypoints = DistributeOctTree(vKeyPointsGood, 0, width, 0, height, nKeypointsNum);
    }
    else vKeypoints = vKeyPointsGood;
    // cout << "OctTree costs: " << timer.Toc() << endl;

    localDescriptors = cv::Mat(vKeypoints.size(), 256, CV_32F);
    Tensor tWarp(DT_FLOAT, TensorShape({(int)vKeypoints.size(), 2}));
    auto pWarp = tWarp.tensor<float, 2>();
    for (int temp = 0; temp < vKeypoints.size(); ++temp)
    {
        pWarp(temp * 2 + 0) = scaleWidth * vKeypoints[temp].pt.x;
        pWarp(temp * 2 + 1) = scaleHeight * vKeypoints[temp].pt.y;
    }

    // timer.Tic();
    ResamplerTF(outputs[1], tWarp, localDescriptors);
    // cout << "Resampler cost: " << timer.Toc() << endl;

    // timer.Tic();
    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }
    // cout << "Normalize cost: " << timer.Toc() << endl;
    return true;
}

bool DetectFull(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                int nKeypointsNum, int nRadius, float threshold)
{
    vKeypoints.clear();
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
    Status status = pModel->mSession->Run({{"image:0", tImage},{"pred/simple_nms/radius", tRadius},},
                                   {"scores_dense", "local_descriptor_map", "global_descriptor"}, {}, &outputs);
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

    // timer.Tic();
    cv::KeyPoint keypoint;
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
    // cout << "Threshold costs: " << timer.Toc() << endl;

    // vKeyPointsGood = NMS(vKeyPointsGood, width, height, nRadius);

    // timer.Tic();
    if (vKeyPointsGood.size() > nKeypointsNum)
    {
        vKeypoints = DistributeOctTree(vKeyPointsGood, 0, width, 0, height, nKeypointsNum);
    }
    else vKeypoints = vKeyPointsGood;
    // cout << "OctTree costs: " << timer.Toc() << endl;

    localDescriptors = cv::Mat(vKeypoints.size(), 256, CV_32F);
    Tensor tWarp(DT_FLOAT, TensorShape({(int)vKeypoints.size(), 2}));
    auto pWarp = tWarp.tensor<float, 2>();
    for (int temp = 0; temp < vKeypoints.size(); ++temp)
    {
        pWarp(temp * 2 + 0) = scaleWidth * vKeypoints[temp].pt.x;
        pWarp(temp * 2 + 1) = scaleHeight * vKeypoints[temp].pt.y;
    }

    // timer.Tic();
    ResamplerTF(outputs[1], tWarp, localDescriptors);
    // cout << "Resampler cost: " << timer.Toc() << endl;

    // timer.Tic();
    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }
    // cout << "Normalize cost: " << timer.Toc() << endl;
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

int main(int argc, char* argv[])
{
    settings = new Settings(strSettingsPath, 0);
    pModel = new HFNetTFModelV2(settings->strModelPath());

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(dbStart, files.size() - dbEnd);

    cv::Mat image;
    vector<KeyPoint> vKeypoints;
    cv::Mat localDescriptors, globalDescriptors;
    
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
        else if (command == 'd') threshold += 0.01;
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
        
        DetectFull(image, vKeypoints, localDescriptors, globalDescriptors, 1000, nNMSRadius, threshold);
        cout << "Get features number: " << vKeypoints.size() << endl;
        
        ShowKeypoints("press 'q' to exit", image, vKeypoints);
        cout << endl;
        command = cv::waitKey();
    }
    cv::destroyAllWindows();

    // detect full dataset
    {
        image = imread(strDatasetPath + files[0], IMREAD_GRAYSCALE);
        if (settings->needToResize())
            cv::resize(image, image, settings->newImSize());
        DetectOnlyLocal(image, vKeypoints, localDescriptors, settings->nFeatures(), settings->nNMSRadius(), settings->threshold());
        
        timerDetect.clearBuff();
        timerRun.clearBuff();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            DetectOnlyLocal(image, vKeypoints, localDescriptors, settings->nFeatures(), settings->nNMSRadius(), settings->threshold());
            timerDetect.Toc();
        }
        cout << "Only detect the local keypoints: " << endl
             << "run cost time: " << timerRun.aveCost() << " milliseconds" << endl
             << "detect cost time: " << timerDetect.aveCost() << " milliseconds" << endl;
    }
    {
        image = imread(strDatasetPath + files[0], IMREAD_GRAYSCALE);
        if (settings->needToResize())
            cv::resize(image, image, settings->newImSize());
        DetectFull(image, vKeypoints, localDescriptors, globalDescriptors, settings->nFeatures(), settings->nNMSRadius(), settings->threshold());
        
        timerDetect.clearBuff();
        timerRun.clearBuff();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            DetectFull(image, vKeypoints, localDescriptors, globalDescriptors, settings->nFeatures(), settings->nNMSRadius(), settings->threshold());
            timerDetect.Toc();
        }
        cout << "Detect the full features: " << endl
             << "run cost time: " << timerRun.aveCost() << " milliseconds" << endl
             << "detect cost time: " << timerDetect.aveCost() << " milliseconds" << endl;
    }
    {
        HFextractor extractor = HFextractor(settings->nFeatures(),settings->nNMSRadius(),settings->threshold(),pModel);
        image = imread(strDatasetPath + files[0], IMREAD_GRAYSCALE);
        if (settings->needToResize())
            cv::resize(image, image, settings->newImSize());
        extractor(image, vKeypoints, localDescriptors, globalDescriptors);

        timerDetect.clearBuff();
        timerRun.clearBuff();
        vKeypoints.clear();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            extractor(image, vKeypoints, localDescriptors, globalDescriptors);
            timerDetect.Toc();
        }
        cout << "Detect the full features with HFextractor: " << endl
             << "detect cost time: " << timerDetect.aveCost() << " milliseconds" << endl;
    }

    system("pause");

    return 0;
}