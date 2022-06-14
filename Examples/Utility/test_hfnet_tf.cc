/**
 * To test the tensorflow api, and the base function of HFNet
 * 
 * Result:
session->Run() output size: 4
outputs[0].shape(): [1,1000,2]
outputs[1].shape(): [1,1000,256]
outputs[2].shape(): [1,1000]
outputs[3].shape(): [1,4096]
 * 
Only detect the local keypoints: 
cost time: 19445 milliseconds
average detect time: 9.56468

Only detect the full features: 
cost time: 24720 milliseconds
average detect time: 12.1594

Only detect the full features: 
cost time: 24961 milliseconds
average detect time: 12.2779
 *
 */
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>

#include "Settings.h"
#include "Extractors/HFNetTFModel.h"
#include "Extractors/HFextractor.h"
#include "Examples/Utility/utility_common.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;
using namespace tensorflow;

Settings *settings;
HFNetTFModel *pModel;

void Mat2Tensor(const cv::Mat &image, tensorflow::Tensor *tensor)
{
    float *p = tensor->flat<float>().data();
    cv::Mat imagepixel(image.rows, image.cols, CV_32F, p);
    image.convertTo(imagepixel, CV_32F);
}

bool DetectOnlyLocal(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors,
                     int nKeypointsNum = 1000, float threshold = 0.02, int nRadius = 4)
{
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    
    vector<Tensor> outputs;
    Status status = pModel->mSession->Run({{"image:0", tImage},{"pred/simple_nms/radius", tRadius},{"pred/top_k_keypoints/k", tKeypointsNum}},
                                          {"keypoints", "local_descriptors", "scores"}, {}, &outputs);
    if (!status.ok()) return false;

     int nResNumber = outputs[0].shape().dim_size(1);

    auto vResKeypoints = outputs[0].tensor<int32, 3>();
    auto vResLocalDes = outputs[1].tensor<float, 3>();
    auto vResScores = outputs[2].tensor<float, 2>();

    vKeypoints.clear();
    localDescriptors = cv::Mat::zeros(nResNumber, 256, CV_32F);

    KeyPoint kp;
    for(int index = 0; index < nResNumber; index++)
    {
        if (vResScores(index) < threshold) continue;
        kp.pt = Point2f(vResKeypoints(2 * index), vResKeypoints(2 * index + 1));
        kp.response = vResScores(index);
        vKeypoints.emplace_back(kp);
        for (int temp = 0; temp < 256; ++temp)
        {
            localDescriptors.ptr<float>(index)[temp] = vResLocalDes(256 * index + temp); 
        }
    }
    return true;
}

bool DetectFull(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                int nKeypointsNum = 1000, float threshold = 0.02, int nRadius = 4)
{
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    
    vector<Tensor> outputs;
    Status status = pModel->mSession->Run({{"image:0", tImage},{"pred/simple_nms/radius", tRadius},{"pred/top_k_keypoints/k", tKeypointsNum}},
                                          {"keypoints", "local_descriptors", "scores", "global_descriptor"}, {}, &outputs);
    if (!status.ok()) return false;

    int nResNumber = outputs[0].shape().dim_size(1);

    auto vResKeypoints = outputs[0].tensor<int32, 3>();
    auto vResLocalDes = outputs[1].tensor<float, 3>();
    auto vResScores = outputs[2].tensor<float, 2>();
    auto vResGlobalDes = outputs[3].tensor<float, 2>();

    // cout << "session->Run() output size: " << outputs.size() << endl;
    // cout << "outputs[0].shape(): " << outputs[0].shape() << endl;
    // cout << "outputs[1].shape(): " << outputs[1].shape() << endl;
    // cout << "outputs[2].shape(): " << outputs[2].shape() << endl;
    // cout << "outputs[3].shape(): " << outputs[3].shape() << endl;

    vKeypoints.clear();
    localDescriptors = cv::Mat::zeros(nResNumber, 256, CV_32F);
    globalDescriptors = cv::Mat::zeros(4096, 1, CV_32F);

    KeyPoint kp;
    for(int index = 0; index < nResNumber; index++)
    {
        if (vResScores(index) < threshold) continue;
        kp.pt = Point2f(vResKeypoints(2 * index), vResKeypoints(2 * index + 1));
        kp.response = vResScores(index);
        vKeypoints.emplace_back(kp);
        for (int temp = 0; temp < 256; ++temp)
        {
            localDescriptors.ptr<float>(index)[temp] = vResLocalDes(256 * index + temp); 
        }
    }
    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDes(temp);
    }
    return true;
}

bool DetectOnlyGlobal(const cv::Mat &image, cv::Mat &globalDescriptors,
                      int nKeypointsNum = 1000, int nRadius = 4)
{
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    
    vector<Tensor> outputs;
    Status status = pModel->mSession->Run({{"image:0", tImage},{"pred/simple_nms/radius", tRadius},{"pred/top_k_keypoints/k", tKeypointsNum}},
                                          {"global_descriptor"}, {}, &outputs);
    if (!status.ok()) return false;

    auto vResGlobalDes = outputs[0].tensor<float, 2>();

    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDes(temp);
    }
    return true;
}

int main(int argc, char* argv[])
{
    const string strDatasetPath("/media/llm/Datasets/EuRoC/MH_04_difficult/mav0/cam0/data/");
    const string strSettingsPath("Examples/Monocular-Inertial/EuRoC.yaml");

    settings = new Settings(strSettingsPath, 0);
    pModel = new HFNetTFModel(settings->strResamplerPath(), settings->strModelPath());

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(420, files.size());

    cv::Mat image;
    vector<KeyPoint> vKeypoints;
    cv::Mat localDescriptors, globalDescriptors;
    
    // randomly detect an image and show the results
    char command = ' ';
    float threshold = 0;
    int nNMSRadius = 4;
    int select = 0;
    while(1)
    {
        if (command == 'q') break;
        else if (command == 'a') threshold = std::max(threshold - 0.01, 0.0);
        else if (command == 'd') threshold += 0.01;
        else if (command == 's') select = std::max(select - 1, 0);
        else if (command == 'w') select += 1;
        else if (command == 'z') nNMSRadius = std::max(nNMSRadius - 1, 0);
        else if (command == 'c') nNMSRadius += 1;
        else select = distribution(generator);
        cout << "command: " << command << endl;
        cout << "select: " << select << endl;
        cout << "threshold: " << threshold << endl;
        cout << "nNMSRadius: " << nNMSRadius << endl;

        image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        
        DetectFull(image, vKeypoints, localDescriptors, globalDescriptors, 1000, threshold, nNMSRadius);
        cout << "Get features number: " << vKeypoints.size() << endl;
        
        ShowKeypoints("press 'q' to exit", image, vKeypoints);
        cout << endl;
        command = cv::waitKey();
    }
    cv::destroyAllWindows();

    // detect full dataset
    {
        auto t1 = chrono::steady_clock::now();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            DetectOnlyLocal(image, vKeypoints, localDescriptors);
        }
        auto t2 = chrono::steady_clock::now();
        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        cout << "Only detect the local keypoints: " << endl
             << "cost time: " << t << " milliseconds" << endl
             << "average detect time: " << (double)t / files.size() << endl << endl;
    }
    {
        auto t1 = chrono::steady_clock::now();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            DetectFull(image, vKeypoints, localDescriptors, globalDescriptors);
        }
        auto t2 = chrono::steady_clock::now();
        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        cout << "Detect the full features: " << endl
             << "cost time: " << t << " milliseconds" << endl
             << "average detect time: " << (double)t / files.size() << endl << endl;
    }
    {
        HFextractor extractor = HFextractor(settings->nFeatures(),settings->nNMSRadius(),settings->threshold(),settings->scaleFactor(),settings->nLevels(),pModel);
        auto t1 = chrono::steady_clock::now();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            extractor(image, vKeypoints, localDescriptors, globalDescriptors);
        }
        auto t2 = chrono::steady_clock::now();
        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        cout << "Detect the full features with HFextractor: " << endl
             << "cost time: " << t << " milliseconds" << endl
             << "average detect time: " << (double)t / files.size() << endl << endl;
    }


    // cout << "detect: {\"global_descriptor\"}" << endl;
    // t1 = chrono::steady_clock::now();
    // for (const string& file : files)
    // {
    //     image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
    //     DetectOnlyGlobal(image, globalDescriptors);
    // }
    // t2 = chrono::steady_clock::now();
    // t = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    // std::cout << "cost time: " << t << " milliseconds" << endl;
    // std::cout << "average detect time: " << (double)t / files.size() << endl;

    system("pause");

    return 0;
}