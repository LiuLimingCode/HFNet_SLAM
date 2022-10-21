/**
 * To Test the performance of different dector
 * 
 * What we found in this test:
 * 1. HFNet do not need OctTree as it has NMS
 * 2. It is not necessary to calculate the response of keypoints, because it is only used in OctTree
 */
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>

#include "../include/Settings.h"
#include "../include/Extractors/HFNetTFModelV2.h"

#include "ORBextractor.h"

#include "app/utility_common.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;

Settings *settings;

const string strDatasetPath("/media/llm/Datasets/EuRoC/MH_01_easy/mav0/cam0/data/");
const string strSettingsPath("Examples/Monocular-Inertial/EuRoC.yaml");
const int dbStart = 420;
const int dbEnd = 50;

// const string strDatasetPath("/media/llm/Datasets/TUM-VI/dataset-corridor4_512_16/mav0/cam0/data/");
// const string strSettingsPath("/home/llm/ROS/HFNet_SLAM/Comparison/app/TUM-VI.yaml");
// const string strTFModelPath("/home/llm/ROS/HFNet_SLAM/model/hfnet_tf_v2_NMS2");
// const int dbStart = 50;
// const int dbEnd = 50;

const string strTFModelPath("/home/llm/ROS/HFNet_SLAM/model/hfnet_tf_v2_NMS2");

int main(int argc, char* argv[])
{

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    settings = new Settings(strSettingsPath, 0);

    vector<BaseModel*> vpModels;
    cv::Size ImSize = settings->newImSize();
    float scale = 1.0;
    for (int level = 0; level < settings->nLevels(); ++level)
    {
        cv::Vec4i inputShape{1, cvRound(ImSize.height * scale), cvRound(ImSize.width * scale), 1};
        BaseModel *pNewModel;
        if (level == 0) pNewModel = new HFNetTFModelV2(strTFModelPath, kImageToLocalAndIntermediate, inputShape);
        else pNewModel = new HFNetTFModelV2(strTFModelPath, kImageToLocal, inputShape);
        vpModels.emplace_back(pNewModel);
        scale /= settings->scaleFactor();
    }

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(0, files.size() - 1);

    cv::Mat image;
    vector<KeyPoint> keypoints;
    cv::Mat localDescripotrs, globlaDescriptors;
    vector<int> vLapping = {0,1000};

    cv::namedWindow("ORB-SLAM3");
    cv::moveWindow("ORB-SLAM3", 0, 0);
    cv::namedWindow("HFNet-SLAM");
    cv::moveWindow("HFNet-SLAM", 0, 540);

    char command = ' ';
    float threshold = 0.01;
    int nNMSRadius = 4;
    int nFeatures = 1000;
    int select = 0;
    while(1)
    {
        if (command == 'x') break;
        else if (command == 'a') threshold = std::max(threshold - 0.001, 0.005);
        else if (command == 'd') threshold += 0.001;
        else if (command == 's') select = std::max(select - 1, 0);
        else if (command == 'w') select += 1;
        else if (command == 'z') nNMSRadius = std::max(nNMSRadius - 1, 0);
        else if (command == 'c') nNMSRadius += 1;
        else if (command == 'q') nFeatures = std::max(nFeatures - 200, 0);
        else if (command == 'e') nFeatures += 200;
        else select = distribution(generator);
        cout << "command: " << command << endl;
        cout << "select: " << select << endl;
        cout << "nFeatures: " << nFeatures << endl;
        cout << "threshold: " << threshold << endl;
        cout << "nNMSRadius: " << nNMSRadius << endl;

        image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        if (settings->needToResize())
            cv::resize(image, image, settings->newImSize());

        {
            cout << "============= ORB-SLAM3 =============" << endl;
            auto t1 = chrono::steady_clock::now();
            ORBextractor extractor(nFeatures, settings->scaleFactor(), settings->nLevels(), 20, 7);
            extractor(image, cv::Mat(), keypoints, localDescripotrs, vLapping);
            auto t2 = chrono::steady_clock::now();
            auto t = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            cout << "cost time: " << t << endl;
            ShowKeypoints("ORB-SLAM3", image, keypoints);
            cout << "key point number: " << keypoints.size() << endl;
        }

        {
            cout << "============= HFNet-SLAM =============" << endl;
            HFextractor extractorHF(nFeatures, threshold, nNMSRadius, settings->scaleFactor(), settings->nLevels(), vpModels);
            auto t1 = chrono::steady_clock::now();
            extractorHF(image, keypoints, localDescripotrs, globlaDescriptors);
            auto t2 = chrono::steady_clock::now();
            auto t = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            cout << "cost time: " << t << endl;
            ShowKeypoints("HFNet-SLAM", image, keypoints);
            cout << "key point number: " << keypoints.size() << endl;
        }

        command = cv::waitKey();
    };

    system("pause");

    return 0;
}