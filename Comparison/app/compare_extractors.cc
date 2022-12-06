#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>

#include "../include/Extractors/HFextractor.h"
#include "ORBextractor.h"

#include "../include/utility_common.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;

int main(int argc, char* argv[])
{
    if (argc != 3) {
        cerr << endl << "Usage: compare_extractors path_to_dataset path_to_model" << endl;
        return -1;
    }
    const string strDatasetPath = string(argv[1]);
    const string strModelPath = string(argv[2]);

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    if (files.empty()) {
        cout << "Error, failed to find any valid image in: " << strDatasetPath << endl;
        return 1;
    }
    cv::Size ImSize = imread(strDatasetPath + files[0], IMREAD_GRAYSCALE).size();
    if (ImSize.area() == 0) {
        cout << "Error, failed to read the image at: " << strDatasetPath + files[0] << endl;
        return 1;
    }

    const int nLevels = 4;
    const float scaleFactor = 1.2f;
    InitAllModels(strModelPath, kHFNetRTModel, ImSize, nLevels, scaleFactor);
    // InitAllModels(strModelPath, kHFNetTFModel, ImSize, nLevels, scaleFactor);
    auto vpModels = GetModelVec();

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

    char command = 0;
    float threshold = 0.01;
    int nFeatures = 1000;
    int select = 0;
    while(1)
    {
        if (command == 'x') break;
        else if (command == 'a') threshold = std::max(threshold - 0.001, 0.005);
        else if (command == 'd') threshold += 0.001;
        else if (command == 's') select = std::max(select - 1, 0);
        else if (command == 'w') select += 1;
        else if (command == 'q') nFeatures = std::max(nFeatures - 200, 0);
        else if (command == 'e') nFeatures += 200;
        else if (command == ' ') select = distribution(generator);
        cout << "command: " << command << endl;
        cout << "select: " << select << endl;
        cout << "nFeatures: " << nFeatures << endl;
        cout << "threshold: " << threshold << endl;

        image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);

        {
            cout << "============= ORB-SLAM3 =============" << endl;
            auto t1 = chrono::steady_clock::now();
            ORBextractor extractor(nFeatures, scaleFactor, 8, 20, 7);
            extractor(image, cv::Mat(), keypoints, localDescripotrs, vLapping);
            auto t2 = chrono::steady_clock::now();
            auto t = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            cout << "cost time: " << t << endl;
            ShowKeypoints("ORB-SLAM3", image, keypoints);
            cout << "key point number: " << keypoints.size() << endl;
        }

        {
            cout << "============= HFNet-SLAM =============" << endl;
            HFextractor extractorHF(nFeatures, threshold, scaleFactor, nLevels, vpModels);
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