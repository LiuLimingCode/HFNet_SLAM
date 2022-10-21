/**
 * To Test the performance of different dector
 * 
======================================
Evaluate the run time perfomance in dataset: 

Detect the full features with TestExtractor ExtractUsingParallel(): 
Get features number: 728
pyramid costs: 1.30138 ± 1.69273
level 0 costs : 170.373 ± 42.9945
level 1 costs : 0 ± 0
level 2 costs : 0 ± 0
level 3 costs : 0 ± 0
level 4 costs : 0 ± 0
level 5 costs : 0 ± 0
level 6 costs : 0 ± 0
level 7 costs : 0 ± 0
copy costs: 0.104862 ± 0.0486709
total costs: 171.792 ± 43.0424

Detect the full features with TestExtractor ExtractUsingFor(): 
Get features number: 1123
pyramid costs: 2.41067 ± 1.14057
level 0 costs : 28.6432 ± 0.660343
level 1 costs : 9.86764 ± 0.401312
level 2 costs : 6.57248 ± 0.245541
level 3 costs : 4.60928 ± 0.149062
level 4 costs : 3.01519 ± 0.108606
level 5 costs : 2.18284 ± 0.0931418
level 6 costs : 1.62319 ± 0.0720224
level 7 costs : 1.1764 ± 0.0617994
copy costs: 0.174835 ± 0.113503
total costs: 60.2931 ± 1.74302

Detect the full features with HFextractor: 
Get features number: 728
pyramid costs: 0 ± 0
level 0 costs : 0 ± 0
level 1 costs : 0 ± 0
level 2 costs : 0 ± 0
level 3 costs : 0 ± 0
level 4 costs : 0 ± 0
level 5 costs : 0 ± 0
level 6 costs : 0 ± 0
level 7 costs : 0 ± 0
copy costs: 0 ± 0
total costs: 60.0883 ± 4.10698
 */
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>

#include "Settings.h"
#include "Extractors/HFNetVINOModel.h"
#include "Extractors/HFextractor.h"
#include "Examples/Utility/utility_common.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;

Settings *settings;

// const string strDatasetPath("/media/llm/Datasets/EuRoC/MH_01_easy/mav0/cam0/data/");
// const string strSettingsPath("Examples/Monocular-Inertial/EuRoC.yaml");
// const int dbStart = 420;
// const int dbEnd = 50;

const string strDatasetPath("/media/llm/Datasets/TUM-VI/dataset-corridor4_512_16/mav0/cam0/data/");
const string strSettingsPath("Examples/Monocular-Inertial/TUM-VI.yaml");
const int dbStart = 50;
const int dbEnd = 50;

const std::string strXmlPath("/home/llm/ROS/HFNet_SLAM/model/hfnet_vino_local_f16/saved_model.xml");
const std::string strBinPath("/home/llm/ROS/HFNet_SLAM/model/hfnet_vino_local_f16/saved_model.bin");
const int nLevels = 8;
const float scaleFactor = 1.2;

TicToc TimerPyramid;
TicToc TimerDetectPerLevel[nLevels];
TicToc TimerCopy;
TicToc TimerTotal;

void ClearTimer()
{
    TimerPyramid.clearBuff();
    for (auto &timer : TimerDetectPerLevel) timer.clearBuff();
    TimerCopy.clearBuff();
    TimerTotal.clearBuff();
}

void PrintTimer()
{
    cout << "pyramid costs: " << TimerPyramid.aveCost() << " ± " << TimerPyramid.devCost() << endl;
    for (int level = 0; level < nLevels; ++level)
        cout << "level " << level << " costs : " << TimerDetectPerLevel[level].aveCost() << " ± " << TimerDetectPerLevel[level].devCost() << endl;
    cout << "copy costs: " << TimerCopy.aveCost() << " ± " << TimerCopy.devCost() << endl;
    cout << "total costs: " << TimerTotal.aveCost() << " ± " << TimerTotal.devCost() << endl;
}

struct TestExtractor : public HFextractor
{
TestExtractor(int nfeatures, float threshold, int nNMSRadius,
                float scaleFactor, int nlevels, const std::vector<BaseModel*>& vpModels) :
    HFextractor(nfeatures, threshold, nNMSRadius, scaleFactor, nLevels, vpModels){}

int ExtractUsingFor(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints,
                cv::Mat &localDescriptors, cv::Mat &globalDescriptors)
{
    if(image.empty()) return -1;
    assert(image.type() == CV_8UC1 );

    TimerPyramid.Tic();
    ComputePyramid(image);
    TimerPyramid.Toc();

    int nKeypoints = 0;
    vector<vector<cv::KeyPoint>> allKeypoints(nlevels);
    vector<cv::Mat> allDescriptors(nlevels);
    for (int level = 0; level < nlevels; ++level)
    {
        TimerDetectPerLevel[level].Tic();
        if (level == 0)
        {
            mvpModels[level]->Detect(mvImagePyramid[level], allKeypoints[level], allDescriptors[level], globalDescriptors, mnFeaturesPerLevel[level], threshold, nNMSRadius);
        }
        else
        {
            mvpModels[level]->Detect(mvImagePyramid[level], allKeypoints[level], allDescriptors[level], mnFeaturesPerLevel[level], threshold, ceil(nNMSRadius*mvInvScaleFactor[level]));
        }
        TimerDetectPerLevel[level].Toc();
        nKeypoints += allKeypoints[level].size();
    }

    TimerCopy.Tic();
    vKeyPoints.clear();
    vKeyPoints.reserve(nKeypoints);
    for (int level = 0; level < nlevels; ++level)
    {
        for (auto keypoint : allKeypoints[level])
        {
            keypoint.octave = level;
            keypoint.pt *= mvScaleFactor[level];
            vKeyPoints.emplace_back(keypoint);
        }
    }
    cv::vconcat(allDescriptors.data(), allDescriptors.size(), localDescriptors);
    TimerCopy.Toc();

    return vKeyPoints.size();
}

class DetectParallel : public cv::ParallelLoopBody
{
public:

    DetectParallel (vector<cv::KeyPoint> *allKeypoints, cv::Mat *allDescriptors, cv::Mat *globalDescriptors, TestExtractor* pExtractor)
        : mAllKeypoints(allKeypoints), mAllDescriptors(allDescriptors), mGlobalDescriptors(globalDescriptors), mpExtractor(pExtractor) {}

    virtual void operator ()(const cv::Range& range) const CV_OVERRIDE
    {
        for (int level = range.start; level != range.end; ++level)
        {
            if (level == 0)
            {
                mpExtractor->mvpModels[level]->Detect(mpExtractor->mvImagePyramid[level], mAllKeypoints[level], mAllDescriptors[level], *mGlobalDescriptors, mpExtractor->mnFeaturesPerLevel[level], mpExtractor->threshold, mpExtractor->nNMSRadius);
            }
            else
            {
                mpExtractor->mvpModels[level]->Detect(mpExtractor->mvImagePyramid[level], mAllKeypoints[level], mAllDescriptors[level], mpExtractor->mnFeaturesPerLevel[level], mpExtractor->threshold, ceil(mpExtractor->nNMSRadius*mpExtractor->GetInverseScaleFactors()[level]));
            }
        }
    }

    DetectParallel& operator=(const DetectParallel &) {
        return *this;
    };
private:
    vector<cv::KeyPoint> *mAllKeypoints;
    cv::Mat *mAllDescriptors;
    cv::Mat *mGlobalDescriptors;
    TestExtractor* mpExtractor;
};

int ExtractUsingParallel(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints,
                cv::Mat &localDescriptors, cv::Mat &globalDescriptors)
{
    if(image.empty()) return -1;
    assert(image.type() == CV_8UC1 );

    TimerPyramid.Tic();
    ComputePyramid(image);
    TimerPyramid.Toc();

    int nKeypoints = 0;
    vector<vector<cv::KeyPoint>> allKeypoints(nlevels);
    vector<cv::Mat> allDescriptors(nlevels);
    
    TimerDetectPerLevel[0].Tic();
    DetectParallel detector(allKeypoints.data(), allDescriptors.data(), &globalDescriptors, this);
    cv::parallel_for_(cv::Range(0, nLevels), detector);
    TimerDetectPerLevel[0].Toc();

    for (int level = 0; level < nlevels; ++level)
        nKeypoints += allKeypoints[level].size();

    TimerCopy.Tic();
    vKeyPoints.clear();
    vKeyPoints.reserve(nKeypoints);
    for (int level = 0; level < nlevels; ++level)
    {
        for (auto keypoint : allKeypoints[level])
        {
            keypoint.octave = level;
            keypoint.pt *= mvScaleFactor[level];
            vKeyPoints.emplace_back(keypoint);
        }
    }
    cv::vconcat(allDescriptors.data(), allDescriptors.size(), localDescriptors);
    TimerCopy.Toc();

    return vKeyPoints.size();
}
};

int main(int argc, char* argv[])
{
    Eigen::setNbThreads(std::max(Eigen::nbThreads() / 2, 1));
    settings = new Settings(strSettingsPath, 0);
    vector<BaseModel*> vpModels;

    cv::Size ImSize = settings->newImSize();
    float scale = 1.0;
    for (int level = 0; level < nLevels; ++level)
    {
        cv::Vec4i inputShape{1, cvRound(ImSize.height * scale), cvRound(ImSize.width * scale), 1};
        BaseModel *pNewModel;
        if (level == 0) pNewModel = new HFNetVINOModel(strXmlPath, strBinPath, kImageToLocalAndIntermediate, inputShape);
        else pNewModel = new HFNetVINOModel(strXmlPath, strBinPath, kImageToLocal, inputShape);
        vpModels.emplace_back(pNewModel);
        scale /= scaleFactor;
    }

    TestExtractor *pExtractor = new TestExtractor(settings->nFeatures(), settings->threshold(), settings->nNMSRadius(), scaleFactor, nLevels, vpModels);

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(dbStart, files.size() - dbEnd);

    cv::Mat image;
    vector<KeyPoint> vKeyPoints;
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
        TimerTotal.Tic();
        pExtractor->ExtractUsingParallel(image, vKeyPoints, localDescriptors, globalDescriptors);
        TimerTotal.Toc();

        cout << "Get features number: " << vKeyPoints.size() << endl;
        PrintTimer();
        
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
            TimerTotal.Tic();
            pExtractor->ExtractUsingParallel(image, vKeyPoints, localDescriptors, globalDescriptors);
            TimerTotal.Toc();
        }
        cout << "Detect the full features with TestExtractor ExtractUsingParallel(): " << endl
             << "Get features number: " << vKeyPoints.size() << endl;
        PrintTimer();
    }

    {
        cout << endl;
        ClearTimer();
        for (int index = 0; index < files.size() / 10; ++index)
        {
            image = imread(strDatasetPath + files[index], IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            TimerTotal.Tic();
            pExtractor->ExtractUsingFor(image, vKeyPoints, localDescriptors, globalDescriptors);
            TimerTotal.Toc();
        }
        cout << "Detect the full features with TestExtractor ExtractUsingFor(): " << endl
             << "Get features number: " << vKeyPoints.size() << endl;
        PrintTimer();
    }

    {
        cout << endl;
        ClearTimer();

        HFextractor extractor = HFextractor(settings->nFeatures(),settings->threshold(),settings->nNMSRadius(),scaleFactor,nLevels,vpModels);
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            TimerTotal.Tic();
            extractor(image, vKeyPoints, localDescriptors, globalDescriptors);
            TimerTotal.Toc();
        }
        cout << "Detect the full features with HFextractor: " << endl
             << "Get features number: " << vKeyPoints.size() << endl;
        PrintTimer();
    }

    cout << endl << "Press 'ENTER' to exit" << endl;
    getchar();

    return 0;
}