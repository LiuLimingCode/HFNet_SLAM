/**
 * To Test the performance of different dector
 * 
Detect the full features with TestExtractor: 
Get features number: 767
pyramid costs: 0.536115
level 0 costs : 8.26348
level 1 costs : 4.03005
level 2 costs : 2.88576
level 3 costs : 2.24501
level 4 costs : 1.76847
level 5 costs : 1.71524
level 6 costs : 1.69197
level 7 costs : 1.62562
copy costs: 0.0957283
total costs: 24.8614
 */
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>

#include "Settings.h"
#include "Extractors/HFNetTFModelV2.h"
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

const std::string strModelPath("/home/llm/ROS/HFNet_ORBSLAM3_v2/model/hfnet_tf_v2_NMS2/");
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
    cout << "pyramid costs: " << TimerPyramid.aveCost() << endl;
    for (int level = 0; level < nLevels; ++level) cout << "level " << level << " costs : " << TimerDetectPerLevel[level].aveCost() << endl;
    cout << "copy costs: " << TimerCopy.aveCost() << endl;
    cout << "total costs: " << TimerTotal.aveCost() << endl;
}

struct TestExtractor : public HFextractor
{
TestExtractor(int nfeatures, int nNMSRadius, float threshold,
                float scaleFactor, int nlevels, const std::vector<BaseModel*>& vpModels) :
    HFextractor(nfeatures, nNMSRadius, threshold, scaleFactor, nLevels, vpModels){}

int test(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints,
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
            mvpModels[level]->DetectOnlyLocal(mvImagePyramid[level], allKeypoints[level], allDescriptors[level], mnFeaturesPerLevel[level], threshold, ceil(nNMSRadius*mvInvScaleFactor[level]));
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
                mpExtractor->mvpModels[level]->DetectOnlyLocal(mpExtractor->mvImagePyramid[level], mAllKeypoints[level], mAllDescriptors[level], mpExtractor->mnFeaturesPerLevel[level], mpExtractor->threshold, ceil(mpExtractor->nNMSRadius*mpExtractor->GetInverseScaleFactors()[level]));
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

int test_v2(const cv::Mat &image, std::vector<cv::KeyPoint>& vKeyPoints,
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
        BaseModel *pNewModel = new HFNetTFModelV2(strModelPath);
        if (level == 0) pNewModel->Compile(inputShape, false);
        else pNewModel->Compile(inputShape, true);
        vpModels.emplace_back(pNewModel);
        scale /= scaleFactor;
    }

    TestExtractor *pExtractor = new TestExtractor(settings->nFeatures(), settings->nNMSRadius(), settings->threshold(), scaleFactor, nLevels, vpModels);

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
        pExtractor->test_v2(image, vKeyPoints, localDescriptors, globalDescriptors);
        TimerTotal.Toc();

        cout << "Get features number: " << vKeyPoints.size() << endl;
        PrintTimer();
        
        ShowKeypoints("press 'q' to exit", image, vKeyPoints);
        cout << endl;
        command = cv::waitKey();
    }
    cv::destroyAllWindows();

    // detect full dataset
    {
        image = imread(strDatasetPath + files[0], IMREAD_GRAYSCALE);
        if (settings->needToResize())
            cv::resize(image, image, settings->newImSize());
        pExtractor->test_v2(image, vKeyPoints, localDescriptors, globalDescriptors);
        
        ClearTimer();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            TimerTotal.Tic();
            pExtractor->test_v2(image, vKeyPoints, localDescriptors, globalDescriptors);
            TimerTotal.Toc();
        }
        cout << "Detect the full features with TestExtractor: " << endl
             << "Get features number: " << vKeyPoints.size() << endl;
        PrintTimer();
    }
    {
        HFextractor extractor = HFextractor(settings->nFeatures(),settings->nNMSRadius(),settings->threshold(),scaleFactor,nLevels,vpModels);
        image = imread(strDatasetPath + files[0], IMREAD_GRAYSCALE);
        if (settings->needToResize())
            cv::resize(image, image, settings->newImSize());
        extractor(image, vKeyPoints, localDescriptors, globalDescriptors);

        ClearTimer();
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
             << "Get features number: " << vKeyPoints.size() << endl
             << "total costs: " << TimerTotal.aveCost() << " milliseconds" << endl;
    }

    getchar();

    return 0;
}