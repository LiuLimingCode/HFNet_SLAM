/**
 *
GetNCandidateLoopFrameCV():
Query cost time: 1339
Query cost time: 1328
GetNCandidateLoopFrameEigen():
Query cost time: 245
Query cost time: 259
 * Eigen is much faster than OpenCV
 */
#include <iostream>
#include <chrono>
#include <unordered_set>
#include "Eigen/Core"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "Frame.h"
#include "Settings.h"
#include "Extractors/HFextractor.h"
#include "Examples/Utility/utility_common.h"
#include "CameraModels/Pinhole.h"

#include "utility_common.h"

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace ORB_SLAM3;

Settings *settings;

struct TestKeyFrame
{
double mTimeStamp;
const cv::Mat imgLeft;
const cv::Mat mGlobalDescriptors;
float mPlaceRecognitionScore;

TestKeyFrame(Frame &F) :
mTimeStamp(F.mTimeStamp), imgLeft(F.imgLeft), mGlobalDescriptors(F.mGlobalDescriptors) {}
};

typedef unordered_set<TestKeyFrame*> KeyFrameDB;

vector<TestKeyFrame*> GetNCandidateLoopFrameCV(Frame* query, const KeyFrameDB &db, int k)
{
    //vector<Frame*> res = db;
    vector<TestKeyFrame*> res(k);
    for (auto it = db.begin(); it != db.end(); ++it)
    {
        TestKeyFrame *pKF = *it;
        pKF->mPlaceRecognitionScore = cv::norm(query->mGlobalDescriptors - pKF->mGlobalDescriptors, cv::NORM_L2);
    }
    //std::nth_element(res.begin(), res.end(), res.begin() + k, )
    std::partial_sort_copy(db.begin(), db.end(), res.begin(), res.end(), [](TestKeyFrame* const f1, TestKeyFrame* const f2) {
        return f1->mPlaceRecognitionScore < f2->mPlaceRecognitionScore;
    });
    return res;
}

vector<TestKeyFrame*> GetNCandidateLoopFrameEigen(Frame* query, const KeyFrameDB &db, int k)
{
    vector<TestKeyFrame*> res(k);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> queryDescriptors(query->mGlobalDescriptors.ptr<float>(), query->mGlobalDescriptors.rows, query->mGlobalDescriptors.cols);
    for (auto it = db.begin(); it != db.end(); ++it)
    {
        TestKeyFrame *pKF = *it;
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> pKFDescriptors(pKF->mGlobalDescriptors.ptr<float>(), pKF->mGlobalDescriptors.rows, pKF->mGlobalDescriptors.cols);
        pKF->mPlaceRecognitionScore = (queryDescriptors - pKFDescriptors).norm();
    }
    std::partial_sort_copy(db.begin(), db.end(), res.begin(), res.end(), [](TestKeyFrame* const f1, TestKeyFrame* const f2) {
        return f1->mPlaceRecognitionScore < f2->mPlaceRecognitionScore;
    });
    return res;
}

void ShowImageWithText(const string &title, const cv::Mat &image, const string &str)
{
    cv::Mat plot;
    cv::cvtColor(image, plot, cv::COLOR_GRAY2RGB);
    cv::putText(plot, str, cv::Point2d(0, 30),cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
    cv::imshow(title, plot);
}

// const string strDatasetPath("/media/llm/Datasets/EuRoC/MH_01_easy/mav0/cam0/data/");
// const string strSettingsPath("Examples/Monocular-Inertial/EuRoC.yaml");
// const int dbStart = 420;
// const int dbEnd = 50;

const string strDatasetPath("/media/llm/Datasets/TUM-VI/dataset-corridor4_512_16/mav0/cam0/data/");
const string strSettingsPath("Examples/Monocular-Inertial/TUM-VI.yaml");
const int dbStart = 50;
const int dbEnd = 50;

const int nKeyFrame = 200;

int main(int argc, char** argv)
{
    settings = new Settings(strSettingsPath, 0);
    HFNetTFModel *pModel = new HFNetTFModel(settings->strResamplerPath(), settings->strModelPath());
    HFextractor *pExtractor = new HFextractor(settings->nFeatures(),settings->nNMSRadius(),settings->threshold(),1.0,1,{pModel});
    GeometricCamera* pCamera = settings->camera1();
    cv::Mat distCoef = settings->camera1DistortionCoef();
    const float bf = 0;
    const float thDepth = 0;

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    cout << "Got [" << files.size() << "] images in dataset" << endl;

    int end = files.size() - dbEnd;
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(dbStart, end);

    const float step = (end - dbStart) / (float)nKeyFrame;
    if (end <= dbStart + nKeyFrame) exit(-1);
    cout << "Dataset range: [" << dbStart << " ~ " << end << "]" << ", nKeyFrame: " << nKeyFrame << endl;

    KeyFrameDB vKeyFrameDB;
    vector<cv::Mat> vImageDatabase;
    float cur = dbStart;
    while (cur < end)
    {
        int select = cur;
        cv::Mat imageGray = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        Frame F = Frame(imageGray,select,pExtractor,pCamera,distCoef,bf,thDepth);
        F.imgLeft = imageGray;
        TestKeyFrame *pKF = new TestKeyFrame(F);
        vKeyFrameDB.insert(pKF);
        cur += step;
    }

    cv::namedWindow("Query Image");
    cv::moveWindow("Query Image", 0 ,0);
    cv::namedWindow("Candidate 1");
    cv::moveWindow("Candidate 1", 820, 0);
    cv::namedWindow("Candidate 2");
    cv::moveWindow("Candidate 2", 0, 540);
    cv::namedWindow("Candidate 3");
    cv::moveWindow("Candidate 3", 820, 540);

    char command = ' ';
    int select = dbStart;
    int plot = 0;
    while (1)
    {
        if (command == 'q') break;
        else if (command == 'w') select += 1, plot = 0;
        else if (command == 's') select -= 1, plot = 0;
        else if (command == 'd') plot += 1;
        else if (command == 'a') plot = max(plot - 1, 0);
        else select = distribution(generator), plot = 0;

        cv::Mat imageGray = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        Frame *pKF = new Frame(imageGray,select,pExtractor,pCamera,distCoef,bf,thDepth);

        auto t1 = chrono::steady_clock::now();
        auto res = GetNCandidateLoopFrameEigen(pKF, vKeyFrameDB, vKeyFrameDB.size());
        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "Query cost time: " << t << endl;

        ShowImageWithText("Query Image", imageGray, std::to_string((int)pKF->mTimeStamp));
        for (int index = 0; index < 3; ++index)
        {
            int plotIndex = plot * 3 + index;
            ShowImageWithText("Candidate " + std::to_string(index + 1), res[plotIndex]->imgLeft,
                std::to_string((int)res[plotIndex]->mTimeStamp) + ":" + std::to_string(res[plotIndex]->mPlaceRecognitionScore));
        }

        command = cv::waitKey();
    }

    system("pause");
}