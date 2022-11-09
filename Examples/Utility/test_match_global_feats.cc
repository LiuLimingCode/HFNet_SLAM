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
#include "Extractors/HFNetTFModelV2.h"
#include "utility_common.h"
#include "CameraModels/Pinhole.h"

#include "utility_common.h"

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace ORB_SLAM3;

Settings *settings;

struct KeyFrameHFNetSLAM
{
int mnFrameId;
const cv::Mat imgLeft;
const cv::Mat mGlobalDescriptors;
float mPlaceRecognitionScore = 1.0;

KeyFrameHFNetSLAM(int id, const cv::Mat im, const cv::Mat globalDescriptors) :
mnFrameId(id), imgLeft(im), mGlobalDescriptors(globalDescriptors) {}
};

typedef vector<KeyFrameHFNetSLAM*> KeyFrameDB;

vector<KeyFrameHFNetSLAM*> GetNCandidateLoopFrameCV(KeyFrameHFNetSLAM* query, const KeyFrameDB &db, int k)
{
    //vector<Frame*> res = db;
    vector<KeyFrameHFNetSLAM*> res(k);
    for (auto it = db.begin(); it != db.end(); ++it)
    {
        KeyFrameHFNetSLAM *pKF = *it;
        pKF->mPlaceRecognitionScore = cv::norm(query->mGlobalDescriptors - pKF->mGlobalDescriptors, cv::NORM_L2);
    }
    //std::nth_element(res.begin(), res.end(), res.begin() + k, )
    std::partial_sort_copy(db.begin(), db.end(), res.begin(), res.end(), [](KeyFrameHFNetSLAM* const f1, KeyFrameHFNetSLAM* const f2) {
        return f1->mPlaceRecognitionScore < f2->mPlaceRecognitionScore;
    });
    return res;
}

vector<KeyFrameHFNetSLAM*> GetNCandidateLoopFrameEigen(KeyFrameHFNetSLAM* query, const KeyFrameDB &db, int k)
{
    if (db.front()->mnFrameId > query->mnFrameId - 30) return vector<KeyFrameHFNetSLAM*>();

    int count = 0;
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> queryDescriptors(query->mGlobalDescriptors.ptr<float>(), query->mGlobalDescriptors.rows, query->mGlobalDescriptors.cols);
    for (auto it = db.begin(); it != db.end(); ++it)
    {
        KeyFrameHFNetSLAM *pKF = *it;
        if (pKF->mnFrameId > query->mnFrameId - 30) break;
        count++;
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> pKFDescriptors(pKF->mGlobalDescriptors.ptr<float>(), pKF->mGlobalDescriptors.rows, pKF->mGlobalDescriptors.cols);
        pKF->mPlaceRecognitionScore = (queryDescriptors - pKFDescriptors).norm();
    }
    vector<KeyFrameHFNetSLAM*> res(k);
    std::partial_sort_copy(db.begin(), db.begin() + count, res.begin(), res.end(), [](KeyFrameHFNetSLAM* const f1, KeyFrameHFNetSLAM* const f2) {
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

const string strDatasetPath("/media/llm/Datasets/TUM-VI/dataset-outdoors7_512_16/mav0/cam0/data/");
const string strSettingsPath("Examples/Monocular-Inertial/TUM-VI.yaml");
const int dbStart = 0;
const int dbEnd = 12000;

int main(int argc, char** argv)
{
    settings = new Settings(strSettingsPath, 0);
    HFNetTFModelV2 *pModelImageToLocalAndInter = new HFNetTFModelV2(settings->strTFModelPath(), kImageToLocalAndIntermediate, {1, settings->newImSize().height, settings->newImSize().width, 1});
    HFNetTFModelV2 *pModelInterToGlobal = new HFNetTFModelV2(settings->strTFModelPath(), kIntermediateToGlobal, {1, settings->newImSize().height/8, settings->newImSize().width/8, 96});
    GeometricCamera* pCamera = settings->camera1();
    cv::Mat distCoef;
    if(settings->needToUndistort()){
        distCoef = settings->camera1DistortionCoef();
    }
    else{
        distCoef = cv::Mat::zeros(4,1,CV_32F);
    }

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    cout << "Got [" << files.size() << "] images in dataset" << endl;

    int end = files.size() - dbEnd;
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(dbStart, end);

    const int step = 4;
    int nKeyFrame = (dbStart - end) / step;

    if (nKeyFrame <= 30) exit(-1);
    cout << "Dataset range: [" << dbStart << " ~ " << end << "]" << ", nKeyFrame: " << nKeyFrame << endl;

    KeyFrameDB vKeyFrameDB;
    vKeyFrameDB.reserve(nKeyFrame);
    vector<cv::Mat> vImageDatabase;
    float cur = dbStart;
    while (cur < end)
    {
        int select = cur;
        cv::Mat image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        if (settings->needToResize())
            cv::resize(image, image, settings->newImSize());

        vector<cv::KeyPoint> vKeyPoints;
        cv::Mat localDescriptors, globalDescriptors, intermediate;
        pModelImageToLocalAndInter->Detect(image, vKeyPoints, localDescriptors, intermediate, 1000, 0.01);
        pModelInterToGlobal->Detect(intermediate, globalDescriptors);

        KeyFrameHFNetSLAM *pKF = new KeyFrameHFNetSLAM(select, image.clone(), globalDescriptors.clone());
        vKeyFrameDB.emplace_back(pKF);
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
    int select = 0;
    while (1)
    {
        if (command == 'q') break;
        else if (command == 'w') select += 1;
        else if (command == 's') select -= 1;
        else select = distribution(generator);

        cv::Mat image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        if (settings->needToResize())
            cv::resize(image, image, settings->newImSize());

        vector<cv::KeyPoint> vKeyPoints;
        cv::Mat localDescriptors, globalDescriptors, intermediate;
        pModelImageToLocalAndInter->Detect(image, vKeyPoints, localDescriptors, intermediate, 1000, 0.01);
        pModelInterToGlobal->Detect(intermediate, globalDescriptors);

        KeyFrameHFNetSLAM *pKF = new KeyFrameHFNetSLAM(select, image.clone(), globalDescriptors.clone());

        auto t1 = chrono::steady_clock::now();
        auto res = GetNCandidateLoopFrameEigen(pKF, vKeyFrameDB, 3);
        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "Query cost time: " << t << endl;

        ShowImageWithText("Query Image", image, std::to_string((int)pKF->mnFrameId));
        for (int index = 0; index < 3; ++index)
        {
            if (index < res.size())
                ShowImageWithText("Candidate " + std::to_string(index + 1), res[index]->imgLeft,
                    std::to_string((int)res[index]->mnFrameId) + ":" + std::to_string(res[index]->mPlaceRecognitionScore));
            else {
                Mat empty;
                empty.create(cv::Size(100,100), CV_8UC1);
                cv::imshow("Candidate " + std::to_string(index + 1), empty);
            }
        }

        command = cv::waitKey();
    }

    system("pause");
}