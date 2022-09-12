#include <iostream>
#include <chrono>
#include <unordered_set>
#include "Eigen/Core"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "../include/Frame.h"
#include "../include/Settings.h"
#include "../include/Extractors/HFextractor.h"
#include "../include/Extractors/HFNetTFModelV2.h"

#include "ORBextractor.h"
#include "ORBVocabulary.h"

#include "app/utility_common.h"

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace ORB_SLAM3;

Settings *settings;

struct KeyFrameHFNetSLAM
{
int mnFrameId;
const cv::Mat imgLeft;
cv::Mat mGlobalDescriptors;

float mPlaceRecognitionScore = 1.0;

KeyFrameHFNetSLAM(int id, const cv::Mat im, HFNetTFModelV2* pModelInter, HFNetTFModelV2* pModelGlobal) :
    mnFrameId(id), imgLeft(im) {
        vector<cv::KeyPoint> vKeyPoints;
        cv::Mat localDescriptors, intermediate;
        pModelInter->Detect(im, vKeyPoints, localDescriptors, intermediate, 1000, 0.01, 4);
        pModelGlobal->Detect(intermediate, mGlobalDescriptors);
    }
};

typedef vector<KeyFrameHFNetSLAM*> KeyFrameDBHFNetSLAM;

struct KeyFrameORBSLAM3
{
int mnFrameId;
const cv::Mat imgLeft;

std::vector<cv::KeyPoint> mvKeys;
cv::Mat mDescriptors;
DBoW2::BowVector mBowVec;
DBoW2::FeatureVector mFeatVec;

float mPlaceRecognitionScore = 1.0;

// static std::vector<list<KeyFrameORBSLAM3*> > mvInvertedFile;

KeyFrameORBSLAM3(int id, const cv::Mat im, ORBVocabulary* mpORBvocabulary, ORBextractor* extractorLeft) :
    mnFrameId(id), imgLeft(im) {
        vector<int> vLapping = {0,1000};
        (*extractorLeft)(imgLeft, cv::Mat(), mvKeys, mDescriptors, vLapping);
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);

        // if (mvInvertedFile.empty()) mvInvertedFile.resize(mpORBvocabulary->size());
        // for(DBoW2::BowVector::const_iterator vit= mBowVec.begin(), vend=mBowVec.end(); vit!=vend; vit++)
        //     mvInvertedFile[vit->first].push_back(this);
    }
};
typedef vector<KeyFrameORBSLAM3*> KeyFrameDBORBSLAM3;

vector<KeyFrameORBSLAM3*> GetNCandidateLoopFrameORBSLAM3(ORBVocabulary* mpVoc, KeyFrameORBSLAM3* query, const KeyFrameDBORBSLAM3 &db, int k)
{
    if (db.front()->mnFrameId > query->mnFrameId - 30) return vector<KeyFrameORBSLAM3*>();

    int count = 0;
    for (auto it = db.begin(); it != db.end(); ++it)
    {
        KeyFrameORBSLAM3 *pKF = *it;
        if (pKF->mnFrameId > query->mnFrameId - 30) break;
        count++;
        pKF->mPlaceRecognitionScore = mpVoc->score(pKF->mBowVec,query->mBowVec);
    }
    vector<KeyFrameORBSLAM3*> res(k);
    std::partial_sort_copy(db.begin(), db.begin() + count, res.begin(), res.end(), [](KeyFrameORBSLAM3* const f1, KeyFrameORBSLAM3* const f2) {
        return f1->mPlaceRecognitionScore > f2->mPlaceRecognitionScore;
    });
    return res;
}

vector<KeyFrameHFNetSLAM*> GetNCandidateLoopFrameHFNetSLAM(KeyFrameHFNetSLAM* query, const KeyFrameDBHFNetSLAM &db, int k)
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
const int dbEnd = 0;

int main(int argc, char** argv)
{
    settings = new Settings(strSettingsPath, 0);
    HFNetTFModelV2 *pModelImageToLocalAndInter = new HFNetTFModelV2(settings->strTFModelPath(), kImageToLocalAndIntermediate, {1, settings->newImSize().height, settings->newImSize().width, 1});
    HFNetTFModelV2 *pModelInterToGlobal = new HFNetTFModelV2(settings->strTFModelPath(), kIntermediateToGlobal, {1, settings->newImSize().height/8, settings->newImSize().width/8, 96});

    ORBextractor extractorORB(1000, 1.2, 8, 20, 7);

    const string strVocFileORB("/home/llm/ROS/HFNet_ORBSLAM3_v2/Comparison/Vocabulary/ORBvoc.txt");
    ORBVocabulary vocabORB;
    if(!vocabORB.loadFromTextFile(strVocFileORB))
    {
        cerr << "Falied to open at: " << strVocFileORB << endl;
        exit(-1);
    }

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
    int nKeyFrame = (end - dbStart) / step;

    if (nKeyFrame <= 30) exit(-1);
    cout << "Dataset range: [" << dbStart << " ~ " << end << "]" << ", nKeyFrame: " << nKeyFrame << endl;

    KeyFrameDBHFNetSLAM vKeyFrameDBHFNetSLAM;
    vKeyFrameDBHFNetSLAM.reserve(nKeyFrame);
    KeyFrameDBORBSLAM3 vKeyFrameDBORBSLAM3;
    vKeyFrameDBORBSLAM3.reserve(nKeyFrame);
    vector<cv::Mat> vImageDatabase;
    float cur = dbStart;
    while (cur < end)
    {
        int select = cur;
        cv::Mat image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        if (settings->needToResize())
            cv::resize(image, image, settings->newImSize());

        // KeyFrameHFNetSLAM *pKFHF = new KeyFrameHFNetSLAM(select, image, pModelImageToLocalAndInter, pModelInterToGlobal);
        // vKeyFrameDBHFNetSLAM.emplace_back(pKFHF);

        KeyFrameORBSLAM3 *pKFORB = new KeyFrameORBSLAM3(select, image, &vocabORB, &extractorORB);
        vKeyFrameDBORBSLAM3.emplace_back(pKFORB);

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
    int select = 18599;
    while (1)
    {
        if (command == 'w') select += 1;
        else if (command == 'x') select -= 1;
        else if (command == '1') select = 8317;
        else if (command == '2') select = 17885;
        else if (command == '3') select = 18599;
        // else select = distribution(generator);

        cv::Mat image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        if (settings->needToResize())
            cv::resize(image, image, settings->newImSize());

        KeyFrameHFNetSLAM *pKFHF = new KeyFrameHFNetSLAM(select, image, pModelImageToLocalAndInter, pModelInterToGlobal);

        KeyFrameORBSLAM3 *pKFORB = new KeyFrameORBSLAM3(select, image, &vocabORB, &extractorORB);

        auto t1 = chrono::steady_clock::now();
        // auto res = GetNCandidateLoopFrameHFNetSLAM(pKF, vKeyFrameDBHFNetSLAM, 3);
        auto res = GetNCandidateLoopFrameORBSLAM3(&vocabORB, pKFORB, vKeyFrameDBORBSLAM3, 3);
        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "Query cost time: " << t << endl;

        ShowImageWithText("Query Image", image, std::to_string((int)pKFORB->mnFrameId));
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