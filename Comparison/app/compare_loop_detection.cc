#include <iostream>
#include <chrono>
#include <unordered_set>
#include "Eigen/Core"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "../include/Frame.h"
#include "../include/Extractors/HFextractor.h"

#include "ORBextractor.h"
#include "ORBVocabulary.h"

#include "../include/utility_common.h"

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace ORB_SLAM3;

struct TestKeyFrame
{
    int mnFrameId;
    float mPlaceRecognitionScore = 1.0;
};

struct KeyFrameHFNetSLAM : public TestKeyFrame
{
    cv::Mat mGlobalDescriptors;

    KeyFrameHFNetSLAM(int id, const cv::Mat im, BaseModel* pModel) {
        mnFrameId = id;
        vector<cv::KeyPoint> vKeyPoints;
        cv::Mat localDescriptors, intermediate;
        pModel->Detect(im, vKeyPoints, localDescriptors, mGlobalDescriptors, 1000, 0.01);
    }
};

struct KeyFrameORBSLAM3 : public TestKeyFrame
{
    std::vector<cv::KeyPoint> mvKeys;
    cv::Mat mDescriptors;
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    KeyFrameORBSLAM3(int id, const cv::Mat im, ORBVocabulary* mpORBvocabulary, ORBextractor* extractorLeft) {
        mnFrameId = id;
        vector<int> vLapping = {0,1000};
        (*extractorLeft)(im, cv::Mat(), mvKeys, mDescriptors, vLapping);
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
};

typedef vector<TestKeyFrame*> KeyFrameDB;

KeyFrameDB GetNCandidateLoopFrameORBSLAM3(ORBVocabulary* mpVoc, KeyFrameORBSLAM3* query, const KeyFrameDB &db, int k)
{
    if (db.front()->mnFrameId >= query->mnFrameId - 30) return KeyFrameDB();

    int count = 0;
    for (auto it = db.begin(); it != db.end(); ++it)
    {
        KeyFrameORBSLAM3 *pKF = static_cast<KeyFrameORBSLAM3*>(*it);
        if (pKF->mnFrameId >= query->mnFrameId - 30) break;
        count++;
        pKF->mPlaceRecognitionScore = mpVoc->score(pKF->mBowVec,query->mBowVec);
    }
    KeyFrameDB res(min(k, count));
    std::partial_sort_copy(db.begin(), db.begin() + count, res.begin(), res.end(), [](TestKeyFrame* const f1, TestKeyFrame* const f2) {
        return f1->mPlaceRecognitionScore > f2->mPlaceRecognitionScore;
    });
    return res;
}

KeyFrameDB GetNCandidateLoopFrameHFNetSLAM(KeyFrameHFNetSLAM* query, const KeyFrameDB &db, int k)
{
    if (db.front()->mnFrameId >= query->mnFrameId - 30) return KeyFrameDB();

    int count = 0;
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> queryDescriptors(query->mGlobalDescriptors.ptr<float>(), query->mGlobalDescriptors.rows, query->mGlobalDescriptors.cols);
    for (auto it = db.begin(); it != db.end(); ++it)
    {
        KeyFrameHFNetSLAM *pKF = static_cast<KeyFrameHFNetSLAM*>(*it);
        if (pKF->mnFrameId >= query->mnFrameId - 30) break;
        count++;
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> pKFDescriptors(pKF->mGlobalDescriptors.ptr<float>(), pKF->mGlobalDescriptors.rows, pKF->mGlobalDescriptors.cols);
        pKF->mPlaceRecognitionScore = 1 - (queryDescriptors - pKFDescriptors).norm();
    }
    KeyFrameDB res(min(k, count));
    std::partial_sort_copy(db.begin(), db.begin() + count, res.begin(), res.end(), [](TestKeyFrame* const f1, TestKeyFrame* const f2) {
        return f1->mPlaceRecognitionScore > f2->mPlaceRecognitionScore;
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

int main(int argc, char** argv)
{
    if (argc != 4) {
        cerr << endl << "Usage: compare_loop_detection path_to_dataset path_to_model path_to_vocabulary" << endl;   
        return -1;
    }
    const string strDatasetPath = string(argv[1]);
    const string strModelPath = string(argv[2]);
    const string strVocFileORB = string(argv[3]);

    // By default, the Eigen will use the maximum number of threads in OpenMP.
    // However, this will somehow slow down the calculation of dense matrix multiplication.
    // Therefore, use only half of the thresds.
    Eigen::setNbThreads(std::max(Eigen::nbThreads() / 2, 1));

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    if (files.empty()) {
        std::cout << "Error, failed to find any valid image in: " << strDatasetPath << std::endl;
        return 1;
    }
    cv::Size ImSize = imread(strDatasetPath + files[0], IMREAD_GRAYSCALE).size();
    if (ImSize.area() == 0) {
        std::cout << "Error, failed to read the image at: " << strDatasetPath + files[0] << std::endl;
        return 1;
    }
    std::cout << "Got [" << files.size() << "] images in dataset" << std::endl;

    BaseModel *pModel = InitRTModel(strModelPath, kImageToLocalAndGlobal, {1, ImSize.height, ImSize.width, 1});
    // BaseModel *pModel = InitTFModel(strModelPath, kImageToLocalAndGlobal, {1, ImSize.height, ImSize.width, 1});

    ORBextractor extractorORB(1000, 1.2, 8, 20, 7);

    ORBVocabulary vocabORB;
    if(!vocabORB.loadFromTextFile(strVocFileORB))
    {
        cerr << "Falied to open at: " << strVocFileORB << std::endl;
        exit(-1);
    }

    int start = 0;
    int end = files.size();

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(30, end);

    const int step = 4;
    int nKeyFrame = (end - start) / step;

    if (nKeyFrame <= 30) exit(-1);
    std::cout << "Dataset range: [" << start << " ~ " << end << "]" << ", nKeyFrame: " << nKeyFrame << std::endl;

    KeyFrameDB vKeyFrameDBHFNetSLAM, vKeyFrameDBORBSLAM3;
    vKeyFrameDBHFNetSLAM.reserve(nKeyFrame);
    vKeyFrameDBORBSLAM3.reserve(nKeyFrame);
    float cur = start;
    while (cur < end)
    {
        int select = cur;
        cv::Mat image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);

        KeyFrameHFNetSLAM *pKFHF = new KeyFrameHFNetSLAM(select, image, pModel);
        vKeyFrameDBHFNetSLAM.emplace_back(pKFHF);

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

    char command = 0;
    bool showHF = true;
    int select = distribution(generator);
    while (1)
    {
        if (command == 'w') select += 1;
        else if (command == 'x') select -= 1;
        else if (command == 'q') showHF = !showHF;
        else if (command == ' ') select = distribution(generator);

        cv::Mat image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);

        KeyFrameHFNetSLAM *pKFHF = new KeyFrameHFNetSLAM(select, image, pModel);

        KeyFrameORBSLAM3 *pKFORB = new KeyFrameORBSLAM3(select, image, &vocabORB, &extractorORB);

        auto t1 = chrono::steady_clock::now();
        vector<TestKeyFrame*> res;
        if (showHF)
            res = GetNCandidateLoopFrameHFNetSLAM(pKFHF, vKeyFrameDBHFNetSLAM, 3);
        else
            res = GetNCandidateLoopFrameORBSLAM3(&vocabORB, pKFORB, vKeyFrameDBORBSLAM3, 3);
        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        if (showHF) std::cout << "HFNet-SLAM: " << std::endl;
        else std::cout << "ORB-SLAM3: " << std::endl;
        std::cout << "Query cost time: " << t << std::endl;
        
        ShowImageWithText("Query Image", image, std::to_string((int)pKFORB->mnFrameId));
        for (size_t index = 0; index < 3; ++index)
        {
            if (index < res.size()) {
                cv::Mat image = imread(strDatasetPath + files[res[index]->mnFrameId], IMREAD_GRAYSCALE);
                ShowImageWithText("Candidate " + std::to_string(index + 1), image,
                    std::to_string((int)res[index]->mnFrameId) + ":" + std::to_string(res[index]->mPlaceRecognitionScore));
            }
            else {
                Mat empty = cv::Mat::zeros(ImSize, CV_8U);
                cv::imshow("Candidate " + std::to_string(index + 1), empty);
            }
        }

        command = cv::waitKey();
    }

    system("pause");
}