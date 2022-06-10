/**
 * A test for matching
 * 
 * 
 * Result:
ORB + BFMatcher:
match costs time: 5ms
matches total number: 490
correct matches number: 331
match correct percentage: 0.67551
ORB + SearchByBoW:
vocab costs time: 7ms
match costs time: 0ms
matches total number: 304
correct matches number: 216
match correct percentage: 0.710526
HF + BFMatcher_L1:
match costs time: 28ms
matches total number: 831
correct matches number: 614
match correct percentage: 0.738869
HF + BFMatcher_L2:
match costs time: 15ms
matches total number: 832
correct matches number: 637
match correct percentage: 0.765625
HF + SearchByBoW_L1:
vocab costs time: 45ms
match costs time: 4ms
matches total number: 769
correct matches number: 342
match correct percentage: 0.444733
HF + SearchByBoW_L2:
vocab costs time: 45ms
match costs time: 3ms
matches total number: 693
correct matches number: 342
match correct percentage: 0.493506
HF + SearchByBoWV2:
match costs time: 75ms
matches total number: 934
correct matches number: 342
match correct percentage: 0.366167
 * 1. HFNet is way better than ORB, but it is more time-consuming
 * 2. The L1 and L2 descriptor distance is the same for HFNet, but L2 norm is more effective
 * 3. SearchByBoW will increase the matching time
 * 4. SearchByBoW can increase the correct percentage of ORB descriptor
 * 5. SearchByBoW does not work well for HF descriptor, maybe it is because the vocabulary for HF is bad.
 * 6. The vocabulary costs too much time!
 * 7. TODO: 加了去畸变后效果变差了，为什么？
 */
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>

#include "Frame.h"
#include "Settings.h"
#include "ORBmatcher.h"
#include "Extractors/ORBextractor.h"
#include "Extractors/HFextractor.h"
#include "Examples/Utility/utility_common.h"
#include "CameraModels/Pinhole.h"

#include "fbow.h"
#include "ORBVocabulary.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;

Settings *settings;

std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

const int HISTO_LENGTH = 30;

float DescriptorDistanceHamming(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return (float)dist;
}

float DescriptorDistanceL1(const cv::Mat &a, const cv::Mat &b)
{
    assert(a.cols == b.cols);
    const float *pa = a.ptr<float>();
    const float *pb = b.ptr<float>();

    float dist=0;
    for (int i = 0 ; i < a.cols  ; i++){
        dist += fabs(pa[i] - pb[i]);
    }

    return dist;
}

// L1 ≈ sqrt(L1Plus) * 12.7
float DescriptorDistanceL1Plus(const cv::Mat &a, const cv::Mat &b)
{
    assert(a.cols == b.cols);
    // 用Eigen试试?
    
    Mat dist = 2 * (1 - a * b.t());
    return dist.at<float>(0,0);
}

float DescriptorDistanceL2(const cv::Mat &a, const cv::Mat &b)
{
    assert(a.cols == b.cols);

    return cv::norm(a - b, cv::NORM_L2);
}

// float DescriptorDistanceL1(const cv::Mat &a, const cv::Mat &b)
// {
//     assert(a.cols == b.cols);

//     return 2 * (1 - a * b.t());
// }

void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}

typedef std::map<uint32_t,std::vector<uint32_t>> FeatureVector;
// This function is original from ORB-SLAM3
int SearchByBoW(int normType, float mfNNratio, bool mbCheckOrientation, int threshold,
                std::vector<cv::KeyPoint>& keypoints1, cv::Mat& descriptors1, FeatureVector& featVec1, 
                std::vector<cv::KeyPoint>& keypoints2, cv::Mat& descriptors2, FeatureVector& featVec2,
                std::vector<cv::DMatch>& vMatches)
{
    std::vector<cv::DMatch> matches(keypoints1.size(), cv::DMatch(-1,-1,-1));

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    
    int nmatches = 0;

    FeatureVector::const_iterator f1it = featVec1.begin();
    FeatureVector::const_iterator f2it = featVec2.begin();
    FeatureVector::const_iterator f1end = featVec1.end();
    FeatureVector::const_iterator f2end = featVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                const cv::Mat &d1= descriptors1.row(idx1);

                float bestDist1 = std::numeric_limits<float>::max();
                int bestIdx2 = -1;
                float bestDist2 = std::numeric_limits<float>::max();

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    const cv::Mat &d2 = descriptors2.row(idx2);

                    float dist = std::numeric_limits<float>::max();
                    if (normType == cv::NORM_HAMMING)
                    {
                        dist = DescriptorDistanceHamming(d1, d2);
                    }
                    else if (normType == cv::NORM_L1)
                    {
                        dist = DescriptorDistanceL1Plus(d1, d2);
                    }
                    else if (normType == cv::NORM_L2)
                    {
                        dist = DescriptorDistanceL2(d1, d2);
                    }

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<threshold) //TODO: How to decide this threshold
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        DMatch match;
                        match.queryIdx = idx1;
                        match.trainIdx = bestIdx2;
                        match.distance = bestDist1;
                        match.imgIdx = 0;
                        matches[idx1] = match;

                        if(mbCheckOrientation)
                        {
                            const float factor = 1.0f/HISTO_LENGTH;
                            float rot = keypoints1[idx1].angle-keypoints2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }
            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = featVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = featVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                matches[rotHist[i][j]]=cv::DMatch(-1,-1,-1);
                nmatches--;
            }
        }
    }

    vMatches.clear();
    for (auto m : matches)
    {
        if (m.trainIdx != -1)
            vMatches.emplace_back(m);
    }

    return nmatches;
}

int SearchByBoWV2(float mfNNratio, int threshold,
                      cv::Mat& descriptors1, cv::Mat& descriptors2,
                      std::vector<cv::DMatch>& vMatches)
{
    vMatches.clear();
    vMatches.reserve(descriptors1.rows);

    cv::Mat distance = 2 * (1 - descriptors1 * descriptors2.t());

    for(int idx1=0; idx1 < distance.rows; idx1++)
    {

        float bestDist1 = std::numeric_limits<float>::max();
        int bestIdx2 = -1;
        float bestDist2 = std::numeric_limits<float>::max();

        for(int idx2=0; idx2<distance.cols; idx2++)
        {
            float dist =distance.at<float>(idx1, idx2);

            if(dist<bestDist1)
            {
                bestDist2=bestDist1;
                bestDist1=dist;
                bestIdx2=idx2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

        if(bestDist1<threshold)
        {
            if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
            {
                DMatch match;
                match.queryIdx = idx1;
                match.trainIdx = bestIdx2;
                match.distance = bestDist1;
                match.imgIdx = 0;
                vMatches.emplace_back(match);
            }
        }
    }

    return vMatches.size();
}

int main(int argc, char* argv[])
{
    const string strModelPath("model/hfnet/");
    const string strResamplerPath("/home/llm/src/tensorflow_cc-2.9.0/tensorflow_cc/install/lib/core/user_ops/resampler/python/ops/_resampler_ops.so");
    const string strDatasetPath("/media/llm/Datasets/EuRoC/MH_01_easy/mav0/cam0/data/");
    const string strSettingsPath("Examples/Monocular-Inertial/EuRoC.yaml");

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    settings = new Settings(strSettingsPath, 0);
    HFNetTFModel::Ptr hfModel = make_shared<HFNetTFModel>(strResamplerPath, strModelPath);

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(1000, files.size() - 1000);

    ORBextractor extractorORB(settings->nFeatures(), settings->scaleFactor(), settings->nLevels(), settings->initThFAST(), settings->minThFAST());
    HFextractor extractorHF(settings->nFeatures(), settings->scaleFactor(), settings->nLevels(), hfModel);
    ORBmatcher matcher(0.9, true);

    const string strVocFileORB("Vocabulary/ORBvoc.txt");
    const string strVocFileHF("Vocabulary/DXSLAM.fbow");

    fbow::Vocabulary vocabHF;
    ORBVocabulary vocabORB;

    if(!vocabORB.loadFromTextFile(strVocFileORB))
    {
        cerr << "Falied to open at: " << strVocFileORB << endl;
        exit(-1);
    }
    vocabHF.readFromFile(strVocFileHF);
    if(!vocabHF.isValid())
    {
        cerr << "Falied to open at: " << strVocFileHF << endl;
        exit(-1);
    }

    char command = ' ';
    bool showUndistort = false;
    unsigned int select;
    auto cameraMatrix = settings->camera1();
    auto distCoef = settings->camera1DistortionCoef();
    do {
        if (command != 'u') select = distribution(generator);
        else showUndistort = !showUndistort;
        cout << command << endl;
        cout << select << endl;
        cv::Mat imageRaw1 = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        cv::Mat imageRaw2 = imread(strDatasetPath + files[select + 10], IMREAD_GRAYSCALE);
        vector<int> vLapping = {0,1000};


        std::vector<cv::KeyPoint> keypointsORB1, keypointsORB2;
        cv::Mat descriptorsORB1, descriptorsORB2;
        extractorORB(imageRaw1, cv::Mat(), keypointsORB1, descriptorsORB1, vLapping);
        extractorORB(imageRaw2, cv::Mat(), keypointsORB2, descriptorsORB2, vLapping);


        std::vector<cv::KeyPoint> keypointsHF1, keypointsHF2;
        cv::Mat descriptorsHF1, descriptorsHF2;
        extractorHF(imageRaw1, cv::Mat(), keypointsHF1, descriptorsHF1, vLapping);
        extractorHF(imageRaw2, cv::Mat(), keypointsHF2, descriptorsHF2, vLapping);


        cv::Mat image1, image2;
        if (showUndistort)
        {
            cv::undistort(imageRaw1, image1, static_cast<Pinhole*>(cameraMatrix)->toK(), distCoef);
            cv::undistort(imageRaw2, image2, static_cast<Pinhole*>(cameraMatrix)->toK(), distCoef);
            keypointsORB1 = undistortPoints(keypointsORB1, static_cast<Pinhole*>(cameraMatrix)->toK(), distCoef);
            keypointsORB2 = undistortPoints(keypointsORB2, static_cast<Pinhole*>(cameraMatrix)->toK(), distCoef);
            keypointsHF1 = undistortPoints(keypointsHF1, static_cast<Pinhole*>(cameraMatrix)->toK(), distCoef);
            keypointsHF2 = undistortPoints(keypointsHF2, static_cast<Pinhole*>(cameraMatrix)->toK(), distCoef);
        }
        else
        {
            image1 = imageRaw1, image2 = imageRaw2;
        }

        cout << "-------------------------------------------------------" << endl;
        std::vector<cv::DMatch> matchesORB, inlierMatchesORB, wrongMatchesORB;
        {
            auto t1 = chrono::steady_clock::now();
            cv::BFMatcher cvMatcherORB(cv::NORM_HAMMING, true);
            cvMatcherORB.match(descriptorsORB1, descriptorsORB2, matchesORB);
            auto t2 = chrono::steady_clock::now();
            auto timeCost = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            cv::Mat plotORB = ShowCorrectMatches(image1, image2, keypointsORB1, keypointsORB2, matchesORB, inlierMatchesORB, wrongMatchesORB);
            cv::imshow("ORB + BFMatcher", plotORB);
            cout << "ORB + BFMatcher:" << endl;
            cout << "match costs time: " << timeCost << "ms" << endl;
            cout << "matches total number: " << matchesORB.size() << endl;
            cout << "correct matches number: " << inlierMatchesORB.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesORB.size() / matchesORB.size() << endl;
        }
        {
            auto t1 = chrono::steady_clock::now();
            DBoW2::BowVector bowVecORB1, bowVecORB2;
            DBoW2::FeatureVector featVecORB1, featVecORB2;
            auto descVecORB1 = toDescriptorVector(descriptorsORB1);
            auto descVecORB2 = toDescriptorVector(descriptorsORB2);
            vocabORB.transform(descVecORB1, bowVecORB1, featVecORB1, 4);
            vocabORB.transform(descVecORB2, bowVecORB2, featVecORB2, 4);
            auto t2 = chrono::steady_clock::now();
            SearchByBoW(cv::NORM_HAMMING, 0.9, true, 50,
                    keypointsORB1, descriptorsORB1, featVecORB1, 
                    keypointsORB2, descriptorsORB2, featVecORB2, matchesORB);
            auto t3 = chrono::steady_clock::now();
            auto timeVocabCost = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            auto timeMatchCost = chrono::duration_cast<chrono::milliseconds>(t3 - t2).count();
            cv::Mat plotORB = ShowCorrectMatches(image1, image2, keypointsORB1, keypointsORB2, matchesORB, inlierMatchesORB, wrongMatchesORB);
            cv::imshow("ORB + SearchByBoW", plotORB);
            cout << "ORB + SearchByBoW:" << endl;
            cout << "vocab costs time: " << timeVocabCost << "ms" << endl;
            cout << "match costs time: " << timeMatchCost << "ms" << endl;
            cout << "matches total number: " << matchesORB.size() << endl;
            cout << "correct matches number: " << inlierMatchesORB.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesORB.size() / matchesORB.size() << endl;
        }


        std::vector<cv::DMatch> matchesHF, inlierMatchesHF, wrongMatchesHF;
        {
            auto t1 = chrono::steady_clock::now();
            cv::BFMatcher cvMatcherHF(cv::NORM_L1, true);
            cvMatcherHF.match(descriptorsHF1, descriptorsHF2, matchesHF);
            auto t2 = chrono::steady_clock::now();
            auto timeCost = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            cv::Mat plotHF = ShowCorrectMatches(image1, image2, keypointsHF1, keypointsHF2, matchesHF, inlierMatchesHF, wrongMatchesHF);
            cv::imshow("HF + BFMatcher", plotHF);
            cout << "HF + BFMatcher_L1:" << endl;
            cout << "match costs time: " << timeCost << "ms" << endl;
            cout << "matches total number: " << matchesHF.size() << endl;
            cout << "correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesHF.size() / matchesHF.size() << endl;
        }
        {
            auto t1 = chrono::steady_clock::now();
            cv::BFMatcher cvMatcherHF(cv::NORM_L2, true);
            cvMatcherHF.match(descriptorsHF1, descriptorsHF2, matchesHF);
            auto t2 = chrono::steady_clock::now();
            auto timeCost = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            FindCorrectMatches(keypointsHF1, keypointsHF2, matchesHF, inlierMatchesHF, wrongMatchesHF);
            cout << "HF + BFMatcher_L2:" << endl;
            cout << "match costs time: " << timeCost << "ms" << endl;
            cout << "matches total number: " << matchesHF.size() << endl;
            cout << "correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesHF.size() / matchesHF.size() << endl;
        }
        {
            auto t1 = chrono::steady_clock::now();
            fbow::fBow bowVecHF1, bowVecHF2;
            fbow::fBow2 featVecHF1, featVecHF2;
            vocabHF.transform(descriptorsHF1, 3, bowVecHF1, featVecHF1);
            vocabHF.transform(descriptorsHF2, 3, bowVecHF2, featVecHF2);
            auto t2 = chrono::steady_clock::now();
            SearchByBoW(cv::NORM_L1, 0.9, false, 15,
                    keypointsHF1, descriptorsHF1, featVecHF1, 
                    keypointsHF2, descriptorsHF2, featVecHF2, matchesHF);
            auto t3 = chrono::steady_clock::now();
            auto timeVocabCost = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            auto timeMatchCost = chrono::duration_cast<chrono::milliseconds>(t3 - t2).count();
            cv::Mat plotHF = ShowCorrectMatches(image1, image2, keypointsHF1, keypointsHF2, matchesHF, inlierMatchesHF, wrongMatchesHF);
            cv::imshow("HF + SearchByBoW_L1", plotHF);
            cout << "HF + SearchByBoW_L1:" << endl;
            cout << "vocab costs time: " << timeVocabCost << "ms" << endl;
            cout << "match costs time: " << timeMatchCost << "ms" << endl;
            cout << "matches total number: " << matchesHF.size() << endl;
            cout << "correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesHF.size() / matchesHF.size() << endl;
        }
        {
            auto t1 = chrono::steady_clock::now();
            fbow::fBow bowVecHF1, bowVecHF2;
            fbow::fBow2 featVecHF1, featVecHF2;
            vocabHF.transform(descriptorsHF1, 3, bowVecHF1, featVecHF1);
            vocabHF.transform(descriptorsHF2, 3, bowVecHF2, featVecHF2);
            auto t2 = chrono::steady_clock::now();
            SearchByBoW(cv::NORM_L2, 0.9, false, 15,
                    keypointsHF1, descriptorsHF1, featVecHF1, 
                    keypointsHF2, descriptorsHF2, featVecHF2, matchesHF);
            auto t3 = chrono::steady_clock::now();
            auto timeVocabCost = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            auto timeMatchCost = chrono::duration_cast<chrono::milliseconds>(t3 - t2).count();
            cout << "HF + SearchByBoW_L2:" << endl;
            cout << "vocab costs time: " << timeVocabCost << "ms" << endl;
            cout << "match costs time: " << timeMatchCost << "ms" << endl;
            cout << "matches total number: " << matchesHF.size() << endl;
            cout << "correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesHF.size() / matchesHF.size() << endl;
        }
        {
            auto t1 = chrono::steady_clock::now();
            SearchByBoWV2(0.9, 15, descriptorsHF1, descriptorsHF2, matchesHF);
            auto t2 = chrono::steady_clock::now();
            auto timeCost = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            cout << "HF + SearchByBoWV2:" << endl;
            cout << "match costs time: " << timeCost << "ms" << endl;
            cout << "matches total number: " << matchesHF.size() << endl;
            cout << "correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesHF.size() / matchesHF.size() << endl;
        }
    } while ((command = cv::waitKey()) != 'q');

    return 0;
}
