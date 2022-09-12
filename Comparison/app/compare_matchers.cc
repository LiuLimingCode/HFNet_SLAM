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

#include "../include/Settings.h"
#include "../include/CameraModels/Pinhole.h"
#include "../include/Extractors/HFNetTFModelV2.h"

#include "ORBextractor.h"
#include "ORBVocabulary.h"

#include "app/utility_common.h"

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

int DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
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

    return dist;
}

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

// This function is original from ORB-SLAM3
const int HISTO_LENGTH = 30;
int SearchByBoWORBSLAM3(float mfNNratio, bool mbCheckOrientation, int threshold,
                std::vector<cv::KeyPoint>& vKeysUn1, cv::Mat& Descriptors1, DBoW2::FeatureVector& vFeatVec1, 
                std::vector<cv::KeyPoint>& vKeysUn2, cv::Mat& Descriptors2, DBoW2::FeatureVector& vFeatVec2,
                std::vector<cv::DMatch>& vMatches)
{
    vector<int> vpMatches12(vKeysUn1.size(),-1);
    vector<bool> vbMatched2(vKeysUn2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    if(vbMatched2[idx2])
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1,d2);

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
                        vpMatches12[idx1]=bestIdx2;
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
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
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
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
                vpMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }
    }

    vMatches.clear();
    for (int index = 0; index < vpMatches12.size(); ++index)
    {
        if (vpMatches12[index] != -1)
        {
            DMatch match;
            match.queryIdx = index;
            match.trainIdx = vpMatches12[index];
            match.imgIdx = 0;
            vMatches.emplace_back(match);
        }
    }

    return nmatches;
}

int SearchByBoWHFNetSLAM(float mfNNratio, float threshold, bool mutual,
                         cv::Mat& Descriptors1, cv::Mat& Descriptors2,
                         std::vector<cv::DMatch>& vMatches)
{
    vMatches.clear();
    vMatches.reserve(Descriptors1.rows);

    assert(Descriptors1.isContinuous() && Descriptors2.isContinuous());
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> des1(Descriptors1.ptr<float>(), Descriptors1.rows, Descriptors1.cols);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> des2(Descriptors2.ptr<float>(), Descriptors2.rows, Descriptors2.cols);
    // cv::Mat distanceCV = 2 * (1 - Descriptors1 * Descriptors2.t());
    Eigen::MatrixXf distance = 2 * (Eigen::MatrixXf::Ones(Descriptors1.rows, Descriptors2.rows) - des1 * des2.transpose());

    vector<int> matchedIdx2(Descriptors2.rows, -1);
    vector<float> matchedDist(Descriptors2.rows, std::numeric_limits<float>::max());
    for(int idx1=0; idx1 < distance.rows(); idx1++)
    {
        float bestDist1 = std::numeric_limits<float>::max();
        int bestIdx2 = -1;
        float bestDist2 = std::numeric_limits<float>::max();

        for(int idx2=0; idx2<distance.cols(); idx2++)
        {
            float dist = distance(idx1, idx2);

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
            if(bestDist1 < mfNNratio * bestDist2)
            {

                int bestCrossIdx1 = -1;
                if (mutual)
                {
                    // cross check
                    float bestDist = std::numeric_limits<float>::max();
                    for(int crossIdx1 = 0; crossIdx1 < distance.rows(); crossIdx1++)
                    {
                        float dist = distance(crossIdx1, bestIdx2);
                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestCrossIdx1 = crossIdx1;
                        }
                    }
                }
                

                if (!mutual || bestCrossIdx1 == idx1)
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
    }
    
    return vMatches.size();
}

// const string strDatasetPath("/media/llm/Datasets/EuRoC/MH_04_difficult/mav0/cam0/data/");
// const string strSettingsPath("Examples/Monocular-Inertial/EuRoC.yaml");
// const int dbStart = 420;
// const int dbEnd = 50;

const string strDatasetPath("/media/llm/Datasets/TUM-VI/dataset-outdoors3_512_16/mav0/cam0/data/");
const string strSettingsPath("Examples/Monocular-Inertial//TUM-VI.yaml");
const int dbStart = 50;
const int dbEnd = 50;

const string strVocFileORB("/home/llm/ROS/HFNet_ORBSLAM3_v2/Comparison/Vocabulary/ORBvoc.txt");

int main(int argc, char* argv[])
{
    // By default, the Eigen will use the maximum number of threads in OpenMP.
    // However, this will somehow slow down the calculation of dense matrix multiplication.
    // Therefore, use only half of the thresds.
    Eigen::setNbThreads(std::max(Eigen::nbThreads() / 2, 1));
    
    const string strTFModelPath("/home/llm/ROS/HFNet_ORBSLAM3_v2/model/hfnet_tf_v2_NMS2");

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

    std::default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<unsigned int> distribution(dbStart, files.size() - dbEnd);

    int nFeatures = settings->nFeatures();
    float threshold = settings->threshold();
    int nNMSRadius = settings->nNMSRadius();
    ORBextractor extractorORB(1250, 1.2, 8, 20, 7);
    HFextractor extractorHF(nFeatures, threshold, nNMSRadius, settings->scaleFactor(), settings->nLevels(), vpModels);

    ORBVocabulary vocabORB;
    if(!vocabORB.loadFromTextFile(strVocFileORB))
    {
        cerr << "Falied to open at: " << strVocFileORB << endl;
        exit(-1);
    }

    char command = ' ';
    float matchThreshold = 1;
    float ratioThreshold = 1.0;
    bool showUndistort = false;
    int select = 14980;
    bool showWholeMatches = false;
    auto cameraMatrix = settings->camera1();
    cv::Mat distCoef;
    if(settings->needToUndistort()) distCoef = settings->camera1DistortionCoef();
    else distCoef = cv::Mat::zeros(4,1,CV_32F);
    while(1)
    {
        if (command == 'x') break;
        else if (command == 'u') showUndistort = !showUndistort;
        else if (command == 'w') select += 1;
        else if (command == 's') select -= 1;
        else if (command == 'a') threshold = std::max(threshold - 0.001, 0.005);
        else if (command == 'd') threshold += 0.001;
        else if (command == 'z') nNMSRadius = std::max(nNMSRadius - 1, 0);
        else if (command == 'c') nNMSRadius += 1;
        else if (command == 'q') matchThreshold -= 0.05;
        else if (command == 'e') matchThreshold += 0.05;
        else if (command == 'r') ratioThreshold -= 0.1;
        else if (command == 'f') ratioThreshold = std::min(ratioThreshold + 0.1, 1.0);
        else if (command == 'i') showWholeMatches = !showWholeMatches;
        else select = distribution(generator);

        cout << "command: " << command << endl;
        cout << "select: " << select << endl;
        cout << "threshold: " << threshold << endl;
        cout << "nNMSRadius: " << nNMSRadius << endl;
        cout << "matchThreshold: " << matchThreshold << endl;
        cout << "ratioThreshold: " << ratioThreshold << endl;
        cv::Mat imageRaw1 = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        if (settings->needToResize()) cv::resize(imageRaw1, imageRaw1, settings->newImSize());
        cv::Mat imageRaw2 = imread(strDatasetPath + files[select + 10], IMREAD_GRAYSCALE);
        if (settings->needToResize()) cv::resize(imageRaw2, imageRaw2, settings->newImSize());
        vector<int> vLapping = {0,1000};


        std::vector<cv::KeyPoint> keypointsORB1, keypointsORB2;
        cv::Mat descriptorsORB1, descriptorsORB2;
        extractorORB(imageRaw1, cv::Mat(), keypointsORB1, descriptorsORB1, vLapping);
        extractorORB(imageRaw2, cv::Mat(), keypointsORB2, descriptorsORB2, vLapping);

        extractorHF.nNMSRadius = nNMSRadius;
        extractorHF.threshold = threshold;
        std::vector<cv::KeyPoint> keypointsHF1, keypointsHF2;
        cv::Mat descriptorsHF1, descriptorsHF2, globalDescriptors;
        extractorHF(imageRaw1, keypointsHF1, descriptorsHF1, globalDescriptors);
        extractorHF(imageRaw2, keypointsHF2, descriptorsHF2, globalDescriptors);


        cv::Mat image1, image2;
        if (showUndistort && settings->needToUndistort())
        {
            cv::undistort(imageRaw1, image1, cameraMatrix->toK(), distCoef);
            cv::undistort(imageRaw2, image2, cameraMatrix->toK(), distCoef);
            keypointsORB1 = undistortPoints(keypointsORB1, cameraMatrix->toK(), distCoef);
            keypointsORB2 = undistortPoints(keypointsORB2, cameraMatrix->toK(), distCoef);
            keypointsHF1 = undistortPoints(keypointsHF1, cameraMatrix->toK(), distCoef);
            keypointsHF2 = undistortPoints(keypointsHF2, cameraMatrix->toK(), distCoef);
        }
        else
        {
            image1 = imageRaw1, image2 = imageRaw2;
        }

        cout << "-------------------------------------------------------" << endl;
        {
            std::vector<cv::DMatch> matchesORB, inlierMatchesORB, wrongMatchesORB;
            TicToc timer;
            timer.Tic();
            DBoW2::BowVector bowVecORB1, bowVecORB2;
            DBoW2::FeatureVector featVecORB1, featVecORB2;
            auto descVecORB1 = toDescriptorVector(descriptorsORB1);
            auto descVecORB2 = toDescriptorVector(descriptorsORB2);
            vocabORB.transform(descVecORB1, bowVecORB1, featVecORB1, 4);
            vocabORB.transform(descVecORB2, bowVecORB2, featVecORB2, 4);
            timer.Toc(); timer.Tic();
            SearchByBoWORBSLAM3(0.9, true, 50,
                    keypointsORB1, descriptorsORB1, featVecORB1, 
                    keypointsORB2, descriptorsORB2, featVecORB2, matchesORB);
            timer.Toc();
            cv::Mat E = FindCorrectMatchesByEssentialMat(keypointsORB1, keypointsORB2, matchesORB, cameraMatrix->toK(), inlierMatchesORB, wrongMatchesORB);
            cv::Mat plotORB = ShowCorrectMatches(image1, image2, keypointsORB1, keypointsORB2, inlierMatchesORB, wrongMatchesORB, showWholeMatches);
            cv::imshow("ORB + SearchByBoW", plotORB);
            cout << "ORB + SearchByBoW:" << endl;
            cout << "vocab costs time: " << timer.timeBuff[0] << "ms" << endl;
            cout << "match costs time: " << timer.timeBuff[1] << "ms" << endl;
            cout << "matches total number: " << matchesORB.size() << endl;
            cout << "correct matches number: " << inlierMatchesORB.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesORB.size() / matchesORB.size() << endl;
        }


        {
            std::vector<cv::DMatch> matchesHF, thresholdMatchesHF, inlierMatchesHF, wrongMatchesHF;
            TicToc timer;
            timer.Tic();
            cv::BFMatcher cvMatcherHF(cv::NORM_L2, true);
            cvMatcherHF.match(descriptorsHF1, descriptorsHF2, matchesHF);
            timer.Toc();
            for (auto& match : matchesHF)
            {
                if (match.distance > matchThreshold) continue;
                thresholdMatchesHF.emplace_back(match);
            }
            cv::Mat E = FindCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, thresholdMatchesHF, cameraMatrix->toK(), inlierMatchesHF, wrongMatchesHF);
            cv::Mat plotHF = ShowCorrectMatches(image1, image2, keypointsHF1, keypointsHF2, inlierMatchesHF, wrongMatchesHF, showWholeMatches);
            cv::imshow("HF + BFMatcher_L2", plotHF);
            cout << "HF + BFMatcher_L2:" << endl;
            cout << "match costs time: " << timer.timeBuff[0] << "ms" << endl;
            cout << "matches total number: " << matchesHF.size() << endl;
            cout << "threshold matches total number: " << thresholdMatchesHF.size() << endl;
            cout << "correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesHF.size() / thresholdMatchesHF.size() << endl;
        }
        {
            std::vector<cv::DMatch> matchesHF, inlierMatchesHF, wrongMatchesHF;
            TicToc timer;
            timer.Tic();
            SearchByBoWHFNetSLAM(ratioThreshold, matchThreshold * matchThreshold, true, descriptorsHF1, descriptorsHF2, matchesHF);
            timer.Toc();
            cv::Mat E = FindCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, matchesHF, cameraMatrix->toK(), inlierMatchesHF, wrongMatchesHF);
            cv::Mat plotHF = ShowCorrectMatches(image1, image2, keypointsHF1, keypointsHF2, inlierMatchesHF, wrongMatchesHF, showWholeMatches);
            cv::imshow("HF + SearchByBoWV2", plotHF);
            cout << "HF + SearchByBoWV2:" << endl;
            cout << "match costs time: " << timer.timeBuff[0] << "ms" << endl;
            cout << "matches total number: " << matchesHF.size() << endl;
            cout << "correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesHF.size() / matchesHF.size() << endl;
        }

        command = cv::waitKey();
    }

    return 0;
}