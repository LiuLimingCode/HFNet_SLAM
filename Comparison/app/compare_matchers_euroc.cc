#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>

#include "../include/CameraModels/Pinhole.h"
#include "../include/Extractors/HFextractor.h"

#include "ORBextractor.h"
#include "ORBVocabulary.h"

#include "../include/utility_common.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;

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
    vector<int> vpMatches12 = vector<int>(vKeysUn1.size(),-1);
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

int SearchByBoWHFNetSLAM_BFMatcher(float mfNNratio, float threshold, bool mutual,
                            cv::Mat& Descriptors1, cv::Mat& Descriptors2,
                            std::vector<cv::DMatch>& vMatches)
{
    vMatches.clear();

    cv::BFMatcher matcher(cv::NORM_L2, mutual);
    vector<cv::DMatch> matches;
    matcher.match(Descriptors1, Descriptors2, matches);

    vMatches.reserve(matches.size());
    for (const auto &m : matches)
    {
        if (m.distance > threshold) continue;
        vMatches.emplace_back(m);
    }

    return vMatches.size();
}

string strDatasetPath;
string strVocFileORB;
string strModelPath;

int nFeatures = 350;
const float scaleFactor = 1.0f;
const int nLevels = 1;
const float fThreshold = 0.01;

TicToc timerORB, timerHF_NN, timerHF_NN_Mutual, timerHF_NN_Mutual_Ratio, timerHF_Slow;
vector<float> vfInlierRatioORB, vfInlierRatioHF_NN, vfInlierRatioHF_NN_Mutual, vfInlierRatioHF_NN_Mutual_Ratio, vfInlierRatioHF_Slow;
vector<int> vnInlierNumORB, vnInlierNumHF_NN, vnInlierNumHF_NN_Mutual, vnInlierNumHF_NN_Mutual_Ratio, vnInlierNumHF_Slow;

void clearResult()
{
    timerORB.timeBuff.clear();
    vnInlierNumORB.clear();
    vfInlierRatioORB.clear();
    timerHF_NN.timeBuff.clear();
    vnInlierNumHF_NN.clear();
    vfInlierRatioHF_NN.clear(); 
    timerHF_NN_Mutual.timeBuff.clear();
    vnInlierNumHF_NN_Mutual.clear();
    vfInlierRatioHF_NN_Mutual.clear(); 
    timerHF_NN_Mutual_Ratio.timeBuff.clear();
    vnInlierNumHF_NN_Mutual_Ratio.clear();
    vfInlierRatioHF_NN_Mutual_Ratio.clear();
    timerHF_Slow.timeBuff.clear();
    vnInlierNumHF_Slow.clear();
    vfInlierRatioHF_Slow.clear();
}

void saveResult(std::string sequenceName)
{
    const std::string strPathSaving = "evaluation/compare_matchers_euroc/";
    const std::string strFileSaving =  sequenceName + ".csv";
    system(("mkdir -p " + strPathSaving).c_str());

    ofstream f;
    f.open(strPathSaving + strFileSaving);
    f << fixed;

    // titile
    f << "TimeORB" << ","
      << "InlierNumORB" << ","
      << "InlierRatioORB" << ","
      << "TimeHF_NN" << ","
      << "InlierNumHF_NN" << ","
      << "InlierRatioHF_NN" << ","
      << "TimeHF_NN_Mutual" << ","
      << "InlierNumHF_NN_Mutual" << ","
      << "InlierRatioHF_NN_Mutual" << ","
      << "TimeHF_NN_Mutual_Ratio" << ","
      << "InlierNumHF_NN_Mutual_Ratio" << ","
      << "InlierRatioHF_NN_Mutual_Ratio" << ","
      << "TimeHF_Slow" << ","
      << "InlierNumHF_Slow" << ","
      << "InlierRatioHF_Slow" << endl;

    for (int index = 0; index < vnInlierNumORB.size(); ++index)
    {
        f << timerORB.timeBuff[index] << ","
          << vnInlierNumORB[index] << ","
          << vfInlierRatioORB[index] << ","
          << timerHF_NN.timeBuff[index] << ","
          << vnInlierNumHF_NN[index] << ","
          << vfInlierRatioHF_NN[index] << "," 
          << timerHF_NN_Mutual.timeBuff[index] << ","
          << vnInlierNumHF_NN_Mutual[index] << ","
          << vfInlierRatioHF_NN_Mutual[index] << "," 
          << timerHF_NN_Mutual_Ratio.timeBuff[index] << ","
          << vnInlierNumHF_NN_Mutual_Ratio[index] << ","
          << vfInlierRatioHF_NN_Mutual_Ratio[index] << ","
          << timerHF_Slow.timeBuff[index] << ","
          << vnInlierNumHF_Slow[index] << ","
          << vfInlierRatioHF_Slow[index] << endl;
    }
    
    f.close();
}

void evaluation(const string sequenceName, const cv::Mat &cameraMatrix, const cv::Mat &distCoef, 
    ORBVocabulary &vocabORB, ORBextractor &extractorORB, HFextractor &extractorHF)
{
    cout << "Running evaluation on " + sequenceName << endl;
    const std::string strSequencePath = strDatasetPath + "/" + sequenceName + "/mav0/cam0/data/";
    vector<string> sequenceData = GetPngFiles(strSequencePath);
    const int interval = 10;

    for (size_t select = 0; select < sequenceData.size() - interval; select += interval)
    {
        auto file1 = sequenceData[select];
        auto file2 = sequenceData[select + interval];
        cv::Mat imageRaw1 = imread(strSequencePath + file1, IMREAD_GRAYSCALE);
        cv::Mat imageRaw2 = imread(strSequencePath + file2, IMREAD_GRAYSCALE);
        vector<int> vLapping = {0,1000};

        std::vector<cv::KeyPoint> keypointsORB1, keypointsORB2;
        cv::Mat descriptorsORB1, descriptorsORB2;
        extractorORB(imageRaw1, cv::Mat(), keypointsORB1, descriptorsORB1, vLapping);
        extractorORB(imageRaw2, cv::Mat(), keypointsORB2, descriptorsORB2, vLapping);

        extractorHF.threshold = fThreshold;
        std::vector<cv::KeyPoint> keypointsHF1, keypointsHF2;
        cv::Mat descriptorsHF1, descriptorsHF2, globalDescriptors;
        extractorHF(imageRaw1, keypointsHF1, descriptorsHF1, globalDescriptors);
        extractorHF(imageRaw2, keypointsHF2, descriptorsHF2, globalDescriptors);

        keypointsORB1 = undistortPoints(keypointsORB1, cameraMatrix, distCoef);
        keypointsORB2 = undistortPoints(keypointsORB2, cameraMatrix, distCoef);
        keypointsHF1 = undistortPoints(keypointsHF1, cameraMatrix, distCoef);
        keypointsHF2 = undistortPoints(keypointsHF2, cameraMatrix, distCoef);

        // ORB-SLAM3
        {
            const float matchThreshold = 50;
            const float ratioThreshold = 0.9;

            std::vector<cv::DMatch> matchesORB, inlierMatchesORB, wrongMatchesORB;
            DBoW2::BowVector bowVecORB1, bowVecORB2;
            DBoW2::FeatureVector featVecORB1, featVecORB2;
            auto descVecORB1 = toDescriptorVector(descriptorsORB1);
            auto descVecORB2 = toDescriptorVector(descriptorsORB2);
            vocabORB.transform(descVecORB1, bowVecORB1, featVecORB1, 4);
            vocabORB.transform(descVecORB2, bowVecORB2, featVecORB2, 4);
            timerORB.Tic();
            SearchByBoWORBSLAM3(ratioThreshold, true, matchThreshold,
                    keypointsORB1, descriptorsORB1, featVecORB1, 
                    keypointsORB2, descriptorsORB2, featVecORB2, matchesORB);
            timerORB.Toc();
            cv::Mat E = FindCorrectMatchesByEssentialMat(keypointsORB1, keypointsORB2, matchesORB, cameraMatrix, inlierMatchesORB, wrongMatchesORB);
            vnInlierNumORB.push_back(inlierMatchesORB.size());
            vfInlierRatioORB.push_back((float)inlierMatchesORB.size()/(float)(matchesORB.size()));
        }

        // HFNet-SLAM (NN)
        {
            bool mutual = false;
            const float matchThreshold = 0.6;
            const float ratioThreshold = 1.0;

            std::vector<cv::DMatch> matchesHF, inlierMatchesHF, wrongMatchesHF;
            timerHF_NN.Tic();
            SearchByBoWHFNetSLAM(ratioThreshold, matchThreshold * matchThreshold, mutual, descriptorsHF1, descriptorsHF2, matchesHF);
            timerHF_NN.Toc();
            cv::Mat E = FindCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, matchesHF, cameraMatrix, inlierMatchesHF, wrongMatchesHF);
            vnInlierNumHF_NN.push_back(inlierMatchesHF.size());
            vfInlierRatioHF_NN.push_back((float)inlierMatchesHF.size()/(float)(matchesHF.size()));
        }

        // HFNet-SLAM (NN + mutual)
        {
            bool mutual = true;
            const float matchThreshold = 0.6;
            const float ratioThreshold = 1.0;

            std::vector<cv::DMatch> matchesHF, inlierMatchesHF, wrongMatchesHF;
            timerHF_NN_Mutual.Tic();
            SearchByBoWHFNetSLAM(ratioThreshold, matchThreshold * matchThreshold, mutual, descriptorsHF1, descriptorsHF2, matchesHF);
            timerHF_NN_Mutual.Toc();
            cv::Mat E = FindCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, matchesHF, cameraMatrix, inlierMatchesHF, wrongMatchesHF);
            vnInlierNumHF_NN_Mutual.push_back(inlierMatchesHF.size());
            vfInlierRatioHF_NN_Mutual.push_back((float)inlierMatchesHF.size()/(float)(matchesHF.size()));
        }

        // HFNet-SLAM (NN + mutual + ratio)
        {
            bool mutual = true;
            const float matchThreshold = 0.6;
            const float ratioThreshold = 0.9;

            std::vector<cv::DMatch> matchesHF, inlierMatchesHF, wrongMatchesHF;
            timerHF_NN_Mutual_Ratio.Tic();
            SearchByBoWHFNetSLAM(ratioThreshold, matchThreshold * matchThreshold, mutual, descriptorsHF1, descriptorsHF2, matchesHF);
            timerHF_NN_Mutual_Ratio.Toc();
            cv::Mat E = FindCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, matchesHF, cameraMatrix, inlierMatchesHF, wrongMatchesHF);
            vnInlierNumHF_NN_Mutual_Ratio.push_back(inlierMatchesHF.size());
            vfInlierRatioHF_NN_Mutual_Ratio.push_back((float)inlierMatchesHF.size()/(float)(matchesHF.size()));
        }

        // HFNet-SLAM match without SIMD (NN + mutual + ratio)
        {
            bool mutual = true;
            const float matchThreshold = 0.6;
            const float ratioThreshold = 0.9;

            std::vector<cv::DMatch> matchesHF, inlierMatchesHF, wrongMatchesHF;
            timerHF_Slow.Tic();
            SearchByBoWHFNetSLAM_BFMatcher(ratioThreshold, matchThreshold, mutual, descriptorsHF1, descriptorsHF2, matchesHF);
            timerHF_Slow.Toc();
            cv::Mat E = FindCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, matchesHF, cameraMatrix, inlierMatchesHF, wrongMatchesHF);
            vnInlierNumHF_Slow.push_back(inlierMatchesHF.size());
            vfInlierRatioHF_Slow.push_back((float)inlierMatchesHF.size()/(float)(matchesHF.size()));
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 5) {
        cerr << endl << "Usage: compare_matchers_euroc path_to_dataset path_to_model path_to_vocabulary feature_number" << endl;   
        return -1;
    }
    strDatasetPath = string(argv[1]);
    strModelPath = string(argv[2]);
    strVocFileORB = string(argv[3]);
    nFeatures = atoi(argv[4]);

    // By default, the Eigen will use the maximum number of threads in OpenMP.
    // However, this will somehow slow down the calculation of dense matrix multiplication.
    // Therefore, use only half of the thresds.
    Eigen::setNbThreads(std::max(Eigen::nbThreads() / 2, 1));

    cv::Size ImSize(752, 480);
    cv::Vec4i inputShape{1, ImSize.height, ImSize.width, 1};
    BaseModel *pNewModel = InitRTModel(strModelPath, kImageToLocalAndIntermediate, inputShape);
    // BaseModel *pNewModel = InitTFModel(strTFModelPath, kImageToLocalAndIntermediate, inputShape);

    ORBextractor extractorORB(nFeatures, scaleFactor, nLevels, 20, 7);
    HFextractor extractorHF(nFeatures, fThreshold, pNewModel);

    ORBVocabulary vocabORB;
    if(!vocabORB.loadFromTextFile(strVocFileORB))
    {
        cerr << "Falied to open at: " << strVocFileORB << endl;
        exit(-1);
    }

    const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 458.654, 0.0, 367.215, 0.0, 457.296, 248.375, 0.0, 0.0, 1.0);
    const cv::Mat distCoeffs = (cv::Mat_<double>(4, 1) << -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);

    clearResult();
    evaluation("MH_01_easy", cameraMatrix, distCoeffs, vocabORB, extractorORB, extractorHF);
    evaluation("MH_02_easy", cameraMatrix, distCoeffs, vocabORB, extractorORB, extractorHF);
    evaluation("MH_03_medium", cameraMatrix, distCoeffs, vocabORB, extractorORB, extractorHF);
    evaluation("MH_04_difficult", cameraMatrix, distCoeffs, vocabORB, extractorORB, extractorHF);
    evaluation("MH_05_difficult", cameraMatrix, distCoeffs, vocabORB, extractorORB, extractorHF);
    saveResult("MH");

    clearResult();
    evaluation("V1_01_easy", cameraMatrix, distCoeffs, vocabORB, extractorORB, extractorHF);
    evaluation("V1_02_medium", cameraMatrix, distCoeffs, vocabORB, extractorORB, extractorHF);
    evaluation("V1_03_difficult", cameraMatrix, distCoeffs, vocabORB, extractorORB, extractorHF);
    saveResult("V1");

    clearResult();
    evaluation("V2_01_easy", cameraMatrix, distCoeffs, vocabORB, extractorORB, extractorHF);
    evaluation("V2_02_medium", cameraMatrix, distCoeffs, vocabORB, extractorORB, extractorHF);
    evaluation("V2_03_difficult", cameraMatrix, distCoeffs, vocabORB, extractorORB, extractorHF);
    saveResult("V2");

    return 0;
}