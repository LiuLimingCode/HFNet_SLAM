#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>

#include "../include/Settings.h"
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

int nFeatures = 1000;
const int nLevels = 1;
const float scaleFactor = 1.2f;
const float fThreshold = 0.01;

std::vector<std::tuple<double, std::string, std::string>> ReadTUMRGBDDataset(const std::string& strDatasetPath)
{
    std::vector<std::tuple<double, std::string, std::string>> res;

    {
        const std::string strFile = strDatasetPath + "/associations.txt";
        
        FILE *f = fopen(strFile.c_str(), "r");
        if (f == nullptr) {
            cout << "[ERROR]: can't load associations.txt; wrong path: " << strFile.c_str() << endl;
            return res;
        }

        while(!feof(f)) {
            char line[300];
            fgets(line, 300, f);
            
            double tImage, tDepth;
            char strImage[100], strDepth[100];

            auto result = sscanf(line, "%lf %s %lf %s\n", 
                &tImage, strImage, &tDepth, strDepth);

            res.push_back(make_tuple(tImage, strDatasetPath + std::string(strImage), strDatasetPath + std::string(strDepth)));
        }

        fclose(f);
    }

    return res;
}

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
    const std::string strPathSaving = "evaluation/compare_matchers_tum_rgbd/";
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
    const std::string strSequencePath = strDatasetPath + "/" + sequenceName + "/";
    auto sequenceData = ReadTUMRGBDDataset(strSequencePath); // get all image files

    const int interval = 10;

    for (size_t select = 0; select < sequenceData.size() - interval; select += interval)
    {
        cv::Mat imageRaw1 = imread(std::get<1>(sequenceData[select]), IMREAD_GRAYSCALE);
        cv::Mat imageRaw2 = imread(std::get<1>(sequenceData[select + interval]), IMREAD_GRAYSCALE);
        cv::Mat depth1 = imread(std::get<2>(sequenceData[select]), IMREAD_UNCHANGED);
        cv::Mat depth2 = imread(std::get<2>(sequenceData[select + interval]), IMREAD_UNCHANGED);
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
            FindCorrectMatchesByPnP(keypointsORB1, depth1, keypointsORB2, cameraMatrix, distCoef, matchesORB, inlierMatchesORB, wrongMatchesORB);
            vnInlierNumORB.push_back(inlierMatchesORB.size());
            if (matchesORB.empty()) vfInlierRatioORB.push_back(0);
            else vfInlierRatioORB.push_back((float)inlierMatchesORB.size()/(float)(matchesORB.size()));
            // cv::Mat plotORB = ShowCorrectMatches(imageRaw1, imageRaw2, keypointsORB1, keypointsORB2, inlierMatchesORB, wrongMatchesORB);
            // cv::imshow("plotORB", plotORB);
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
            FindCorrectMatchesByPnP(keypointsHF1, depth1, keypointsHF2, cameraMatrix, distCoef, matchesHF, inlierMatchesHF, wrongMatchesHF);
            vnInlierNumHF_NN.push_back(inlierMatchesHF.size());
            if (matchesHF.empty()) vfInlierRatioHF_NN.push_back(0);
            else vfInlierRatioHF_NN.push_back((float)inlierMatchesHF.size()/(float)(matchesHF.size()));
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
            FindCorrectMatchesByPnP(keypointsHF1, depth1, keypointsHF2, cameraMatrix, distCoef, matchesHF, inlierMatchesHF, wrongMatchesHF);
            vnInlierNumHF_NN_Mutual.push_back(inlierMatchesHF.size());
            if (matchesHF.empty()) vfInlierRatioHF_NN_Mutual.push_back(0);
            else vfInlierRatioHF_NN_Mutual.push_back((float)inlierMatchesHF.size()/(float)(matchesHF.size()));
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
            FindCorrectMatchesByPnP(keypointsHF1, depth1, keypointsHF2, cameraMatrix, distCoef, matchesHF, inlierMatchesHF, wrongMatchesHF);
            vnInlierNumHF_NN_Mutual_Ratio.push_back(inlierMatchesHF.size());
            if (matchesHF.empty()) vfInlierRatioHF_NN_Mutual_Ratio.push_back(0);
            else vfInlierRatioHF_NN_Mutual_Ratio.push_back((float)inlierMatchesHF.size()/(float)(matchesHF.size()));
            // cv::Mat plotHF = ShowCorrectMatches(imageRaw1, imageRaw2, keypointsHF1, keypointsHF2, inlierMatchesHF, wrongMatchesHF);
            // cv::imshow("plotHF", plotHF);
        }

        // HFNet-SLAM match without SIMD (NN + mutual + ratio)
        {
            bool mutual = false;
            const float matchThreshold = 0.6;
            const float ratioThreshold = 0.9;

            std::vector<cv::DMatch> matchesHF, inlierMatchesHF, wrongMatchesHF;
            timerHF_Slow.Tic();
            SearchByBoWHFNetSLAM_BFMatcher(ratioThreshold, matchThreshold, mutual, descriptorsHF1, descriptorsHF2, matchesHF);
            timerHF_Slow.Toc();
            FindCorrectMatchesByPnP(keypointsHF1, depth1, keypointsHF2, cameraMatrix, distCoef, matchesHF, inlierMatchesHF, wrongMatchesHF);
            vnInlierNumHF_Slow.push_back(inlierMatchesHF.size());
            if (matchesHF.empty()) vfInlierRatioHF_Slow.push_back(0);
            else vfInlierRatioHF_Slow.push_back((float)inlierMatchesHF.size()/(float)(matchesHF.size()));
        }
        // cv::waitKey();
    }
}

cv::Size ImSize(640,480);

int main(int argc, char* argv[])
{
    if (argc != 5) {
        cerr << endl << "Usage: compare_matchers_tum_rgbd path_to_dataset path_to_model path_to_vocabulary feature_number" << endl;   
        return -1;
    }
    strDatasetPath = string(argv[1]);
    strModelPath = string(argv[2]);
    strVocFileORB = string(argv[3]);
    nFeatures = atoi(argv[4]);

    cout << "nFeatures: " << nFeatures << endl;

    // By default, the Eigen will use the maximum number of threads in OpenMP.
    // However, this will somehow slow down the calculation of dense matrix multiplication.
    // Therefore, use only half of the thresds.
    Eigen::setNbThreads(std::max(Eigen::nbThreads() / 2, 1));

    InitAllModels(strModelPath, kHFNetRTModel, ImSize, nLevels, scaleFactor);
    auto vpModels = GetModelVec();

    ORBextractor extractorORB(nFeatures, scaleFactor, nLevels, 20, 7);
    HFextractor extractorHF(nFeatures, fThreshold, scaleFactor, nLevels, vpModels);

    ORBVocabulary vocabORB;
    if(!vocabORB.loadFromTextFile(strVocFileORB))
    {
        cerr << "Falied to open at: " << strVocFileORB << endl;
        exit(-1);
    }

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    // cv::Mat distCoef = (cv::Mat_<double>(5, 1) << 0.231222, -0.784899, -0.003257, -0.000105, 0.917205);
    cv::Mat distCoef;

    clearResult();
    evaluation("rgbd_dataset_freiburg2_pioneer_360", cameraMatrix, distCoef, vocabORB, extractorORB, extractorHF);
    evaluation("rgbd_dataset_freiburg2_pioneer_slam", cameraMatrix, distCoef, vocabORB, extractorORB, extractorHF);
    evaluation("rgbd_dataset_freiburg2_pioneer_slam2", cameraMatrix, distCoef, vocabORB, extractorORB, extractorHF);
    evaluation("rgbd_dataset_freiburg2_pioneer_slam3", cameraMatrix, distCoef, vocabORB, extractorORB, extractorHF);
    saveResult("pioneer");

    return 0;
}