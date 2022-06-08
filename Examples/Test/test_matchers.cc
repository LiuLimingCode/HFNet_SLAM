/**
 * A test for matching
 * 
 * 
 * Result:
 * 1. HFNet is way better than ORB
 * 2. The L1 and L2 descriptor distance is the same for HFNet 
 * ORB successful matches number: 415
 * HFNet successful matches number with L2: 761
 * HFNet successful matches number with L1: 758
 * 
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
#include "Examples/Test/test_utility.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;

Settings *settings;

void ShowMatchesCV(const cv::Mat &image1, const cv::Mat &image2,
                   const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2,
                   const cv::Mat & descriptors1, const cv::Mat &descriptors2, 
                   int normType, cv::Mat &matchesPlot, std::vector<cv::DMatch> &matches)
{
    cv::BFMatcher cvMatcher(normType, true);
    cvMatcher.match(descriptors1, descriptors2, matches);

    vector<cv::Point2f> vPt1, vPt2;
    for (const auto &match : matches)
    {
        vPt1.emplace_back(keypoints1[match.queryIdx].pt);
        vPt2.emplace_back(keypoints2[match.trainIdx].pt);
    }

    cv::Mat homography;
    std::vector<int> inliers;
    cv::findHomography(vPt1, vPt2, cv::RANSAC, 3.0, inliers);

    std::vector<cv::DMatch> inlierMatches;
    inlierMatches.reserve(matches.size());
    for (size_t index = 0; index < matches.size(); ++index)
    {
        if (inliers[index]) inlierMatches.emplace_back(matches[index]);
    }

    cv::drawMatches(image1, keypoints1, image2, keypoints2, inlierMatches, matchesPlot, cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255));
}

int main(int argc, char* argv[])
{
    const string strModelPath ="model/hfnet/";
    const string strResamplerPath ="/home/llm/src/tensorflow_cc-2.9.0/tensorflow_cc/install/lib/core/user_ops/resampler/python/ops/_resampler_ops.so";
    const string strDatasetPath("/media/llm/Datasets/EuRoC/MH_01_easy/mav0/cam0/data/");
    const string strSettingsPath("Examples/Monocular-Inertial/EuRoC.yaml");

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    settings = new Settings(strSettingsPath, 0);
    HFNetTFModel::Ptr hfModel = make_shared<HFNetTFModel>(strResamplerPath, strModelPath);

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(1000, files.size() - 1000);

    ORBextractor extractorORB(settings->nFeatures(), settings->scaleFactor(), settings->nLevels(), settings->initThFAST(), settings->minThFAST());
    HFextractor extractorHF(settings->nFeatures(), settings->scaleFactor(), settings->nLevels(), hfModel);
    cv::Mat distCoef = settings->camera1DistortionCoef();
    ORBmatcher matcher(0.9, true);

    cv::namedWindow("ORB");
    cv::moveWindow("ORB", 0, 0);
    cv::namedWindow("HFNet");
    cv::moveWindow("HFNet", 0, 540);

    do {
        unsigned int select = distribution(generator);
        cv::Mat image1 = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        cv::Mat image2 = imread(strDatasetPath + files[select + 10], IMREAD_GRAYSCALE);

        std::vector<cv::KeyPoint> keypointsORB1, keypointsORB2;
        cv::Mat descriptorsORB1, descriptorsORB2;
        
        vector<int> vLapping = {0,1000};

        extractorORB(image1, cv::Mat(), keypointsORB1, descriptorsORB1, vLapping);
        extractorORB(image2, cv::Mat(), keypointsORB2, descriptorsORB2, vLapping);

        cv::Mat plotORB;
        std::vector<cv::DMatch> matchesORB;
        ShowMatchesCV(image1, image2, keypointsORB1, keypointsORB2, descriptorsORB1, descriptorsORB2, cv::NormTypes::NORM_HAMMING, plotORB, matchesORB);
        cv::imshow("ORB", plotORB);
        cout << "ORB successful matches number: " << matchesORB.size() << endl;

        std::vector<cv::KeyPoint> keypointsHF1, keypointsHF2;
        cv::Mat descriptorsHF1, descriptorsHF2;
        extractorHF(image1, cv::Mat(), keypointsHF1, descriptorsHF1, vLapping);
        extractorHF(image2, cv::Mat(), keypointsHF2, descriptorsHF2, vLapping);

        cv::Mat plotHF;
        std::vector<cv::DMatch> matchesHF;
        ShowMatchesCV(image1, image2, keypointsHF1, keypointsHF2, descriptorsHF1, descriptorsHF2, cv::NormTypes::NORM_L2, plotHF, matchesHF);
        cv::imshow("HFNet", plotHF);
        cout << "HFNet successful matches number with L2: " << matchesHF.size() << endl;

        ShowMatchesCV(image1, image2, keypointsHF1, keypointsHF2, descriptorsHF1, descriptorsHF2, cv::NormTypes::NORM_L1, plotHF, matchesHF);
        cout << "HFNet successful matches number with L1: " << matchesHF.size() << endl;

        // Frame frameORB1(image1, 0, &extractorORB, nullptr, settings->camera1(), distCoef, 0,0);
        // // KeyFrame keyFrameORB1(frameORB1, nullptr, nullptr);
        // Frame frameORB2(image2, 0.05, &extractorORB, nullptr, settings->camera1(), distCoef, 0,0);
        // // KeyFrame keyFrameORB2(frameORB2, nullptr, nullptr);

        // vector<cv::Point2f> vbPrevMatched;
        // vector<int> mvIniMatches;
        // vbPrevMatched.resize(frameORB1.mvKeysUn.size());
        // for(size_t i=0; i<frameORB1.mvKeysUn.size(); i++)
        //     vbPrevMatched[i]=frameORB1.mvKeysUn[i].pt;

        // fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
        // matcher.SearchForInitialization(frameORB1,frameORB2,vbPrevMatched,mvIniMatches,100);

        
        

    } while (cv::waitKey() != 'q');

    return 0;
}
