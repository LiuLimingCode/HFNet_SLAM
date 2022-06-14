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
#include "Extractors/HFextractor.h"
#include "Examples/Utility/utility_common.h"
#include "CameraModels/Pinhole.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;

Settings *settings;

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
    const string strDatasetPath("/media/llm/Datasets/EuRoC/MH_01_easy/mav0/cam0/data/");
    const string strSettingsPath("Examples/Monocular-Inertial/EuRoC.yaml");

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    settings = new Settings(strSettingsPath, 0);
    HFNetTFModel *pModel = new HFNetTFModel(settings->strResamplerPath(), settings->strModelPath());

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(1000, files.size() - 1000);

    HFextractor extractorHF(settings->nFeatures(),settings->nNMSRadius(),settings->threshold(),settings->scaleFactor(),settings->nLevels(),pModel);

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


        std::vector<cv::KeyPoint> keypointsHF1, keypointsHF2;
        cv::Mat descriptorsHF1, descriptorsHF2;
        cv::Mat globalDescriptorsHF;
        extractorHF(imageRaw1, keypointsHF1, descriptorsHF1, globalDescriptorsHF);
        extractorHF(imageRaw2, keypointsHF2, descriptorsHF2, globalDescriptorsHF);


        cv::Mat image1, image2;
        if (showUndistort)
        {
            cv::undistort(imageRaw1, image1, static_cast<Pinhole*>(cameraMatrix)->toK(), distCoef);
            cv::undistort(imageRaw2, image2, static_cast<Pinhole*>(cameraMatrix)->toK(), distCoef);
            keypointsHF1 = undistortPoints(keypointsHF1, static_cast<Pinhole*>(cameraMatrix)->toK(), distCoef);
            keypointsHF2 = undistortPoints(keypointsHF2, static_cast<Pinhole*>(cameraMatrix)->toK(), distCoef);
        }
        else
        {
            image1 = imageRaw1, image2 = imageRaw2;
        }

        cout << "-------------------------------------------------------" << endl;
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
