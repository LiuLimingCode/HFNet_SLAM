/**
 * A test for matching
 * 
 * 
 * Result:
HF + BFMatcher_L2:
match costs time: 12ms
matches total number: 832
threshold matches total number: 832
correct matches number: 767
match correct percentage: 0.921875
HF + BFMatcher_L1:
match costs time: 25ms
matches total number: 831
threshold matches total number: 829
correct matches number: 780
match correct percentage: 0.940893
HF + SearchByBoWV2:
match costs time: 5ms
matches total number: 934
threshold matches total number: 934
correct matches number: 745
match correct percentage: 0.797645
 * 1. HFNet is way better than ORB, but it is more time-consuming
 * 2. The L1 and L2 descriptor distance is the same for HFNet, but L2 norm is more effective
 * 3. SearchByBoW will increase the matching time
 * 4. SearchByBoW can increase the correct percentage of ORB descriptor
 * 5. SearchByBoW does not work well for HF descriptor, maybe it is because the vocabulary for HF is bad.
 * 6. The vocabulary costs too much time!
 */
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "Frame.h"
#include "Settings.h"
#include "Extractors/HFNetTFModelV2.h"
#include "Extractors/HFextractor.h"
#include "utility_common.h"
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

    // Eigen::MatrixXf des1, des2;
    // cv::cv2eigen(descriptors1, des1);
    // cv::cv2eigen(descriptors2, des2);
    assert(descriptors1.isContinuous() && descriptors2.isContinuous());
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> des1(descriptors1.ptr<float>(), descriptors1.rows, descriptors1.cols);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> des2(descriptors2.ptr<float>(), descriptors2.rows, descriptors2.cols);
    // cv::Mat distanceCV = 2 * (1 - descriptors1 * descriptors2.t());
    Eigen::MatrixXf distance = 2 * (Eigen::MatrixXf::Ones(descriptors1.rows, descriptors2.rows) - des1 * des2.transpose());

    for(int idx1=0; idx1 < distance.rows(); idx1++)
    {

        float bestDist1 = std::numeric_limits<float>::max();
        int bestIdx2 = -1;
        float bestDist2 = std::numeric_limits<float>::max();

        for(int idx2=0; idx2<distance.cols(); idx2++)
        {
            float dist =distance(idx1, idx2);

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

// const string strDatasetPath("/media/llm/Datasets/EuRoC/MH_01_easy/mav0/cam0/data/");
// const string strSettingsPath("Examples/Monocular-Inertial/EuRoC.yaml");
// const int dbStart = 420;
// const int dbEnd = 50;

const string strDatasetPath("/media/llm/Datasets/TUM-VI/dataset-outdoors5_512_16/mav0/cam0/data");
const string strSettingsPath("Examples/Monocular-Inertial/TUM-VI.yaml");
const int dbStart = 50;
const int dbEnd = 50;

int main(int argc, char* argv[])
{
    // By default, the Eigen will use the maximum number of threads in OpenMP.
    // However, this will somehow slow down the calculation of dense matrix multiplication.
    // Therefore, use only half of the thresds.
    Eigen::setNbThreads(std::max(Eigen::nbThreads() / 2, 1));
    
    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    settings = new Settings(strSettingsPath, 0);
    HFNetTFModelV2 *pModel = new HFNetTFModelV2(settings->strTFModelPath(), kImageToLocalAndGlobal, {1, settings->newImSize().height, settings->newImSize().width, 1});

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(dbStart, files.size() - dbEnd);

    HFextractor extractorHF(settings->nFeatures(),settings->threshold(),pModel);

    char command = ' ';
    float threshold = 10;
    bool showUndistort = false;
    int select = 0;
    auto cameraMatrix = settings->camera1()->toK();
    cv::Mat distCoef;
    if(settings->needToUndistort()) distCoef = settings->camera1DistortionCoef();
    else distCoef = cv::Mat::zeros(4,1,CV_32F);
    do {
        if (command == 'u') showUndistort = !showUndistort;
        else if (command == 'w') select += 1;
        else if (command == 's') select -= 1;
        else if (command == 'a') threshold -= 0.5;
        else if (command == 'd') threshold += 0.5;
        else select = distribution(generator);

        cout << command << endl;
        cout << select << endl;
        cout << threshold << endl;
        cv::Mat imageRaw1 = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        cv::Mat imageRaw2 = imread(strDatasetPath + files[select + 10], IMREAD_GRAYSCALE);


        std::vector<cv::KeyPoint> keypointsHF1, keypointsHF2;
        cv::Mat descriptorsHF1, descriptorsHF2;
        cv::Mat globalDescriptorsHF;
        extractorHF(imageRaw1, keypointsHF1, descriptorsHF1, globalDescriptorsHF);
        extractorHF(imageRaw2, keypointsHF2, descriptorsHF2, globalDescriptorsHF);


        cv::Mat image1, image2;
        if (showUndistort && settings->needToUndistort())
        {
            cv::undistort(imageRaw1, image1, cameraMatrix, distCoef);
            cv::undistort(imageRaw2, image2, cameraMatrix, distCoef);
            keypointsHF1 = undistortPoints(keypointsHF1, cameraMatrix, distCoef);
            keypointsHF2 = undistortPoints(keypointsHF2, cameraMatrix, distCoef);
        }
        else
        {
            image1 = imageRaw1, image2 = imageRaw2;
        }

        cout << "-------------------------------------------------------" << endl;
        { // good threshold 0.4~0.55
            std::vector<cv::DMatch> matchesHF, thresholdMatchesHF, inlierMatchesHF, wrongMatchesHF;
            auto t1 = chrono::steady_clock::now();
            cv::BFMatcher cvMatcherHF(cv::NORM_L2, true);
            cvMatcherHF.match(descriptorsHF1, descriptorsHF2, matchesHF);
            auto t2 = chrono::steady_clock::now();
            auto timeCost = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            for (auto& match : matchesHF)
            {
                if (match.distance > threshold * 0.1) continue;
                thresholdMatchesHF.emplace_back(match);
            }
            // {
            //     cv::DMatch match = matchesHF[0];
            //     cv::Mat d1 = descriptorsHF1.row(match.queryIdx);
            //     cv::Mat d2 = descriptorsHF2.row(match.trainIdx);
            //     cout << "distance by BFMatcher: " << match.distance << endl;
            //     cout << "distance by cv::norm: " << cv::norm(d1 - d2, cv::NORM_L2) << endl;
            // }
            cv::Mat E = FindCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, thresholdMatchesHF, cameraMatrix, inlierMatchesHF, wrongMatchesHF);
            cv::Mat plotHF = ShowCorrectMatches(image1, image2, keypointsHF1, keypointsHF2, inlierMatchesHF, wrongMatchesHF);
            cv::imshow("HF + BFMatcher_L2", plotHF);
            cout << "HF + BFMatcher_L2:" << endl;
            cout << "match costs time: " << timeCost << "ms" << endl;
            cout << "matches total number: " << matchesHF.size() << endl;
            cout << "threshold matches total number: " << thresholdMatchesHF.size() << endl;
            cout << "correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesHF.size() / thresholdMatchesHF.size() << endl;
        }
        { // good threshold 5 ~ 7
            std::vector<cv::DMatch> matchesHF, thresholdMatchesHF, inlierMatchesHF, wrongMatchesHF;
            auto t1 = chrono::steady_clock::now();
            cv::BFMatcher cvMatcherHF(cv::NORM_L1, true);
            cvMatcherHF.match(descriptorsHF1, descriptorsHF2, matchesHF);
            auto t2 = chrono::steady_clock::now();
            auto timeCost = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            thresholdMatchesHF.clear();
            for (auto& match : matchesHF)
            {
                if (match.distance > threshold) continue;
                thresholdMatchesHF.emplace_back(match);
            }
            cv::Mat E = FindCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, thresholdMatchesHF, cameraMatrix, inlierMatchesHF, wrongMatchesHF);
            cv::Mat plotHF = ShowCorrectMatches(image1, image2, keypointsHF1, keypointsHF2, inlierMatchesHF, wrongMatchesHF);
            cv::imshow("HF + BFMatcher_L1", plotHF);
            cout << "HF + BFMatcher_L1:" << endl;
            cout << "match costs time: " << timeCost << "ms" << endl;
            cout << "matches total number: " << matchesHF.size() << endl;
            cout << "threshold matches total number: " << thresholdMatchesHF.size() << endl;
            cout << "correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesHF.size() / thresholdMatchesHF.size() << endl;
        }
        { // The speed is faster than BFMather, but rhe correct percentage is lower
            std::vector<cv::DMatch> matchesHF, thresholdMatchesHF, inlierMatchesHF, wrongMatchesHF;
            auto t1 = chrono::steady_clock::now();
            SearchByBoWV2(1, 15, descriptorsHF1, descriptorsHF2, matchesHF);
            auto t2 = chrono::steady_clock::now();
            auto timeCost = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            thresholdMatchesHF.clear();
            for (auto& match : matchesHF)
            {
                if (match.distance > (threshold * 0.1)*(threshold * 0.1)) continue;
                thresholdMatchesHF.emplace_back(match);
            }
            cv::Mat E = FindCorrectMatchesByEssentialMat(keypointsHF1, keypointsHF2, thresholdMatchesHF, cameraMatrix, inlierMatchesHF, wrongMatchesHF);
            cv::Mat plotHF = ShowCorrectMatches(image1, image2, keypointsHF1, keypointsHF2, inlierMatchesHF, wrongMatchesHF);
            cv::imshow("HF + SearchByBoWV2", plotHF);
            cout << "HF + SearchByBoWV2:" << endl;
            cout << "match costs time: " << timeCost << "ms" << endl;
            cout << "matches total number: " << matchesHF.size() << endl;
            cout << "threshold matches total number: " << thresholdMatchesHF.size() << endl;
            cout << "correct matches number: " << inlierMatchesHF.size() << endl;
            cout << "match correct percentage: " << (float)inlierMatchesHF.size() / thresholdMatchesHF.size() << endl;
        }
    } while ((command = cv::waitKey()) != 'q');

    return 0;
}
