#ifndef UTILITY_COMMON_H
#define UTILITY_COMMON_H

#include <chrono>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace cv;
using namespace std;

struct TicToc
{
    TicToc() {};
    void clearBuff() { timeBuff.clear(); }
    void Tic() { t1 = chrono::steady_clock::now(); }
    float Toc()
    {
        t2 = chrono::steady_clock::now();
        float time = chrono::duration<float, std::milli>(t2 - t1).count();
        timeBuff.emplace_back(time);
        return time;
    }
    float aveCost(void)
    { 
        if (timeBuff.empty()) return 0;
        return std::accumulate(timeBuff.begin(), timeBuff.end(), 0.f) / (float)timeBuff.size();
    }
    float devCost(void)
    {
        if (timeBuff.size() <= 1) return 0;
        float average = aveCost();

        float accum = 0;
        int total = 0;
        for(double value : timeBuff)
        {
            if(value == 0)
                continue;
            accum += pow(value - average, 2);
            total++;
        }
        return sqrt(accum / total);
    }
    bool empty() {return timeBuff.empty();}


    std::vector<float> timeBuff;
    chrono::steady_clock::time_point t1;
    chrono::steady_clock::time_point t2;
};

int FilenameFilter(const struct dirent *cur)
{
    std::string str(cur->d_name);
    if(str.find(".png") != std::string::npos){
        return 1;
    }
    return 0;
}

vector<string> GetPngFiles(string png_dir)
{
   struct dirent **namelist;
   std::vector<std::string> ret;
   int n = scandir(png_dir.c_str(), &namelist, FilenameFilter, alphasort);

   if(n < 0){
       return ret;
   }

   for (int i = 0; i < n; i++){
       std::string filepath(namelist[i]->d_name);
       ret.push_back("/" + filepath);
   }

   free(namelist);
   return ret;
}

void ShowKeypoints(const string &title, Mat image, const std::vector<KeyPoint> &keypoints)
{
    Mat image_show;
    cvtColor(image, image_show, COLOR_GRAY2BGR);

    for(const KeyPoint &kp : keypoints){
        cv::circle(image_show, kp.pt, 2, Scalar(0, 255, 0), -1);
    }

    cv::imshow(title.c_str(), image_show);
}

cv::Mat FindCorrectMatchesByHomography
    (const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, 
    const std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &inlierMatches, std::vector<cv::DMatch> &wrongMatches)
{
    if (matches.size() < 10) 
    {
        wrongMatches = matches;
        inlierMatches.clear();
        return cv::Mat();
    }
    vector<cv::Point2f> vPt1, vPt2;
    for (const auto &match : matches)
    {
        vPt1.emplace_back(keypoints1[match.queryIdx].pt);
        vPt2.emplace_back(keypoints2[match.trainIdx].pt);
    }

    cv::Mat homography;
    std::vector<int> inliers;
    homography = cv::findHomography(vPt1, vPt2, cv::RANSAC, 10.0, inliers);

    inlierMatches.clear();
    wrongMatches.clear();
    inlierMatches.reserve(matches.size());
    wrongMatches.reserve(matches.size());
    for (size_t index = 0; index < matches.size(); ++index)
    {
        if (inliers[index]) inlierMatches.emplace_back(matches[index]);
        else wrongMatches.emplace_back(matches[index]);
    }

    return homography;
}

cv::Mat FindCorrectMatchesByEssentialMat
    (const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, const std::vector<cv::DMatch> &matches,
    const cv::Mat &cameraMatrix, std::vector<cv::DMatch> &inlierMatches, std::vector<cv::DMatch> &wrongMatches)
{
    if (matches.size() < 10) 
    {
        wrongMatches = matches;
        inlierMatches.clear();
        return cv::Mat();
    }
    vector<cv::Point2f> vPt1, vPt2;
    for (const auto &match : matches)
    {
        vPt1.emplace_back(keypoints1[match.queryIdx].pt);
        vPt2.emplace_back(keypoints2[match.trainIdx].pt);
    }

    cv::Mat inliers;
    cv::Mat E = findEssentialMat(vPt1, vPt2, cameraMatrix, cv::RANSAC, 0.999, 3.0, inliers);
    inlierMatches.clear();
    wrongMatches.clear();
    inlierMatches.reserve(matches.size());
    wrongMatches.reserve(matches.size());
    for (size_t index = 0; index < matches.size(); ++index)
    {
        if (inliers.at<uchar>(index)) inlierMatches.emplace_back(matches[index]);
        else wrongMatches.emplace_back(matches[index]);
    }

    return E;
}

cv::Mat FindCorrectMatchesByPnP
    (const std::vector<cv::KeyPoint> &keypoints1, const cv::Mat &depthImage1, const std::vector<cv::KeyPoint> &keypoints2,
    const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs,
    std::vector<cv::DMatch> &matches,
    std::vector<cv::DMatch> &inlierMatches, std::vector<cv::DMatch> &wrongMatches)
{
    vector<cv::Point3f> vPt1;
    vector<cv::Point2f> vPt2;
    std::vector<cv::DMatch> matchesTemp;
    for (const auto &match : matches)
    {
        auto kpt1 = keypoints1[match.queryIdx].pt;
        float depth1 = depthImage1.ptr<unsigned short>(int(kpt1.y))[int(kpt1.x)] / 5000.0;
        if (depth1 == 0) continue;

        float ptX = (kpt1.x - cameraMatrix.at<double>(0, 2)) / cameraMatrix.at<double>(0, 0);
        float ptY = (kpt1.y - cameraMatrix.at<double>(1, 2)) / cameraMatrix.at<double>(1, 1);
        vPt1.push_back(cv::Point3f(ptX * depth1, ptY * depth1, depth1));

        vPt2.emplace_back(keypoints2[match.trainIdx].pt);
        matchesTemp.emplace_back(match);
    }

    matches = matchesTemp;
    if (vPt1.size() < 10) 
    {
        wrongMatches = matches;
        inlierMatches.clear();
        return cv::Mat();
    }

    cv::Mat rvec, tvec;
    cv::Mat inliers;
    solvePnPRansac(vPt1, vPt2, cameraMatrix, distCoeffs, rvec, tvec, false, 100, 8.0f, 0.99, inliers);
    inlierMatches.clear();
    wrongMatches.clear();
    inlierMatches.reserve(matchesTemp.size());
    wrongMatches.reserve(matchesTemp.size());
    int inlier = 0, index = 0;
    while (index < matchesTemp.size())
    {
        if (inlier < inliers.rows && index == inliers.at<int>(inlier))
        {
            inlierMatches.emplace_back(matchesTemp[index++]);
            inlier++;
        }
        else
        {
            wrongMatches.emplace_back(matchesTemp[index++]);
        }
    }
    return cv::Mat();
}

cv::Mat ShowCorrectMatches(const cv::Mat &image1, const cv::Mat &image2,
                           const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, 
                           const std::vector<cv::DMatch> &inlierMatches, const std::vector<cv::DMatch> &wrongMatches, bool showSinglePoints = false)
{
    cv::Mat outImage;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, wrongMatches, outImage, cv::Scalar(0, 0, 255), cv::Scalar(-1, -1, -1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::drawMatches(image1, keypoints1, image2, keypoints2, inlierMatches, outImage, cv::Scalar(0, 255, 0), cv::Scalar(-1, -1, -1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    if (showSinglePoints)
    {
        std::vector<bool> matched1(keypoints1.size(), true);
        std::vector<bool> matched2(keypoints2.size(), true);
        for (auto m : inlierMatches)
        {
            matched1[m.queryIdx] = false;
            matched2[m.trainIdx] = false;
        }
        for (auto m : wrongMatches)
        {
            matched1[m.queryIdx] = false;
            matched2[m.trainIdx] = false;
        }
        std::vector<cv::KeyPoint> singlePoint1, singlePoint2;
        for (size_t index = 0; index < matched1.size(); ++index)
        {
            if (matched1[index])
            {
                singlePoint1.emplace_back(keypoints1[index]);
            }
        }
        for (size_t index = 0; index < matched2.size(); ++index)
        {
            if (matched2[index])
            {
                singlePoint2.emplace_back(keypoints2[index]);
            }
        }
        cv::drawMatches(image1, singlePoint1, image2, singlePoint2, std::vector<cv::DMatch>(), outImage, cv::Scalar(0, 255, 0), cv::Scalar(-1, -1, -1), std::vector<char>(), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    }
    return outImage;
}

std::vector<cv::KeyPoint> undistortPoints(const std::vector<cv::KeyPoint> &src, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs)
{
    std::vector<cv::Point2f> mat;
    std::vector<cv::KeyPoint> res;
    for (auto temp : src)
    {
        mat.emplace_back(temp.pt);
    }
    cv::undistortPoints(mat, mat, cameraMatrix, distCoeffs, cameraMatrix);
    for (int index = 0; index < mat.size(); ++index) 
    {
        auto kpt = src[index];
        kpt.pt = mat[index];
        res.emplace_back(kpt);
    }
    return res;
}


std::vector<std::tuple<unsigned long, std::string, Eigen::Isometry3f>> ReadEuRoCDataset(const std::string& strDatasetPath)
{
    std::vector<std::tuple<unsigned long, std::string, Eigen::Isometry3f>> res;

    // read ground truth data
    std::map<unsigned long, Eigen::Isometry3f> sGroundTruth;
    {
        const std::string strFileGT = strDatasetPath + "/state_groundtruth_estimate0/data.csv";
        
        FILE *f = fopen(strFileGT.c_str(), "r");
        if (f == nullptr) {
            cout << "[ERROR]: can't load pose data; wrong path: " << strFileGT.c_str() << endl;
            return res;
        }

        char title[500];
        if (fgets(title, 500, f) == nullptr)
        {
            cout << "[ERROR]: can't load pose data; no data available";
            return res;
        }

        while(!feof(f)) {
            char line[300];
            fgets(line, 300, f);
            for (int index = 0; index < strlen(line); ++index) {
                if (line[index] == ' ') line[index] = ',';
            }
            
            unsigned long time;
            float px, py, pz;
            float qw, qx, qy, qz;

            auto result = sscanf(line, "%lu,%f,%f,%f,%f,%f,%f,%f\n", 
                &time,
                &px, &py, &pz,
                &qx, &qy, &qz, &qw);

            time = time / 100000 * 100000;
            Eigen::Vector3f pose;
            Eigen::Quaternionf rotation;
            pose << px, py, pz;
            rotation.coeffs() << qx, qy, qz, qw;

            sGroundTruth.insert(make_pair(time, Eigen::Isometry3f::Identity()));
            sGroundTruth[time].rotate(rotation);
            sGroundTruth[time].pretranslate(pose);
        }

        fclose(f);
    }

    // read image data
    std::map<unsigned long, std::string> sImagePath;
    {
        const std::string strFileData = strDatasetPath + "/cam0/data.csv";
        
        FILE *f = fopen(strFileData.c_str(), "r");
        if (f == nullptr) {
            cout << "[ERROR]: cannot find " << strFileData << endl;
            return res;
        }

        char title[500];
        if (fgets(title, 500, f) == nullptr)
        {
            cout << "[ERROR]: can't load timestamp data";
            return res;
        }

        while(!feof(f)) {
            char line[300];
            fgets(line, 300, f);
            for (int index = 0; index < strlen(line); ++index) {
                if (line[index] == ' ') line[index] = ',';
            }
            
            unsigned long time;
            char file[100];

            auto result = sscanf(line, "%lu,%s\n", 
                &time, file);
            time = time / 100000 * 100000;
            sImagePath[time] = strDatasetPath + "/cam0/data/" + file;
        }

        fclose(f);
    }
    
    {
        for (auto image : sImagePath) {
            if (!sGroundTruth.count(image.first)) {
                cout << "[WARNING]: failed to match timestamp: " << image.first << endl;
                continue;
            }
            res.push_back(make_tuple(image.first, sImagePath[image.first], sGroundTruth[image.first]));
        }
        std::sort(res.begin(), res.end(), [](const std::tuple<unsigned long, std::string, Eigen::Isometry3f>& d1,
                                             const std::tuple<unsigned long, std::string, Eigen::Isometry3f>& d2) {
            return std::get<0>(d1) < std::get<0>(d2);
        });
    }

    return res;
}


#endif