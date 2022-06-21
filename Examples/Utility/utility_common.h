#ifndef UTILITY_COMMON_H
#define UTILITY_COMMON_H

#include <chrono>
#include <fstream>
#include <dirent.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

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

void FindCorrectMatches(const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, 
                        const std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &inlierMatches, std::vector<cv::DMatch> &wrongMatches)
{
    if (matches.size() < 10) 
    {
        wrongMatches = matches;
        inlierMatches.clear();
        return;
    }
    vector<cv::Point2f> vPt1, vPt2;
    for (const auto &match : matches)
    {
        vPt1.emplace_back(keypoints1[match.queryIdx].pt);
        vPt2.emplace_back(keypoints2[match.trainIdx].pt);
    }

    cv::Mat homography;
    std::vector<int> inliers;
    cv::findHomography(vPt1, vPt2, cv::RANSAC, 10.0, inliers);

    inlierMatches.clear();
    wrongMatches.clear();
    inlierMatches.reserve(matches.size());
    wrongMatches.reserve(matches.size());
    for (size_t index = 0; index < matches.size(); ++index)
    {
        if (inliers[index]) inlierMatches.emplace_back(matches[index]);
        else wrongMatches.emplace_back(matches[index]);
    }
}

cv::Mat ShowCorrectMatches(const cv::Mat &image1, const cv::Mat &image2,
                           const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, 
                           const std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &inlierMatches, std::vector<cv::DMatch> &wrongMatches)
{
    cv::Mat outImage;
    FindCorrectMatches(keypoints1, keypoints2, matches, inlierMatches, wrongMatches);
    cv::drawMatches(image1, keypoints1, image2, keypoints2, wrongMatches, outImage, cv::Scalar(0, 0, 255), cv::Scalar(-1, -1, -1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::drawMatches(image1, keypoints1, image2, keypoints2, inlierMatches, outImage, cv::Scalar(0, 255, 0), cv::Scalar(-1, -1, -1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
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

#endif