#ifndef BASEEXTRACTOR_H
#define BASEEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM3
{

enum ExtractorType {
    kExtractorORB,
    kExtractorHF
};

class BaseExtractor
{
public:

    BaseExtractor(){};
    ~BaseExtractor(){};

    virtual int operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea) = 0;

    int inline GetLevels(void) {
        return nlevels;}

    float inline GetScaleFactor(void) {
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(void) {
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(void) {
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(void) {
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(void) {
        return mvInvLevelSigma2;
    }

    std::vector<cv::Mat> mvImagePyramid;

protected:

    double scaleFactor;
    int nlevels;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};

} //namespace ORB_SLAM

# endif