#ifndef BASEEXTRACTOR_H
#define BASEEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM3
{

enum ExtractorType {
    kExtractorHFNetTF,
};

class BaseExtractor
{
public:

    BaseExtractor(){};
    ~BaseExtractor(){};

    virtual int operator()(const cv::Mat &_image, std::vector<cv::KeyPoint>& _keypoints,
                           cv::Mat &_localDescriptors, cv::Mat &_globalDescriptors) = 0;

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
    bool bUseOctTree;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;    
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    std::vector<int> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int minX,
                                       const int maxX, const int minY, const int maxY, const int N);
};

} //namespace ORB_SLAM

# endif