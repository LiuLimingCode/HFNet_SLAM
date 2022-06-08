#ifndef HFNETEXTRACTOR_H
#define HFNETEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
#include "Extractors/BaseExtractor.h"
#include "Extractors/HFNetTFModel.h"
#include "Extractors/HFNetVINOModel.h"

namespace ORB_SLAM3
{

class HFextractor : public BaseExtractor
{
public:
    
    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    HFextractor(int nfeatures, float scaleFactor, int nlevels,
                HFNetBaseModel::Ptr model);

    ~HFextractor(){}

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    int operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea) override;

protected:

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                           const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

    void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);

    int nfeatures;
    HFNetBaseModel::Ptr model;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;
};

} //namespace ORB_SLAM

#endif
