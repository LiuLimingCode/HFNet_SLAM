
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

#include "Extractors/HFextractor.h"


using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

HFextractor::HFextractor(int _nfeatures, int _nNMSRadius, float _threshold,
                        float _scaleFactor, int _nlevels, BaseModel* _model, bool _bUseOctTree):
        nfeatures(_nfeatures), nNMSRadius(_nNMSRadius), threshold(_threshold), model(_model)
{
    scaleFactor = _scaleFactor;
    nlevels = _nlevels;
    bUseOctTree = _bUseOctTree;
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    //This is for orientation
    // pre-compute the end of a row in a circular patch
    umax.resize(HALF_PATCH_SIZE + 1);

    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
}


int HFextractor::operator() (const cv::Mat &_image, std::vector<cv::KeyPoint>& _keypoints,
                             cv::Mat &_localDescriptors, cv::Mat &_globalDescriptors) {
    if(_image.empty())
        return -1;

    assert(_image.type() == CV_8UC1 );

    if (bUseOctTree)
    {
        std::vector<cv::KeyPoint> vKeypoints;
        cv::Mat lDescriptor;
        model->Detect(_image, vKeypoints, lDescriptor, _globalDescriptors, nfeatures * 10, threshold, nNMSRadius);
        auto remainIdxs = DistributeOctTree(vKeypoints, 0, _image.cols, 0, _image.rows, nfeatures);
        _keypoints.clear();
        _keypoints.reserve(remainIdxs.size());
        _localDescriptors = cv::Mat(remainIdxs.size(), 256, CV_32F);
        for (size_t index = 0; index < remainIdxs.size(); ++index)
        {
            _keypoints.emplace_back(vKeypoints[remainIdxs[index]]);
            lDescriptor.row(remainIdxs[index]).copyTo(_localDescriptors.row(index));
        }
    }
    else
    {
        model->Detect(_image, _keypoints, _localDescriptors, _globalDescriptors, nfeatures, threshold, nNMSRadius);
    }
    return _keypoints.size();
}


} //namespace ORB_SLAM3