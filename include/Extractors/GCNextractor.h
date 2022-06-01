#ifndef GCNEXTRACTOR_H
#define GCNEXTRACTOR_H

#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

// Compile will fail for opimizier since pytorch defined this
#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM3
{

class GCNextractor : public BaseExtractor
{
public:

    enum {HARRIS_SCORE=0, FAST_SCORE=1 };

    GCNextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~GCNextractor(){}

    // Compute the GCN features and descriptors on an image.
    // GCN are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    int operator()( cv::InputArray image, cv::InputArray mask,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::OutputArray descriptors, std::vector<int> &vLappingArea);

protected:

    int nfeatures;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;

    std::vector<int> umax;

    std::shared_ptr<torch::jit::script::Module> module;
};

} //namespace ORB_SLAM

#endif