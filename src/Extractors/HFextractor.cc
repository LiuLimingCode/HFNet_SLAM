
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

HFextractor::HFextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST) {

}

int HFextractor::operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea) {

    return 0;
}

} //namespace ORB_SLAM3