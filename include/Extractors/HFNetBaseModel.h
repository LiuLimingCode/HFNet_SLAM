#ifndef HFNETBASEMODEL_H
#define HFNETBASEMODEL_H

namespace ORB_SLAM3
{

class HFNetBaseModel
{
public:
    typedef std::shared_ptr<HFNetBaseModel> Ptr;

    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &descriptors,
                        int nKeypointsNum = 1000, int nRadius = 4) = 0;
};

}

#endif