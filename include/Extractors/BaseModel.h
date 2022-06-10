#ifndef BaseModel_H
#define BaseModel_H

namespace ORB_SLAM3
{

class BaseModel
{
public:
    typedef std::shared_ptr<BaseModel> Ptr;

    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &descriptors,
                        int nKeypointsNum = 1000, int nRadius = 4) = 0;
};

}

#endif