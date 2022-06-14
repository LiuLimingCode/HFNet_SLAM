#ifndef BaseModel_H
#define BaseModel_H

namespace ORB_SLAM3
{

class BaseModel
{
public:
    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                        int nKeypointsNum, int threshold, int nRadius) = 0;

    virtual bool IsValid(void) = 0;
};

}

#endif