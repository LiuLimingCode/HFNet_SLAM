#ifndef BASEMODEL_H
#define BASEMODEL_H

#include "Settings.h"

namespace ORB_SLAM3
{

class BaseModel
{
public:
    virtual ~BaseModel(void) = default;
    
    virtual bool Detect(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
                        int nKeypointsNum, float threshold, int nRadius) = 0;

    virtual bool DetectOnlyLocal(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors,
                                 int nKeypointsNum, float threshold, int nRadius) = 0; 

    virtual bool IsValid(void) = 0;
};

std::vector<BaseModel*> InitModelsVec(Settings* settings);

std::vector<BaseModel*> GetModelVec(void);

BaseModel* InitModel(Settings *settings);

}

#endif