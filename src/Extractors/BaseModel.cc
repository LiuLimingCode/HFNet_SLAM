#include "Extractors/BaseModel.h"
#include "Extractors/HFNetTFModel.h"

using namespace std;

namespace ORB_SLAM3
{

std::vector<BaseModel*> gvpModels;

std::vector<BaseModel*> InitModelsVec(Settings* settings)
{
    if (gvpModels.size())
    {
        for (auto pModel : gvpModels) delete pModel;
        gvpModels.clear();
    }

    int nLevels = settings->nLevels();
    cv::Size ImSize = settings->newImSize();
    float scaleFactor = settings->scaleFactor();
    gvpModels.reserve(nLevels);
    if (settings->extractorType() == kExtractorHFNetTF)
    {
        HFNetTFModel* pModel = new HFNetTFModel(settings->strResamplerPath(), settings->strModelPath());
        pModel->WarmUp(ImSize, false);
        if (pModel->IsValid())
        {
            cout << "Successfully loaded HFNetTF model" << endl;
            gvpModels.emplace_back(pModel);

            float scale = 1.0;
            for (int level = 1; level < nLevels; ++level)
            {
                scale /= scaleFactor;
                pModel = pModel->clone();
                pModel->WarmUp(cv::Size(cvRound(ImSize.width * scale), cvRound(ImSize.height * scale)), true);
                gvpModels.emplace_back(pModel);
            }
        }
        else
        {
            exit(-1);
        }
    }
    else
    {
        cerr << "Wrong type of detector!" << endl;
        exit(-1);
    }

    return gvpModels;
}

std::vector<BaseModel*> GetModelVec(void)
{
    if (gvpModels.empty())
    {
        cerr << "Try to get models before initialize them" << endl;
        exit(-1);
    }
    return gvpModels;
}

BaseModel* InitModel(Settings *settings)
{
    cv::Size ImSize = settings->newImSize();
    BaseModel* pModel;
    if (settings->extractorType() == kExtractorHFNetTF)
    {
        pModel = new HFNetTFModel(settings->strResamplerPath(), settings->strModelPath());
        if (pModel->IsValid())
        {
            cout << "Successfully loaded HFNetTF model" << endl;
        }
        else exit(-1);
    }
    else
    {
        cerr << "Wrong type of detector!" << endl;
        exit(-1);
    }

    return pModel;
}

}