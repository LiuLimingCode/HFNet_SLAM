/**
 * To Test the performance of different dector
 * 
 * What we found in this test:
 * 1. HFNet do not need OctTree as it has NMS
 * 2. It is not necessary to calculate the response of keypoints, because it is only used in OctTree
 */
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>

#include "Settings.h"
#include "Extractors/HFNetTFModel.h"
#include "Examples/Utility/utility_common.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;

Settings *settings;
const int EDGE_THRESHOLD = 19;

class ExtractorNode
{
public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
    {
        const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
        const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

        //Define boundaries of childs
        n1.UL = UL;
        n1.UR = cv::Point2i(UL.x+halfX,UL.y);
        n1.BL = cv::Point2i(UL.x,UL.y+halfY);
        n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
        n1.vKeys.reserve(vKeys.size());

        n2.UL = n1.UR;
        n2.UR = UR;
        n2.BL = n1.BR;
        n2.BR = cv::Point2i(UR.x,UL.y+halfY);
        n2.vKeys.reserve(vKeys.size());

        n3.UL = n1.BL;
        n3.UR = n1.BR;
        n3.BL = BL;
        n3.BR = cv::Point2i(n1.BR.x,BL.y);
        n3.vKeys.reserve(vKeys.size());

        n4.UL = n3.UR;
        n4.UR = n2.BR;
        n4.BL = n3.BR;
        n4.BR = BR;
        n4.vKeys.reserve(vKeys.size());

        //Associate points to childs
        for(size_t i=0;i<vKeys.size();i++)
        {
            const cv::KeyPoint &kp = vKeys[i];
            if(kp.pt.x<n1.UR.x)
            {
                if(kp.pt.y<n1.BR.y)
                    n1.vKeys.push_back(kp);
                else
                    n3.vKeys.push_back(kp);
            }
            else if(kp.pt.y<n1.BR.y)
                n2.vKeys.push_back(kp);
            else
                n4.vKeys.push_back(kp);
        }

        if(n1.vKeys.size()==1)
            n1.bNoMore = true;
        if(n2.vKeys.size()==1)
            n2.bNoMore = true;
        if(n3.vKeys.size()==1)
            n3.bNoMore = true;
        if(n4.vKeys.size()==1)
            n4.bNoMore = true;

    }

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

static bool compareNodes(pair<int,ExtractorNode*>& e1, pair<int,ExtractorNode*>& e2)
{
    if(e1.first < e2.first){
        return true;
    }
    else if(e1.first > e2.first){
        return false;
    }
    else{
        if(e1.second->UL.x < e2.second->UL.x){
            return true;
        }
        else{
            return false;
        }
    }
}

vector<cv::KeyPoint> DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                       const int &maxX, const int &minY, const int &maxY, const int &N)
{
    // Compute how many initial nodes
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

    const float hX = static_cast<float>(maxX-minX)/nIni;

    list<ExtractorNode> lNodes;

    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for(int i=0; i<nIni; i++)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    //Associate points to childs
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();

    while(lit!=lNodes.end())
    {
        if(lit->vKeys.size()==1)
        {
            lit->bNoMore=true;
            lit++;
        }
        else if(lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size()*4);

    while(!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        while(lit!=lNodes.end())
        {
            if(lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);

                // Add childs if they contain points
                if(n1.vKeys.size()>0)
                {
                    lNodes.push_front(n1);
                    if(n1.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                lit=lNodes.erase(lit);
                continue;
            }
        }

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }
        else if(((int)lNodes.size()+nToExpand*3)>N)
        {

            while(!bFinish)
            {

                prevSize = lNodes.size();

                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end(),compareNodes);
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if((int)lNodes.size()>=N)
                        break;
                }

                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;

            }
        }
    }

    // Retain the best point in each node
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(settings->nFeatures());
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

// Using FAST algorithm. The function is original from ORB-SLAM3
void GetFAST(cv::Mat image, vector<KeyPoint> &vKeypoints, bool bUseOctTree = false)
{
    const float W = 35;
    const int minBorderX = EDGE_THRESHOLD-3;
    const int minBorderY = minBorderX;
    const int maxBorderX = image.cols-EDGE_THRESHOLD+3;
    const int maxBorderY = image.rows-EDGE_THRESHOLD+3;

    vector<cv::KeyPoint> vToDistributeKeys;
    vToDistributeKeys.reserve(settings->nFeatures()*10);

    const float width = (maxBorderX-minBorderX);
    const float height = (maxBorderY-minBorderY);

    const int nCols = width/W;
    const int nRows = height/W;
    const int wCell = ceil(width/nCols);
    const int hCell = ceil(height/nRows);

    for(int i=0; i<nRows; i++)
    {
        const float iniY =minBorderY+i*hCell;
        float maxY = iniY+hCell+6;

        if(iniY>=maxBorderY-3)
            continue;
        if(maxY>maxBorderY)
            maxY = maxBorderY;

        for(int j=0; j<nCols; j++)
        {
            const float iniX =minBorderX+j*wCell;
            float maxX = iniX+wCell+6;
            if(iniX>=maxBorderX-6)
                continue;
            if(maxX>maxBorderX)
                maxX = maxBorderX;

            vector<cv::KeyPoint> vKeysCell;

            FAST(image.rowRange(iniY,maxY).colRange(iniX,maxX),
                vKeysCell,20,true);

            if(vKeysCell.empty())
            {
                FAST(image.rowRange(iniY,maxY).colRange(iniX,maxX),
                    vKeysCell,7,true);
            }
        
            if(!vKeysCell.empty())
            {
                for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                {
                    (*vit).pt.x+=j*wCell;
                    (*vit).pt.y+=i*hCell;
                    vToDistributeKeys.push_back(*vit);
                }
            }
        }
    }

    vKeypoints.reserve(settings->nFeatures());

    if (bUseOctTree)
    {
        vKeypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                        minBorderY, maxBorderY, settings->nFeatures());
    }
    else
    {
        vKeypoints = vToDistributeKeys;
    }

    // Add border to coordinates and scale information
    const int nkps = vKeypoints.size();
    for(int i=0; i<nkps ; i++)
    {
        vKeypoints[i].pt.x+=minBorderX;
        vKeypoints[i].pt.y+=minBorderY;
        vKeypoints[i].octave=0;
        vKeypoints[i].size = 31;
    }
}

// Using HFNet
void GetHFNet(HFNetTFModel *model, cv::Mat image, vector<KeyPoint> &vKeypoints, bool bUseOctTree, int nKeypointsNum, float threshold, int nRadius)
{
    cv::Mat descriptors, globalDescriptors;
    model->Detect(image, vKeypoints, descriptors, globalDescriptors, nKeypointsNum, threshold, nRadius);
    if (bUseOctTree)
    {
        vKeypoints = DistributeOctTree(vKeypoints, 0, image.cols,
                                       0, image.rows, settings->nFeatures());
    }
}

// const string strDatasetPath("/media/llm/Datasets/EuRoC/MH_01_easy/mav0/cam0/data/");
// const string strSettingsPath("Examples/Monocular-Inertial/EuRoC.yaml");
// const int dbStart = 420;
// const int dbEnd = 50;

const string strDatasetPath("/media/llm/Datasets/TUM-VI/dataset-corridor4_512_16/mav0/cam0/data/");
const string strSettingsPath("Examples/Monocular-Inertial/TUM-VI.yaml");
const int dbStart = 50;
const int dbEnd = 50;

int main(int argc, char* argv[])
{
    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    settings = new Settings(strSettingsPath, 0);
    HFNetTFModel *pModel = new HFNetTFModel(settings->strResamplerPath(), settings->strModelPath());

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(dbStart, files.size() - dbEnd);

    cv::Mat image;
    vector<KeyPoint> keypoints;

    cv::namedWindow("FAST");
    cv::moveWindow("FAST", 0, 0);
    cv::namedWindow("FAST & OctTree");
    cv::moveWindow("FAST & OctTree", 820, 0);
    cv::namedWindow("HFNet");
    cv::moveWindow("HFNet", 0, 540);
    cv::namedWindow("HFNet & OctTree");
    cv::moveWindow("HFNet & OctTree", 820, 540);

    char command = ' ';
    float threshold = 0;
    int nNMSRadius = 4;
    int select = 0;
    while(1)
    {
        if (command == 'q') break;
        else if (command == 'a') threshold = std::max(threshold - 0.01, 0.0);
        else if (command == 'd') threshold += 0.01;
        else if (command == 's') select = std::max(select - 1, 0);
        else if (command == 'w') select += 1;
        else if (command == 'z') nNMSRadius = std::max(nNMSRadius - 1, 0);
        else if (command == 'c') nNMSRadius += 1;
        else select = distribution(generator);
        cout << "command: " << command << endl;
        cout << "select: " << select << endl;
        cout << "threshold: " << threshold << endl;
        cout << "nNMSRadius: " << nNMSRadius << endl;

        image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);

        cout << "============= FAST  =============" << endl;
        GetFAST(image, keypoints, false);
        ShowKeypoints("FAST", image, keypoints);
        cout << "key point number: " << keypoints.size() << endl;

        cout << "============= FAST with OctTree =============" << endl;
        GetFAST(image, keypoints, true);
        ShowKeypoints("FAST & OctTree", image, keypoints);
        cout << "key point number: " << keypoints.size() << endl;

        cout << "============= HFNet =============" << endl;
        GetHFNet(pModel, image, keypoints, false, settings->nFeatures()*2, threshold, nNMSRadius);
        ShowKeypoints("HFNet", image, keypoints);
        cout << "key point number: " << keypoints.size() << endl;

        cout << "============= HFNet & OctTree =============" << endl;
        GetHFNet(pModel, image, keypoints, true, settings->nFeatures()*2, threshold, nNMSRadius);
        ShowKeypoints("HFNet & OctTree", image, keypoints);
        cout << "key point number: " << keypoints.size() << endl;

        command = cv::waitKey();
    };

    system("pause");

    return 0;
}