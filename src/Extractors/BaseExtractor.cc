#include "Extractors/BaseExtractor.h"

using namespace std;

namespace ORB_SLAM3
{

class ExtractorNode
{
public:
    ExtractorNode() : bNoMore(false) {}

    void DivideNode(const std::vector<cv::KeyPoint>& vPts, ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<int> vKeyIdxs;
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

void ExtractorNode::DivideNode(const std::vector<cv::KeyPoint>& vPts, ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    //Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeyIdxs.reserve(vKeyIdxs.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeyIdxs.reserve(vKeyIdxs.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeyIdxs.reserve(vKeyIdxs.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeyIdxs.reserve(vKeyIdxs.size());

    //Associate points to childs
    for(size_t i=0;i<vKeyIdxs.size();i++)
    {
        const cv::KeyPoint &kp = vPts[vKeyIdxs[i]];
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeyIdxs.push_back(vKeyIdxs[i]);
            else
                n3.vKeyIdxs.push_back(vKeyIdxs[i]);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeyIdxs.push_back(vKeyIdxs[i]);
        else
            n4.vKeyIdxs.push_back(vKeyIdxs[i]);
    }

    if(n1.vKeyIdxs.size()==1)
        n1.bNoMore = true;
    if(n2.vKeyIdxs.size()==1)
        n2.bNoMore = true;
    if(n3.vKeyIdxs.size()==1)
        n3.bNoMore = true;
    if(n4.vKeyIdxs.size()==1)
        n4.bNoMore = true;

}

std::vector<int> BaseExtractor::DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int minX,
                                                  const int maxX, const int minY, const int maxY, const int N)
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
        ni.vKeyIdxs.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    //Associate points to childs
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        vpIniNodes[kp.pt.x/hX]->vKeyIdxs.push_back(i);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();

    while(lit!=lNodes.end())
    {
        if(lit->vKeyIdxs.size()==1)
        {
            lit->bNoMore=true;
            lit++;
        }
        else if(lit->vKeyIdxs.empty())
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
                lit->DivideNode(vToDistributeKeys,n1,n2,n3,n4);

                // Add childs if they contain points
                if(n1.vKeyIdxs.size()>0)
                {
                    lNodes.push_front(n1);
                    if(n1.vKeyIdxs.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeyIdxs.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeyIdxs.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeyIdxs.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeyIdxs.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeyIdxs.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeyIdxs.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeyIdxs.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeyIdxs.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeyIdxs.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeyIdxs.size(),&lNodes.front()));
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
                    vPrevSizeAndPointerToNode[j].second->DivideNode(vToDistributeKeys,n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeyIdxs.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeyIdxs.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeyIdxs.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeyIdxs.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeyIdxs.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeyIdxs.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeyIdxs.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeyIdxs.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeyIdxs.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeyIdxs.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeyIdxs.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeyIdxs.size(),&lNodes.front()));
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
    vector<int> vResultKeys;
    vResultKeys.reserve(N);
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        vector<int> &vNodeKeyIdxs = lit->vKeyIdxs;
        int maxIdx = vNodeKeyIdxs[0];
        float maxResponse = vToDistributeKeys[maxIdx].response;

        for(size_t k=1;k<vNodeKeyIdxs.size();k++)
        {
            if(vToDistributeKeys[vNodeKeyIdxs[k]].response>maxResponse)
            {
                maxIdx = vNodeKeyIdxs[k];
                maxResponse = vToDistributeKeys[maxIdx].response;
            }
        }

        vResultKeys.push_back(maxIdx);
    }

    return vResultKeys;
}

} //namespace ORB_SLAM3