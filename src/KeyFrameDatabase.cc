/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "KeyFrameDatabase.h"

#include "KeyFrame.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM3
{

void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    mvDatabase.insert(pKF);
}

void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    if (!mvDatabase.count(pKF)) return;
    
    mvDatabase.erase(pKF);
}

void KeyFrameDatabase::clear()
{
    unique_lock<mutex> lock(mMutex);
    mvDatabase.clear();
}

void KeyFrameDatabase::clearMap(Map* pMap)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for(auto it = mvDatabase.begin(), it_next = mvDatabase.begin(); it != mvDatabase.end();  it = it_next)
    {
        it_next++;
        KeyFrame* pKFi = *it;
        if (pMap == pKFi->GetMap())
        {
            // Dont delete the KF because the class Map clean all the KF when it is destroyed
            mvDatabase.erase(it);
        }
    }
}

bool compFirst(const pair<float, KeyFrame*> & a, const pair<float, KeyFrame*> & b)
{
    return a.first > b.first;
}

void KeyFrameDatabase::DetectNBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nNumCandidates)
{
    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe

    set<KeyFrame*> spCandidiateKF;
    {
        unique_lock<mutex> lock(mMutex);

        float bestScore = 0;
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> queryDescriptors(pKF->mGlobalDescriptors.ptr<float>(), pKF->mGlobalDescriptors.rows, pKF->mGlobalDescriptors.cols);
        for (auto it = mvDatabase.begin(); it != mvDatabase.end(); ++it)
        {
            KeyFrame *pKFi = *it;
            // Compute the distance of global descriptors
            // Eigen is much faster than OpenCV Mat
            assert(!pKFi->mGlobalDescriptors.empty());
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> dbDescriptors(pKFi->mGlobalDescriptors.ptr<float>(), pKFi->mGlobalDescriptors.rows, pKFi->mGlobalDescriptors.cols);
            pKFi->mPlaceRecognitionScore = std::max(0.f, 1 - (queryDescriptors - dbDescriptors).norm());
            pKFi->mnPlaceRecognitionQuery = pKF->mnId;
            bestScore = max(pKFi->mPlaceRecognitionScore, bestScore);
        }

        float minScore = bestScore * 0.8f;
        for (auto it = mvDatabase.begin(); it != mvDatabase.end(); ++it)
        {
            KeyFrame *pKFi = *it;
            if (pKFi->mPlaceRecognitionScore > minScore && pKFi->mnPlaceRecognitionQuery == pKF->mnId)
                spCandidiateKF.insert(pKFi);
        }
    }

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for(auto it=spCandidiateKF.begin(), itend=spCandidiateKF.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = *it;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = pKFi->mPlaceRecognitionScore;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnPlaceRecognitionQuery!=pKF->mnId)
                continue;

            accScore+=pKF2->mPlaceRecognitionScore;
            if(pKF2->mPlaceRecognitionScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mPlaceRecognitionScore;
            }

        }
        pKFi->mPlaceRecognitionAccScore = accScore;
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    lAccScoreAndMatch.sort(compFirst);

    vpLoopCand.reserve(nNumCandidates);
    vpMergeCand.reserve(nNumCandidates);
    set<KeyFrame*> spAlreadyAddedKF;
    int i = 0;
    list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin();
    while(i < lAccScoreAndMatch.size() && (vpLoopCand.size() < nNumCandidates || vpMergeCand.size() < nNumCandidates))
    {
        KeyFrame* pKFi = it->second;
        if(pKFi->isBad())
            continue;

        if(!spAlreadyAddedKF.count(pKFi))
        {
            if(pKF->GetMap() == pKFi->GetMap() && vpLoopCand.size() < nNumCandidates)
            {
                vpLoopCand.push_back(pKFi);
            }
            else if(pKF->GetMap() != pKFi->GetMap() && vpMergeCand.size() < nNumCandidates && !pKFi->GetMap()->IsBad())
            {
                vpMergeCand.push_back(pKFi);
            }
            spAlreadyAddedKF.insert(pKFi);
        }
        i++;
        it++;
    }
}


vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F, Map* pMap)
{
    set<KeyFrame*> spCandidiateKF;
    {
        unique_lock<mutex> lock(mMutex);

        float bestScore = 0;
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> queryDescriptors(F->mGlobalDescriptors.ptr<float>(), F->mGlobalDescriptors.rows, F->mGlobalDescriptors.cols);
        for (auto it = mvDatabase.begin(); it != mvDatabase.end(); ++it)
        {
            KeyFrame *pKFi = *it;
            // Compute the distance of global descriptors
            // Eigen is much faster than OpenCV Mat
            assert(!pKFi->mGlobalDescriptors.empty());
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> dbDescriptors(pKFi->mGlobalDescriptors.ptr<float>(), pKFi->mGlobalDescriptors.rows, pKFi->mGlobalDescriptors.cols);
            pKFi->mRelocScore = std::max(0.f, 1 - (queryDescriptors - dbDescriptors).norm());
            pKFi->mnRelocQuery = F->mnId;
            bestScore = max(pKFi->mRelocScore, bestScore);
        }

        const float thresholdScore = 0.5;
        float minScore = std::max(thresholdScore, bestScore * 0.8f);
        for (auto it = mvDatabase.begin(); it != mvDatabase.end(); ++it)
        {
            KeyFrame *pKFi = *it;
            if (pKFi->mRelocScore > minScore && pKFi->mnRelocQuery == F->mnId)
                spCandidiateKF.insert(pKFi);
        }
    }

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for(auto it=spCandidiateKF.begin(), itend=spCandidiateKF.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = *it;
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = pKFi->mRelocScore;
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit;
            if(pKF2->mnRelocQuery!=F->mnId)
                continue;

            accScore+=pKF2->mRelocScore;
            if(pKF2->mRelocScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        pKFi->mRelocAccScore = accScore;
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    lAccScoreAndMatch.sort(compFirst);

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second;
            if (pKFi->GetMap() != pMap)
                continue;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
