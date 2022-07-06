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
    vector<KeyFrame*> vpLoopCandidates(nNumCandidates * 2);
    {
        unique_lock<mutex> lock(mMutex);

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> queryDescriptors(pKF->mGlobalDescriptors.ptr<float>(), pKF->mGlobalDescriptors.rows, pKF->mGlobalDescriptors.cols);
        for (auto it = mvDatabase.begin(); it != mvDatabase.end(); ++it)
        {
            KeyFrame *pKFi = *it;
            // Compute the distance of global descriptors
            // Eigen is much faster than OpenCV Mat
            assert(!pKFi->mGlobalDescriptors.empty());
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> pDBDescriptors(pKFi->mGlobalDescriptors.ptr<float>(), pKFi->mGlobalDescriptors.rows, pKFi->mGlobalDescriptors.cols);
            pKFi->mPlaceRecognitionScore = (queryDescriptors - pDBDescriptors).norm();
            pKFi->mnLoopQuery = pKF->mnId;
        }
        std::partial_sort_copy(mvDatabase.begin(), mvDatabase.end(), vpLoopCandidates.begin(), vpLoopCandidates.end(), [](KeyFrame* const f1, KeyFrame* const f2) {
            return f1->mPlaceRecognitionScore < f2->mPlaceRecognitionScore; // the smaller, the better
        });
    }

    vpLoopCand.reserve(nNumCandidates);
    vpMergeCand.reserve(nNumCandidates);
    set<KeyFrame*> spAlreadyAddedKF;
    int i = 0;
    auto it=vpLoopCandidates.begin();
    while(i < vpLoopCandidates.size() && (vpLoopCand.size() < nNumCandidates || vpMergeCand.size() < nNumCandidates))
    {
        KeyFrame* pKFi = *it;
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
    vector<KeyFrame*> vpLoopCandidates(10);
    {
        unique_lock<mutex> lock(mMutex);

        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> queryDescriptors(F->mGlobalDescriptors.ptr<float>(), F->mGlobalDescriptors.rows, F->mGlobalDescriptors.cols);
        for (auto it = mvDatabase.begin(); it != mvDatabase.end(); ++it)
        {
            KeyFrame *pKFi = *it;
            // Compute the distance of global descriptors
            // Eigen is much faster than OpenCV Mat
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const> pDBDescriptors(pKFi->mGlobalDescriptors.ptr<float>(), pKFi->mGlobalDescriptors.rows, pKFi->mGlobalDescriptors.cols);
            pKFi->mPlaceRecognitionScore = (queryDescriptors - pDBDescriptors).norm();
        }
        std::partial_sort_copy(mvDatabase.begin(), mvDatabase.end(), vpLoopCandidates.begin(), vpLoopCandidates.end(), [](KeyFrame* const f1, KeyFrame* const f2) {
            return f1->mPlaceRecognitionScore < f2->mPlaceRecognitionScore;
        });

    }

    float bestScore = vpLoopCandidates[0]->mPlaceRecognitionScore;
    float maxScoreToRetain = 1.25 * bestScore;

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(vpLoopCandidates.size());
    for(auto it=vpLoopCandidates.begin(), itend=vpLoopCandidates.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = *it;
        const float &si = pKFi->mPlaceRecognitionScore;
        if(si<maxScoreToRetain)
        {
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
