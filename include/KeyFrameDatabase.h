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


#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H

#include <vector>
#include <list>
#include <set>
#include <unordered_set>

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>

#include "opencv2/opencv.hpp"

#include<mutex>

using namespace std;

namespace ORB_SLAM3
{

class KeyFrame;
class Frame;
class Map;


class KeyFrameDatabase
{
    friend class boost::serialization::access;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    KeyFrameDatabase(){}

    void add(KeyFrame* pKF);

    void erase(KeyFrame* pKF);
    
    void clear();
    void clearMap(Map* pMap);

    void DetectNBestCandidates(KeyFrame *pKF, vector<KeyFrame*> &vpLoopCand, vector<KeyFrame*> &vpMergeCand, int nNumCandidates);

    // Relocalization
    std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F, Map* pMap);

    void PreSave();
    void PostLoad(map<long unsigned int, KeyFrame*> mpKFid);

protected:
    // database
    std::unordered_set<KeyFrame*> mvDatabase;

    // Mutex
    std::mutex mMutex;

    cv::BFMatcher* mpMatcher;
};

} //namespace ORB_SLAM

#endif
