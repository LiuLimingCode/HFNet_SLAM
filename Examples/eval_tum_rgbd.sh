#!/bin/bash

pathDataset='/media/llm/Datasets/TUM-RGBD/' # it is necesary to change it by the dataset path
pathEvaluation='./evaluation/TUM-RGBD/'

#------------------------------------
sequenceName='fr1_desk'
echo "Launching $sequenceName with RGB-D sensor"
./Examples/RGB-D/rgbd_tum ./Examples/RGB-D/TUM1.yaml "$pathEvaluation"/"$sequenceName"/ "$pathDataset"/"$sequenceName"/ ./Examples/RGB-D/associations/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"/groundtruth.txt "$pathEvaluation"/"$sequenceName"/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"/

sequenceName='fr1_desk2'
echo "Launching $sequenceName with RGB-D sensor"
./Examples/RGB-D/rgbd_tum ./Examples/RGB-D/TUM1.yaml "$pathEvaluation"/"$sequenceName"/ "$pathDataset"/"$sequenceName"/ ./Examples/RGB-D/associations/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"/groundtruth.txt "$pathEvaluation"/"$sequenceName"/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"/

sequenceName='fr1_room'
echo "Launching $sequenceName with RGB-D sensor"
./Examples/RGB-D/rgbd_tum ./Examples/RGB-D/TUM1.yaml "$pathEvaluation"/"$sequenceName"/ "$pathDataset"/"$sequenceName"/ ./Examples/RGB-D/associations/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"/groundtruth.txt "$pathEvaluation"/"$sequenceName"/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"/

sequenceName='fr1_xyz'
echo "Launching $sequenceName with RGB-D sensor"
./Examples/RGB-D/rgbd_tum ./Examples/RGB-D/TUM1.yaml "$pathEvaluation"/"$sequenceName"/ "$pathDataset"/"$sequenceName"/ ./Examples/RGB-D/associations/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"/groundtruth.txt "$pathEvaluation"/"$sequenceName"/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"/

#------------------------------------
sequenceName='fr2_desk'
echo "Launching $sequenceName with RGB-D sensor"
./Examples/RGB-D/rgbd_tum ./Examples/RGB-D/TUM2.yaml "$pathEvaluation"/"$sequenceName"/ "$pathDataset"/"$sequenceName"/ ./Examples/RGB-D/associations/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"/groundtruth.txt "$pathEvaluation"/"$sequenceName"/trajectory_keyframe.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"/

sequenceName='fr2_xyz'
echo "Launching $sequenceName with RGB-D sensor"
./Examples/RGB-D/rgbd_tum ./Examples/RGB-D/TUM2.yaml "$pathEvaluation"/"$sequenceName"/ "$pathDataset"/"$sequenceName"/ ./Examples/RGB-D/associations/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"/groundtruth.txt "$pathEvaluation"/"$sequenceName"/trajectory_keyframe.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"/

#------------------------------------
sequenceName='fr3_nstr_tex_near'
echo "Launching $sequenceName with RGB-D sensor"
./Examples/RGB-D/rgbd_tum ./Examples/RGB-D/TUM3.yaml "$pathEvaluation"/"$sequenceName"/ "$pathDataset"/"$sequenceName"/ ./Examples/RGB-D/associations/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"/groundtruth.txt "$pathEvaluation"/"$sequenceName"/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"/

sequenceName='fr3_office'
echo "Launching $sequenceName with RGB-D sensor"
./Examples/RGB-D/rgbd_tum ./Examples/RGB-D/TUM3.yaml "$pathEvaluation"/"$sequenceName"/ "$pathDataset"/"$sequenceName"/ ./Examples/RGB-D/associations/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"/groundtruth.txt "$pathEvaluation"/"$sequenceName"/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"/

sequenceName='fr3_office_val'
echo "Launching $sequenceName with RGB-D sensor"
./Examples/RGB-D/rgbd_tum ./Examples/RGB-D/TUM3.yaml "$pathEvaluation"/"$sequenceName"/ "$pathDataset"/"$sequenceName"/ ./Examples/RGB-D/associations/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"/groundtruth.txt "$pathEvaluation"/"$sequenceName"/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"/

sequenceName='fr3_str_tex_far'
echo "Launching $sequenceName with RGB-D sensor"
./Examples/RGB-D/rgbd_tum ./Examples/RGB-D/TUM3.yaml "$pathEvaluation"/"$sequenceName"/ "$pathDataset"/"$sequenceName"/ ./Examples/RGB-D/associations/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"/groundtruth.txt "$pathEvaluation"/"$sequenceName"/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"/

sequenceName='fr3_str_tex_near'
echo "Launching $sequenceName with RGB-D sensor"
./Examples/RGB-D/rgbd_tum ./Examples/RGB-D/TUM3.yaml "$pathEvaluation"/"$sequenceName"/ "$pathDataset"/"$sequenceName"/ ./Examples/RGB-D/associations/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"/groundtruth.txt "$pathEvaluation"/"$sequenceName"/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"/
