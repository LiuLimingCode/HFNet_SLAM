#!/bin/bash
pathDatasetEuroc='/media/llm/Datasets/EuRoC/' #Example, it is necesary to change it by the dataset path

#------------------------------------
# Monocular-Inertial Examples
echo "Launching MH01 with Monocular-Inertial sensor"
sequenceName='MH_01_easy'
./Examples/Monocular-Inertial/mono_inertial_euroc ./Examples/Monocular-Inertial/EuRoC.yaml ./evaluation/"$sequenceName"/ "$pathDatasetEuroc"/"$sequenceName" ./Examples/Monocular-Inertial/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/"$sequenceName"/mav0/state_groundtruth_estimate0/data.csv ./evaluation/"$sequenceName"/trajectory.txt --verbose --save_path ./evaluation/"$sequenceName"/

echo "Launching MH02 with Monocular-Inertial sensor"
sequenceName='MH_02_easy'
./Examples/Monocular-Inertial/mono_inertial_euroc ./Examples/Monocular-Inertial/EuRoC.yaml ./evaluation/"$sequenceName"/ "$pathDatasetEuroc"/"$sequenceName" ./Examples/Monocular-Inertial/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/"$sequenceName"/mav0/state_groundtruth_estimate0/data.csv ./evaluation/"$sequenceName"/trajectory.txt --verbose --save_path ./evaluation/"$sequenceName"/

echo "Launching MH03 with Monocular-Inertial sensor"
sequenceName='MH_03_medium'
./Examples/Monocular-Inertial/mono_inertial_euroc ./Examples/Monocular-Inertial/EuRoC.yaml ./evaluation/"$sequenceName"/ "$pathDatasetEuroc"/"$sequenceName" ./Examples/Monocular-Inertial/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/"$sequenceName"/mav0/state_groundtruth_estimate0/data.csv ./evaluation/"$sequenceName"/trajectory.txt --verbose --save_path ./evaluation/"$sequenceName"/

echo "Launching MH04 with Monocular-Inertial sensor"
sequenceName='MH_04_difficult'
./Examples/Monocular-Inertial/mono_inertial_euroc ./Examples/Monocular-Inertial/EuRoC.yaml ./evaluation/"$sequenceName"/ "$pathDatasetEuroc"/"$sequenceName" ./Examples/Monocular-Inertial/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/"$sequenceName"/mav0/state_groundtruth_estimate0/data.csv ./evaluation/"$sequenceName"/trajectory.txt --verbose --save_path ./evaluation/"$sequenceName"/

echo "Launching MH05 with Monocular-Inertial sensor"
sequenceName='MH_05_difficult'
./Examples/Monocular-Inertial/mono_inertial_euroc ./Examples/Monocular-Inertial/EuRoC.yaml ./evaluation/"$sequenceName"/ "$pathDatasetEuroc"/"$sequenceName" ./Examples/Monocular-Inertial/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/"$sequenceName"/mav0/state_groundtruth_estimate0/data.csv ./evaluation/"$sequenceName"/trajectory.txt --verbose --save_path ./evaluation/"$sequenceName"/

echo "Launching V101 with Monocular-Inertial sensor"
sequenceName='V1_01_easy'
./Examples/Monocular-Inertial/mono_inertial_euroc ./Examples/Monocular-Inertial/EuRoC.yaml ./evaluation/"$sequenceName"/ "$pathDatasetEuroc"/"$sequenceName" ./Examples/Monocular-Inertial/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/"$sequenceName"/mav0/state_groundtruth_estimate0/data.csv ./evaluation/"$sequenceName"/trajectory.txt --verbose --save_path ./evaluation/"$sequenceName"/

echo "Launching V102 with Monocular-Inertial sensor"
sequenceName='V1_02_medium'
./Examples/Monocular-Inertial/mono_inertial_euroc ./Examples/Monocular-Inertial/EuRoC.yaml ./evaluation/"$sequenceName"/ "$pathDatasetEuroc"/"$sequenceName" ./Examples/Monocular-Inertial/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/"$sequenceName"/mav0/state_groundtruth_estimate0/data.csv ./evaluation/"$sequenceName"/trajectory.txt --verbose --save_path ./evaluation/"$sequenceName"/

echo "Launching V103 with Monocular-Inertial sensor"
sequenceName='V1_03_difficult'
./Examples/Monocular-Inertial/mono_inertial_euroc ./Examples/Monocular-Inertial/EuRoC.yaml ./evaluation/"$sequenceName"/ "$pathDatasetEuroc"/"$sequenceName" ./Examples/Monocular-Inertial/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/"$sequenceName"/mav0/state_groundtruth_estimate0/data.csv ./evaluation/"$sequenceName"/trajectory.txt --verbose --save_path ./evaluation/"$sequenceName"/

echo "Launching V201 with Monocular-Inertial sensor"
sequenceName='V2_01_easy'
./Examples/Monocular-Inertial/mono_inertial_euroc ./Examples/Monocular-Inertial/EuRoC.yaml ./evaluation/"$sequenceName"/ "$pathDatasetEuroc"/"$sequenceName" ./Examples/Monocular-Inertial/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/"$sequenceName"/mav0/state_groundtruth_estimate0/data.csv ./evaluation/"$sequenceName"/trajectory.txt --verbose --save_path ./evaluation/"$sequenceName"/

echo "Launching V202 with Monocular-Inertial sensor"
sequenceName='V2_02_medium'
./Examples/Monocular-Inertial/mono_inertial_euroc ./Examples/Monocular-Inertial/EuRoC.yaml ./evaluation/"$sequenceName"/ "$pathDatasetEuroc"/"$sequenceName" ./Examples/Monocular-Inertial/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/"$sequenceName"/mav0/state_groundtruth_estimate0/data.csv ./evaluation/"$sequenceName"/trajectory.txt --verbose --save_path ./evaluation/"$sequenceName"/

echo "Launching V203 with Monocular-Inertial sensor"
sequenceName='V2_03_difficult'
./Examples/Monocular-Inertial/mono_inertial_euroc ./Examples/Monocular-Inertial/EuRoC.yaml ./evaluation/"$sequenceName"/ "$pathDatasetEuroc"/"$sequenceName" ./Examples/Monocular-Inertial/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDatasetEuroc"/"$sequenceName"/mav0/state_groundtruth_estimate0/data.csv ./evaluation/"$sequenceName"/trajectory.txt --verbose --save_path ./evaluation/"$sequenceName"/
