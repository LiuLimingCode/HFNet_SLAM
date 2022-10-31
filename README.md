# HFNet-SLAM

## An accurate and real-time monocular SLAM system with deep features

More detailed README.md is coming soon ....

HFNet-SLAM is the combination and extension of the well-known [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) SLAM framework and a unified CNN model called [HF-Net](https://github.com/ethz-asl/hfnet). It uses the image features from HF-Net to fully replace the hand-crafted ORB features and the BoW method in the ORB-SLAM3 system. This novelty results in better performance in tracking and loop closure, boosting the accuracy of the entire HFNet-SLAM.

## HFNet-SLAM vs ORB-SLAM3

**Better Tracking**:

<img src="https://user-images.githubusercontent.com/52725165/197087949-21196670-335e-4ea9-ac12-f226521da691.png" width="600" title="Better Tracking">

**Better Loop Closure**:

<img src="https://user-images.githubusercontent.com/52725165/197088191-1d01fe8a-02ef-4002-8eeb-3c312ef48eb4.png" width="600" title="Better Loop Closure">

## Prerequisites

### OpenCV

We use OpenCV to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org.

### TensorFlow C++ API

We use TensorFlow C++, CUDA, and cuDNN for model inference. The corresponding version of these libraries should be chosen wisely according to the devices. The following configurations has been tested:

| Name   | Version  |
| --------   | :-------:  |
| Ubuntu | 20.04     |
| GPU     | RTX 2070 Max-Q   |
| NVIDIA Driver   |  510.47  |
| CUDA tool  |  11.6.2 |
| cuDNN | 8.4.1.50 |
| TensorFlow | 1.15, 2.9 |

The download and install instructions of CUDA can be found at: https://developer.nvidia.com/cuda-toolkit.

The instructions of cuDNN can be found at: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html.

An easy method for building TensorFlow C++ API can be found at: https://github.com/FloopCZ/tensorflow_cc.

### Download the HF-Net Model

Download the converted model from [here](https://drive.google.com/file/d/1vBmC5t3NDIBULh8xeM2A_Qs2aDlaMAOz/view?usp=share_link).

More details about the process of converting can be found in the [HF-Net Model Converting](hfnet/README.md) document.

## Building HFNet-SLAM library and examples

```
bash build.sh
```

### ROS(optional)

Some examples using ROS are provided. Building these examples is optional. These have been tested with ROS Noetic under Ubuntu 20.04.

## Evaluation on EuRoC dataset

https://user-images.githubusercontent.com/52725165/197089468-99c7ebf2-18c7-45da-a62e-b69691c3d248.mp4

Evaluate a single sequence with the pure monocular configuration:

```
pathDataset='PATH/Datasets/EuRoC/'
pathEvaluation='./evaluation/Euroc/'
sequenceName='MH01'
./Examples/Monocular/mono_euroc ./Examples/Monocular/EuRoC.yaml "$pathEvaluation"/"$sequenceName"_MONO/ "$pathDataset"/"$sequenceName" ./Examples/Monocular/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py ./evaluation/Ground_truth/EuRoC_left_cam/"$sequenceName"_GT.txt "$pathEvaluation"/"$sequenceName"_MONO/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"_MONO/
```

Evaluate a single sequence with the monocular-inertial configuration:

```
pathDataset='PATH/Datasets/EuRoC/'
pathEvaluation='./evaluation/Euroc/'
sequenceName='MH01'
./Examples/Monocular-Inertial/mono_inertial_euroc ./Examples/Monocular-Inertial/EuRoC.yaml "$pathEvaluation"/"$sequenceName"_MONO_IN/ "$pathDataset"/"$sequenceName" ./Examples/Monocular-Inertial/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"/mav0/state_groundtruth_estimate0/data.csv "$pathEvaluation"/"$sequenceName"_MONO_IN/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"_MONO_IN/
```

Evaluate the whole dataset:

```
bash Examples/eval_euroc.sh
```

## Evaluation on TUM-VI dataset

https://user-images.githubusercontent.com/52725165/197089404-e2c96c00-cfb0-4f73-983a-94476a01c009.mp4

Evaluate a single sequence with the monocular-inertial configuration:

In 'outdoors' sequences, Use './Examples/Monocular-Inertial/TUM-VI_far.yaml' configuration file instead.

```
pathDataset='PATH/Datasets/TUM-VI/'
pathEvaluation='./evaluation/TUM-VI/'
sequenceName='dataset-corridor1_512'
./Examples/Monocular-Inertial/mono_inertial_tum_vi ./Examples/Monocular-Inertial/TUM-VI.yaml "$pathEvaluation"/"$sequenceName"/ "$pathDataset"/"$sequenceName"_16/mav0/cam0/data ./Examples/Monocular-Inertial/TUM_TimeStamps/"$sequenceName".txt ./Examples/Monocular-Inertial/TUM_IMU/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"_16/mav0/mocap0/data.csv "$pathEvaluation"/"$sequenceName"/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"/
```

Evaluate the whole dataset:

```
bash Examples/eval_tum_vi.sh
```

## Evaluation on TUM-RGBD dataset

Evaluate a single sequence with the RGB-D configuration:

```
pathDataset='PATH/Datasets/TUM-RGBD/'
pathEvaluation='./evaluation/TUM-RGBD/'
sequenceName='fr1_desk'
echo "Launching $sequenceName with RGB-D sensor"
./Examples/RGB-D/rgbd_tum ./Examples/RGB-D/TUM1.yaml "$pathEvaluation"/"$sequenceName"/ "$pathDataset"/"$sequenceName"/ ./Examples/RGB-D/associations/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"/groundtruth.txt "$pathEvaluation"/"$sequenceName"/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"/
```

Evaluate the whole dataset:

```
bash Examples/eval_tum_rgbd.sh
```

## Evaluation with ROS

Tested with ROS Noetic and ubuntu 20.04.

1. Add the path including *Examples/ROS/HFNet_SLAM* to the ROS_PACKAGE_PATH environment variable.

```
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:PATH/HFNet_SLAM/Examples/ROS
```
  
2. Execute `build_ros.sh` script:

```
chmod +x build_ros.sh
./build_ros.sh
```

3. We provide some simple nodes with public benchmarks

```
roslaunch HFNet_SLAM mono_euroc.launch # Monocular configuration in EuRoC dataset
roslaunch HFNet_SLAM mono_inertial_euroc.launch # Monocular Inertial configuration in EuRoC dataset
roslaunch HFNet_SLAM rgbd_tum.launch # RGB-D configuration in TUM-RGBD dataset
```