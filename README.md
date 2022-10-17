# HFNet_SLAM

More detailed README.md is coming soon ....

## Prerequisites

### OpenCV

We use OpenCV to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. Required at leat 4.4

### TensorFlow C++ API

We use TensorFlow C++, CUDA, and cuDNN for model inference. The corresponding version of these libraries should be chosen wisely according to the devices. The following configurations has been tested:

| Name   | version  |
| --------   | -------:  |
| Ubuntu | 20.04     |
| GPU     | RTX 2070    |
| NVIDIA Driver   |  510.47  |
| CUDA tool  |  11.6.2 |
| cuDNN | 8.4.1.50 |
| TensorFlow | 2.9.0 |

The download and install instructions of CUDA can be found at: https://developer.nvidia.com/cuda-toolkit
The instructions of cuDNN can be found at: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
An easy method for building TensorFlow C++ API can be found at: https://github.com/FloopCZ/tensorflow_cc

## Building HFNet-SLAM library and examples

```
bash build.sh
```

## Evaluation on EuRoC dataset

Evaluate a single sequence with the pure monocular configuration:

```
sequenceName='MH01'
./Examples/Monocular/mono_euroc ./Examples/Monocular/EuRoC.yaml "$pathEvaluation"/"$sequenceName"_MONO/ "$pathDataset"/"$sequenceName" ./Examples/Monocular/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py ./evaluation/Ground_truth/EuRoC_left_cam/"$sequenceName"_GT.txt "$pathEvaluation"/"$sequenceName"_MONO/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"_MONO/
```

Evaluate a single sequence with the monocular-inertial configuration:

```
sequenceName='MH01'
./Examples/Monocular-Inertial/mono_inertial_euroc ./Examples/Monocular-Inertial/EuRoC.yaml "$pathEvaluation"/"$sequenceName"_MONO_IN/ "$pathDataset"/"$sequenceName" ./Examples/Monocular-Inertial/EuRoC_TimeStamps/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"/mav0/state_groundtruth_estimate0/data.csv "$pathEvaluation"/"$sequenceName"_MONO_IN/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"_MONO_IN/
```

Evaluate the whole dataset:

```
bash Examples/eval_euroc.sh
```

## Evaluation on TUM-VI dataset

Evaluate a single sequence with the monocular-inertial configuration:

In 'outdoors' sequences, Use './Examples/Monocular-Inertial/TUM-VI_far.yaml' configuration file instead.

```
sequenceName='dataset-corridor1_512'
./Examples/Monocular-Inertial/mono_inertial_tum_vi ./Examples/Monocular-Inertial/TUM-VI.yaml "$pathEvaluation"/"$sequenceName"/ "$pathDataset"/"$sequenceName"_16/mav0/cam0/data ./Examples/Monocular-Inertial/TUM_TimeStamps/"$sequenceName".txt ./Examples/Monocular-Inertial/TUM_IMU/"$sequenceName".txt
python3 ./evaluation/evaluate_ate_scale.py "$pathDataset"/"$sequenceName"_16/mav0/mocap0/data.csv "$pathEvaluation"/"$sequenceName"/trajectory.txt --verbose --save_path "$pathEvaluation"/"$sequenceName"/
```

Evaluate the whole dataset:

```
bash Examples/eval_tum_vi.sh
```
