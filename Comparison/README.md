# HFNet-SLAM vs. ORB-SLAM3

This document is to compare the performance of HFNet-SLAM system with ORB-SLAM3 system in terms of feature matching, loop detection, calculation consumption, and so on.

## Build

```
chmod +x build.sh
bash build.sh
```

## Compare extractors

```
# EuRoC Dataset
./app/compare_extractors PATH_TO_EuRoC/MH01/mav0/cam0/data/ PATH_TO_MODEL

# TUM-VI Dataset
./app/compare_extractors PATH_TO_TUMVI/dataset-magistrale1_512_16/mav0/cam0/data/ PATH_TO_MODEL
```

ORB-SLAM3:
cost time: 12 ms
key point number: 1010

HFNet-SLAM (TensorFlow):
cost time: 19 ms
key point number: 1000

HFNet-SLAM (TensorRT):
cost time: 9 ms
key point number: 1000

<img src="result/compare_extractors.png" width="600" title="compare_extractors">

**Comment**: The feature extraction of HFNet-SLAM is more likely to extract features with significant textures. In terms of calculation cost, HFNet-SLAM (TensorFlow) > ORB-SLAM3 > HFNet-SLAM (TensorRT).

## Compare matchers

```
# EuRoC Dataset
./app/compare_matchers PATH_TO_EuRoC/MH01/mav0/cam0/data/ PATH_TO_MODEL Vocabulary/ORBvoc.txt

# TUM-VI Dataset
./app/compare_matchers PATH_TO_TUMVI/dataset-magistrale1_512_16/mav0/cam0/data/ PATH_TO_MODEL Vocabulary/ORBvoc.txt
```

ORB + SearchByBoW:
vocab costs time: 6.74603ms
match costs time: 0.617864ms
matches total number: 112
correct matches number: 95
match correct percentage: 0.848214

<img src="result/compare_matchers ORB-SLAM3.png" width="600" title="compare_matchers ORB-SLAM3">

HF + SearchByBoWV2:
match costs time: 5.46734ms
matches total number: 251
correct matches number: 232
match correct percentage: 0.924303

<img src="result/compare_matchers HFNet-SLAM.png" width="600" title="compare_matchers HFNet-SLAM">

**Comment**: The matching strategy in HFNet-SLAM has higher matching ability, and the SIMD technology improves the efficiency.

## Compare loop detection

```
# EuRoC Dataset
./app/compare_loop_detection PATH_TO_EuRoC/MH01/mav0/cam0/data/ PATH_TO_MODEL Vocabulary/ORBvoc.txt

# TUM-VI Dataset
./app/compare_loop_detection PATH_TO_TUMVI/dataset-magistrale1_512_16/mav0/cam0/data/ PATH_TO_MODEL Vocabulary/ORBvoc.txt
```

ORB-SLAM3: 
Query cost time: 24426

<img src="result/compare_loop_detection ORB-SLAM3.png" width="600" title="compare_loop_detection ORB-SLAM3">

HFNet-SLAM: 
Query cost time: 420

<img src="result/compare_loop_detection HFNet-SLAM.png" width="600" title="compare_loop_detection HFNet-SLAM">

**Comment**: The loop detection of HFNet-SLAM has higher recall and precision compared with ORB-SLAM3. Besides, it is more effective.

## Compare Runtime Performance:

Add `ADD_DEFINITIONS(-DREGISTER_TIMES)` to the root CMakeLists.txt and then rebuild the project to enable runtime analysis. 

<img src="https://user-images.githubusercontent.com/52725165/197371705-e437adc0-ed47-4bb7-a3db-ff0a091b2568.png" title="Comparative Runtime Performance">

**Comment**: HFNet-SLAM has more effective mapping and loop detection threads compared with ORB-SLAM3, but it needs an extra 10 ms and GPU support in the tracking tread because the inference of the HF-Net model uses float64 precision. There is a great potential improvement by using quantization and half-precision technologies to increase the runtime performance.
