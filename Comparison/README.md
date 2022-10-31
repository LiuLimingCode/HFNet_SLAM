# Comparison with ORB-SLAM3

## Compare extractors

ORB-SLAM3:
cost time: 12
key point number: 1010

HFNet-SLAM:
cost time: 19
key point number: 1000

<img src="result/compare_extractors.png" width="600" title="compare_extractors">

Comment: The feature extraction of HFNet-SLAM is more likely to extract features with significant textures. But the calculation is more expensive compared with ORB-SLAM3.

## Compare matchers

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

Comment: The matching strategy in HFNet-SLAM has higher matching ability, and the SIMD technology improves the efficiency.

## Compare loop detection

ORB-SLAM3: 
Query cost time: 24426

<img src="result/compare_loop_detection ORB-SLAM3.png" width="600" title="compare_loop_detection ORB-SLAM3">

HFNet-SLAM: 
Query cost time: 420

<img src="result/compare_loop_detection HFNet-SLAM.png" width="600" title="compare_loop_detection HFNet-SLAM">

Comment: The loop detection of HFNet-SLAM has higher recall and precision compared with ORB-SLAM3. Besides, it is more effective.


**Comparative Runtime Performance**:

<img src="https://user-images.githubusercontent.com/52725165/197371705-e437adc0-ed47-4bb7-a3db-ff0a091b2568.png" title="Comparative Runtime Performance">

HFNet-SLAM has more effective mapping and loop detection threads compared with ORB-SLAM3, but it needs an extra 10 ms and GPU support in the tracking tread because the inference of the HF-Net model uses float64 precision. There is a great potential improvement by using quantization and half-precision technologies to increase the runtime performance.