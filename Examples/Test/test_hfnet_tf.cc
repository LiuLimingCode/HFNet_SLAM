#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>

#include "Extractors/HFNetTFModel.h"
#include "Examples/Test/test_utility.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;

int main(int argc, char* argv[]){

    string model_path ="model/hfnet/";
    string resampler_path ="/home/llm/src/tensorflow_cc-2.9.0/tensorflow_cc/install/lib/core/user_ops/resampler/python/ops/_resampler_ops.so";
    string dataset_path("/media/llm/Datasets/EuRoC/MH_01_easy/mav0/cam0/data/");

    vector<string> files = GetPngFiles(dataset_path); // get all image files
    HFNetTFModel feature_point(resampler_path, model_path);

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(0, files.size() - 1);

    cv::Mat image;
    vector<KeyPoint> keypoints;

    // randomly detect an image and show the results
    {
        int select = distribution(generator);
        image = imread(dataset_path + files[select], IMREAD_GRAYSCALE);
        feature_point.Detect(image, keypoints, 1000);
        ImageShow("superpoint", image, keypoints);
        cv::waitKey();
    }

    // detect full dataset
    auto t1 = chrono::steady_clock::now();
    for (const string& file : files){

        auto t_star = chrono::steady_clock::now();
        image = imread(dataset_path + file, IMREAD_GRAYSCALE);
        feature_point.Detect(image, keypoints, 1000);
        auto t_end = chrono::steady_clock::now();
        const auto t = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_star).count();
        std::cout << t << endl;
    }
    auto t2 = chrono::steady_clock::now();
    const auto t = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "===detect features time comsumed===:  " << t << " milliseconds" << endl;
    std::cout << "===average detect time consumed===:   " << (double)t / files.size() << endl;

    // result
    // ===detect features time comsumed===:  48492 milliseconds
    // ===average detect time consumed===:   13.17

    system("pause");

    return 0;
}