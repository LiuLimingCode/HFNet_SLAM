#include "Extractors/HFNetTFModel.h"
#include <time.h>
#include <chrono>
#include <fstream>
#include <dirent.h>

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;

int filenamefilter(const struct dirent *cur)
{
    std::string str(cur->d_name);
    if(str.find(".png") != std::string::npos){
        return 1;
    }
    return 0;
}

vector<string> getpngFiles(string png_dir)
{
   struct dirent **namelist;
   std::vector<std::string> ret;
   int n = scandir(png_dir.c_str(), &namelist, filenamefilter, alphasort);

   if(n < 0){
       return ret;
   }

   for (int i = 0; i < n; i++){
       std::string filepath(namelist[i]->d_name);
       ret.push_back("/" + filepath);
   }

   free(namelist);
   return ret;
}

void image_show(Mat image, const std::vector<KeyPoint> &keypoints)
{
    Mat image_show;
    cvtColor(image, image_show, COLOR_GRAY2BGR);

    for(const KeyPoint &kp : keypoints){
        cv::circle(image_show, kp.pt, 2, Scalar(0, 255, 0), -1);
    }

    cv::namedWindow("Superpoint");
    cv::imshow("Superpoint",image_show);
}

int main(int argc, char* argv[]){

    string model_path ="model/hfnet/";
    string resampler_path ="/home/llm/src/tensorflow_cc-2.9.0/tensorflow_cc/install/lib/core/user_ops/resampler/python/ops/_resampler_ops.so";
    string dataset_path("/media/llm/Datasets/EuRoC/MH_01_easy/mav0/cam0/data/");

    vector<string> files = getpngFiles(dataset_path); // get all image files
    HFNetTFModel feature_point(resampler_path, model_path);

    cv::Mat image;
    vector<KeyPoint> keypoints;

    // randomly detect an image and show the results
    {
        int select = rand() % files.size();
        image = imread(dataset_path + files[select], IMREAD_GRAYSCALE);
        feature_point.Detect(image, keypoints, 1000);
        image_show(image, keypoints);
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