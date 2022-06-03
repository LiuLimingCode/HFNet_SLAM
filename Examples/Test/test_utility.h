#include <chrono>
#include <fstream>
#include <dirent.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int FilenameFilter(const struct dirent *cur)
{
    std::string str(cur->d_name);
    if(str.find(".png") != std::string::npos){
        return 1;
    }
    return 0;
}

vector<string> GetPngFiles(string png_dir)
{
   struct dirent **namelist;
   std::vector<std::string> ret;
   int n = scandir(png_dir.c_str(), &namelist, FilenameFilter, alphasort);

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

void ImageShow(const string &title, Mat image, const std::vector<KeyPoint> &keypoints)
{
    Mat image_show;
    cvtColor(image, image_show, COLOR_GRAY2BGR);

    for(const KeyPoint &kp : keypoints){
        cv::circle(image_show, kp.pt, 2, Scalar(0, 255, 0), -1);
    }

    cv::imshow(title.c_str(), image_show);
}