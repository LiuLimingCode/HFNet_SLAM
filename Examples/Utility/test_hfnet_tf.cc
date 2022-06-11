/**
 * To test the tensorflow api, and the base function of HFNet
 * 
 * Result:
 * session->Run() output size: 4
 * outputs[0].shape(): [1,1000,2]
 * outputs[1].shape(): [1,1000,256]
 * outputs[2].shape(): [1,1000]
 * outputs[3].shape(): [1,4096]
 */
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>

#include "Extractors/HFNetTFModel.h"
#include "Examples/Utility/utility_common.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;
using namespace tensorflow;

std::unique_ptr<tensorflow::Session> session;
tensorflow::GraphDef graph;

void Mat2Tensor(const cv::Mat &image, tensorflow::Tensor *tensor)
{
    float *p = tensor->flat<float>().data();
    cv::Mat imagepixel(image.rows, image.cols, CV_32F, p);
    image.convertTo(imagepixel, CV_32F);
}

bool Detect_1(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints,
            int nKeypointsNum = 1000, int nRadius = 4)
{
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    
    vector<Tensor> outputs;
    Status status = session->Run({{"image:0", tImage},{"pred/simple_nms/radius", tRadius},{"pred/top_k_keypoints/k", tKeypointsNum}},
                                 {"keypoints"}, {}, &outputs);
    if (!status.ok()) return false;

     int nResNumber = outputs[0].shape().dim_size(1);

    auto vResKeypoints = outputs[0].tensor<int32, 3>();

    vKeypoints.clear();

    KeyPoint kp;
    for(int index = 0; index < nResNumber; index++)
    {
        kp.pt = Point2f(vResKeypoints(2 * index), vResKeypoints(2 * index + 1));
        vKeypoints.push_back(kp);
    }
    return true;
}

bool Detect_2(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors,
            int nKeypointsNum = 1000, int nRadius = 4)
{
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    
    vector<Tensor> outputs;
    Status status = session->Run({{"image:0", tImage},{"pred/simple_nms/radius", tRadius},{"pred/top_k_keypoints/k", tKeypointsNum}},
                                {"keypoints", "local_descriptors"}, {}, &outputs);
    if (!status.ok()) return false;

     int nResNumber = outputs[0].shape().dim_size(1);

    auto vResKeypoints = outputs[0].tensor<int32, 3>();
    auto vResLocalDes = outputs[1].tensor<float, 3>();

    vKeypoints.clear();
    localDescriptors = cv::Mat::zeros(nResNumber, 256, CV_32F);

    KeyPoint kp;
    for(int index = 0; index < nResNumber; index++)
    {
        kp.pt = Point2f(vResKeypoints(2 * index), vResKeypoints(2 * index + 1));
        vKeypoints.push_back(kp);
        for (int temp = 0; temp < 256; ++temp)
        {
            localDescriptors.ptr<float>(index)[temp] = vResLocalDes(256 * index + temp); 
        }
    }
    return true;
}

bool Detect_3(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors,
            int nKeypointsNum = 1000, int nRadius = 4)
{
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    
    vector<Tensor> outputs;
    Status status = session->Run({{"image:0", tImage},{"pred/simple_nms/radius", tRadius},{"pred/top_k_keypoints/k", tKeypointsNum}},
                                {"keypoints", "local_descriptors", "scores"}, {}, &outputs);
    if (!status.ok()) return false;

     int nResNumber = outputs[0].shape().dim_size(1);

    auto vResKeypoints = outputs[0].tensor<int32, 3>();
    auto vResLocalDes = outputs[1].tensor<float, 3>();
    auto vResScores = outputs[2].tensor<float, 2>();

    vKeypoints.clear();
    localDescriptors = cv::Mat::zeros(nResNumber, 256, CV_32F);

    KeyPoint kp;
    for(int index = 0; index < nResNumber; index++)
    {
        kp.pt = Point2f(vResKeypoints(2 * index), vResKeypoints(2 * index + 1));
        kp.response = vResScores(index);
        vKeypoints.push_back(kp);
        for (int temp = 0; temp < 256; ++temp)
        {
            localDescriptors.ptr<float>(index)[temp] = vResLocalDes(256 * index + temp); 
        }
    }
    return true;
}

bool Detect_full(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeypoints, cv::Mat &localDescriptors, cv::Mat &globalDescriptors,
            int nKeypointsNum = 1000, int nRadius = 4)
{
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    
    vector<Tensor> outputs;
    Status status = session->Run({{"image:0", tImage},{"pred/simple_nms/radius", tRadius},{"pred/top_k_keypoints/k", tKeypointsNum}},
                                {"keypoints", "local_descriptors", "scores", "global_descriptor"}, {}, &outputs);
    if (!status.ok()) return false;

    int nResNumber = outputs[0].shape().dim_size(1);

    auto vResKeypoints = outputs[0].tensor<int32, 3>();
    auto vResLocalDes = outputs[1].tensor<float, 3>();
    auto vResScores = outputs[2].tensor<float, 2>();
    auto vResGlobalDes = outputs[3].tensor<float, 2>();

    vKeypoints.clear();
    localDescriptors = cv::Mat::zeros(nResNumber, 256, CV_32F);
    globalDescriptors = cv::Mat::zeros(4096, 1, CV_32F);

    KeyPoint kp;
    for(int index = 0; index < nResNumber; index++)
    {
        kp.pt = Point2f(vResKeypoints(2 * index), vResKeypoints(2 * index + 1));
        kp.response = vResScores(index);
        vKeypoints.push_back(kp);
        for (int temp = 0; temp < 256; ++temp)
        {
            localDescriptors.ptr<float>(index)[temp] = vResLocalDes(256 * index + temp); 
        }
    }
    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDes(temp);
    }
    return true;
}

bool Detect_g(const cv::Mat &image, cv::Mat &globalDescriptors,
            int nKeypointsNum = 1000, int nRadius = 4)
{
    Tensor tKeypointsNum(DT_INT32, TensorShape());
    Tensor tRadius(DT_INT32, TensorShape());
    tKeypointsNum.scalar<int>()() = nKeypointsNum;
    tRadius.scalar<int>()() = nRadius;

    Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
    Mat2Tensor(image, &tImage);
    
    vector<Tensor> outputs;
    Status status = session->Run({{"image:0", tImage},{"pred/simple_nms/radius", tRadius},{"pred/top_k_keypoints/k", tKeypointsNum}},
                                {"global_descriptor"}, {}, &outputs);
    if (!status.ok()) return false;

    auto vResGlobalDes = outputs[0].tensor<float, 2>();

    for (int temp = 0; temp < 4096; ++temp)
    {
        globalDescriptors.ptr<float>(0)[temp] = vResGlobalDes(temp);
    }
    return true;
}

int main(int argc, char* argv[])
{
    const string strModelPath("model/hfnet/");
    const string strResamplerPath("/home/llm/src/tensorflow_cc-2.9.0/tensorflow_cc/install/lib/core/user_ops/resampler/python/ops/_resampler_ops.so");
    const string strDatasetPath("/media/llm/Datasets/EuRoC/MH_04_difficult/mav0/cam0/data/");

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(1020, 1400);

    cv::Mat image;
    vector<KeyPoint> keypoints;
    cv::Mat localDescriptors, globalDescriptors;
    
    {
        TF_Status *status = TF_NewStatus();
        TF_LoadLibrary(strResamplerPath.c_str(), status);
        if (TF_GetCode(status) != TF_OK) {
            std::cerr << "TF_LoadLibrary() error with code: " << TF_GetCode(status) << std::endl;
            std::cerr << "Failed to load resampler.so in path: " << strResamplerPath << std::endl;
            return false;
        }
        std::cout << "Sucessfully loaded resampler.so" << std::endl;
    }

    {
        tensorflow::Status status;
        tensorflow::SessionOptions sessionOptions;
        tensorflow::RunOptions runOptions;
        tensorflow::SavedModelBundle bundle;
        
        status = LoadSavedModel(sessionOptions, runOptions, strModelPath, {tensorflow::kSavedModelTagServe}, &bundle);
        if(!status.ok()){
            std::cerr << "Failed to load HFNet model at path: " << strModelPath <<std::endl;
            return false;
        }

        session = std::move(bundle.session);
        status = session->Create(graph);
        if(!status.ok()){
            std::cerr << "Failed to create graph for HFNet" << std::endl;
            return false;
        }

        std::cout << "Sucessfully loaded HFNet model" << std::endl;
    }

    // randomly detect an image and show the results
    char command = ' ';
    float threshold = 0;
    int select = 0;
    while(1) {
        if (command == 'q') break;
        else if (command == 'a') threshold -= 0.005;
        else if (command == 'd') threshold += 0.005;
        else if (command == 'w') select += 1;
        else if (command == 's') select -= 1;
        else select = distribution(generator);
        cout << "command: " << command << endl;
        cout << "threshold: " << threshold << endl;

        image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        
        Tensor tKeypointsNum(DT_INT32, TensorShape());
        Tensor tRadius(DT_INT32, TensorShape());
        tKeypointsNum.scalar<int>()() = 1000;
        tRadius.scalar<int>()() = 4;

        Tensor tImage(DT_FLOAT, TensorShape({1, image.rows, image.cols, 1}));
        Mat2Tensor(image, &tImage);
        
        vector<Tensor> outputs;
        Status status = session->Run({{"image:0", tImage},{"pred/simple_nms/radius", tRadius},{"pred/top_k_keypoints/k", tKeypointsNum}},
                                    {"keypoints", "local_descriptors", "scores", "global_descriptor"}, {}, &outputs);

        cout << "session->Run() output size: " << outputs.size() << endl;
        cout << "outputs[0].shape(): " << outputs[0].shape() << endl;
        cout << "outputs[1].shape(): " << outputs[1].shape() << endl;
        cout << "outputs[2].shape(): " << outputs[2].shape() << endl;
        cout << "outputs[3].shape(): " << outputs[3].shape() << endl;

        int nResNumber = outputs[0].shape().dim_size(1);

        auto vResKeypoints = outputs[0].tensor<int32, 3>();
        auto vResLocalDes = outputs[1].tensor<float, 3>();
        auto vResScores = outputs[2].tensor<float, 2>();
        auto vResGlobalDes = outputs[3].tensor<float, 2>();

        keypoints.clear();
        localDescriptors = cv::Mat::zeros(nResNumber, 256, CV_32F);
        globalDescriptors = cv::Mat::zeros(4096, 1, CV_32F);

        KeyPoint kp;
        for(int index = 0; index < nResNumber; index++)
        {
            if (vResScores(index) < threshold) continue;
            
            kp.pt = Point2f(vResKeypoints(2 * index), vResKeypoints(2 * index + 1));
            kp.response = vResScores(index);
            keypoints.push_back(kp);
            for (int temp = 0; temp < 256; ++temp)
            {
                localDescriptors.ptr<float>(index)[temp] = vResLocalDes(256 * index + temp); 
            }
        }
        for (int temp = 0; temp < 4096; ++temp)
        {
            globalDescriptors.ptr<float>(0)[temp] = vResGlobalDes(temp);
        }
        ShowKeypoints("press 'q' to exit", image, keypoints);
        command = cv::waitKey();
    }

    // // detect full dataset
    // chrono::steady_clock::time_point t1, t2;
    // int t;

    // cout << "detect: {\"keypoints\"}" << endl;
    // t1 = chrono::steady_clock::now();
    // for (const string& file : files)
    // {
    //     image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
    //     Detect_1(image, keypoints);
    // }
    // t2 = chrono::steady_clock::now();
    // t = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    // std::cout << "cost time: " << t << " milliseconds" << endl;
    // std::cout << "average detect time: " << (double)t / files.size() << endl;


    // cout << "detect: {\"keypoints\", \"local_descriptors\"}" << endl;
    // t1 = chrono::steady_clock::now();
    // for (const string& file : files)
    // {
    //     image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
    //     Detect_2(image, keypoints, localDescriptors);
    // }
    // t2 = chrono::steady_clock::now();
    // t = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    // std::cout << "cost time: " << t << " milliseconds" << endl;
    // std::cout << "average detect time: " << (double)t / files.size() << endl;


    // cout << "detect: {\"keypoints\", \"local_descriptors\", \"scores\"}" << endl;
    // t1 = chrono::steady_clock::now();
    // for (const string& file : files)
    // {
    //     image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
    //     Detect_3(image, keypoints, localDescriptors);
    // }
    // t2 = chrono::steady_clock::now();
    // t = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    // std::cout << "cost time: " << t << " milliseconds" << endl;
    // std::cout << "average detect time: " << (double)t / files.size() << endl;


    // cout << "detect: {\"keypoints\", \"local_descriptors\", \"scores\", \"global_descriptor\"}" << endl;
    // t1 = chrono::steady_clock::now();
    // for (const string& file : files)
    // {
    //     image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
    //     Detect_full(image, keypoints, localDescriptors, globalDescriptors);
    // }
    // t2 = chrono::steady_clock::now();
    // t = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    // std::cout << "cost time: " << t << " milliseconds" << endl;
    // std::cout << "average detect time: " << (double)t / files.size() << endl;


    // cout << "detect: {\"global_descriptor\"}" << endl;
    // t1 = chrono::steady_clock::now();
    // for (const string& file : files)
    // {
    //     image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
    //     Detect_g(image, globalDescriptors);
    // }
    // t2 = chrono::steady_clock::now();
    // t = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    // std::cout << "cost time: " << t << " milliseconds" << endl;
    // std::cout << "average detect time: " << (double)t / files.size() << endl;

    system("pause");

    return 0;
}