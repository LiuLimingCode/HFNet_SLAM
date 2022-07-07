/**
 * To test the VINO api, and the base function of HFNet
 * 
 * Result:

 */
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <random>

#include "Settings.h"
#include "Frame.h"
#include "Extractors/HFNetVINOModel.h"
#include "Extractors/HFextractor.h"
#include "Examples/Utility/utility_common.h"

using namespace cv;
using namespace std;
using namespace ORB_SLAM3;
using namespace ov;

Settings *settings;
HFNetVINOModel *pModel;
TicToc timerDetect;
TicToc timerRun;

void printInputAndOutputsInfo(const ov::Model& network)
{
    std::cout << "model name: " << network.get_friendly_name() << std::endl;

    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node> input : inputs) {
        std::cout << "    inputs" << std::endl;

        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        std::cout << "        input name: " << name << std::endl;

        const ov::element::Type type = input.get_element_type();
        std::cout << "        input type: " << type << std::endl;

        const ov::Shape shape = input.get_shape();
        std::cout << "        input shape: " << shape << std::endl;
    }

    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node> output : outputs) {
        std::cout << "    outputs" << std::endl;

        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        std::cout << "        output name: " << name << std::endl;

        const ov::element::Type type = output.get_element_type();
        std::cout << "        output type: " << type << std::endl;

        const ov::Shape shape = output.get_shape();
        std::cout << "        output shape: " << shape << std::endl;
    }
}

void Mat2Tensor(const cv::Mat &image, ov::Tensor *tensor)
{
    cv::Mat imagePixel(image.rows, image.cols, CV_32F, tensor->data<float>());
    image.convertTo(imagePixel, CV_32F);
}

void ResamplerOV(const ov::Tensor &data, const ov::Tensor &warp, cv::Mat &output)
{
    const int batch_size = data.get_shape()[0];
    const int data_height = data.get_shape()[1];
    const int data_width = data.get_shape()[2];
    const int data_channels = data.get_shape()[3];

    output = cv::Mat(warp.get_shape()[0], data_channels, CV_32F);
    
    const int num_sampling_points = warp.get_shape()[0];
    if (num_sampling_points > 0)
    {
        Resampler(data.data<float>(), warp.data<float>(), output.ptr<float>(),
                  batch_size, data_height, data_width, 
                  data_channels, num_sampling_points);
    }
}

bool DetectOnlyLocal(const cv::Mat &image, std::vector<cv::KeyPoint> &vKeyPoints, cv::Mat &localDescriptors,
                            int nKeypointsNum, float threshold, int nRadius)
{
    vKeyPoints.clear();

    ov::Tensor inputTensor = pModel->mInferRequest->get_input_tensor();
    ov::Shape inputShape = inputTensor.get_shape();
    if (inputShape[2] != image.cols || inputShape[1] != image.rows || inputShape[3] != image.channels())
    {
        cerr << "The input shape in VINO model should be the same as the compile shape" << endl;
        return false;
    }

    Mat2Tensor(image, &inputTensor);
    
    timerRun.Tic();
    pModel->mInferRequest->infer();
    timerRun.Toc();

    ov::Tensor tscoreDense = pModel->mInferRequest->get_tensor(pModel->mpExecutableNet->output("pred/local_head/detector/Squeeze:0"));
    ov::Tensor tLocalDescriptorMap = pModel->mInferRequest->get_tensor(pModel->mpExecutableNet->output("local_descriptor_map"));

    const int width = tscoreDense.get_shape()[2], height = tscoreDense.get_shape()[1];
    const float scaleWidth = (tLocalDescriptorMap.get_shape()[2] - 1.f) / (float)(tscoreDense.get_shape()[2] - 1.f);
    const float scaleHeight = (tLocalDescriptorMap.get_shape()[1] - 1.f) / (float)(tscoreDense.get_shape()[1] - 1.f);

    auto vResScoresDense = tscoreDense.data<float>();
    cv::KeyPoint keypoint;
    vKeyPoints.reserve(2 * nKeypointsNum);
    for (int col = 0; col < width; ++col)
    {
        for (int row = 0; row < height; ++row)
        {
            float score = vResScoresDense[row * width + col];
            if (score >= threshold)
            {
                keypoint.pt.x = col;
                keypoint.pt.y = row;
                keypoint.response = score;
                vKeyPoints.emplace_back(keypoint);
            }
        }
    }

    vKeyPoints = NMS(vKeyPoints, width, height, nRadius);

    if (vKeyPoints.size() > nKeypointsNum)
    {
        std::partial_sort(vKeyPoints.begin(), vKeyPoints.begin() + nKeypointsNum, vKeyPoints.end(), [](const cv::KeyPoint& p1, const cv::KeyPoint& p2) {
            return p1.response > p2.response;
        });
        vKeyPoints.resize(nKeypointsNum);
    }

    localDescriptors = cv::Mat(vKeyPoints.size(), 256, CV_32F);
    ov::Tensor tWarp(ov::element::f32, {(size_t)vKeyPoints.size(), 2});
    auto pWarp = tWarp.data<float>();
    for (size_t temp = 0; temp < vKeyPoints.size(); ++temp)
    {
        pWarp[temp * 2 + 0] = scaleWidth * vKeyPoints[temp].pt.x;
        pWarp[temp * 2 + 1] = scaleHeight * vKeyPoints[temp].pt.y;
    }

    ResamplerOV(tLocalDescriptorMap, tWarp, localDescriptors);

    for (int index = 0; index < localDescriptors.rows; ++index)
    {
        cv::normalize(localDescriptors.row(index), localDescriptors.row(index));
    }

    return true;
}

// const string strDatasetPath("/media/llm/Datasets/EuRoC/MH_04_difficult/mav0/cam0/data/");
// const string strSettingsPath("Examples/Monocular-Inertial/EuRoC.yaml");
// const int dbStart = 420;
// const int dbEnd = 50;

const string strDatasetPath("/media/llm/Datasets/TUM-VI/dataset-corridor4_512_16/mav0/cam0/data/");
const string strSettingsPath("Examples/Monocular-Inertial/TUM-VI.yaml");
const int dbStart = 50;
const int dbEnd = 50;

const std::string strXmlPath("/home/llm/ROS/HFNet_ORBSLAM3_v2/model/hfnet_vino_full_f16/saved_model.xml");
const std::string strBinPath("/home/llm/ROS/HFNet_ORBSLAM3_v2/model/hfnet_vino_full_f16/saved_model.bin");

int main(int argc, char* argv[])
{
    settings = new Settings(strSettingsPath, 0);
    pModel = new HFNetVINOModel(strXmlPath, strBinPath);
    pModel->Compile({1, settings->newImSize().height, settings->newImSize().width, 1}, true);
    printInputAndOutputsInfo(*pModel->mpModel);

    vector<string> files = GetPngFiles(strDatasetPath); // get all image files
    
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(dbStart, files.size() - dbEnd);

    cv::Mat image;
    vector<KeyPoint> vKeyPoints;
    cv::Mat localDescriptors, globalDescriptors;
    
    // randomly detect an image and show the results
    char command = ' ';
    float threshold = 0.005;
    int nNMSRadius = 4;
    int select = 0;
    while(1)
    {
        if (command == 'q') break;
        else if (command == 's') select = std::max(select - 1, 0);
        else if (command == 'w') select += 1;
        else if (command == 'a') threshold = std::max(threshold - 0.005, 0.005);
        else if (command == 'd') threshold += 0.005;
        else if (command == 'z') nNMSRadius = std::max(nNMSRadius - 1, 0);
        else if (command == 'c') nNMSRadius += 1;
        else select = distribution(generator);
        cout << "command: " << command << endl;
        cout << "select: " << select << endl;
        cout << "threshold: " << threshold << endl;
        cout << "nNMSRadius: " << nNMSRadius << endl;

        image = imread(strDatasetPath + files[select], IMREAD_GRAYSCALE);
        if (settings->needToResize())
            cv::resize(image, image, settings->newImSize());
        
        timerDetect.clearBuff();
        timerRun.clearBuff();
        timerDetect.Tic();
        DetectOnlyLocal(image, vKeyPoints, localDescriptors, 1000, threshold, nNMSRadius);
        timerDetect.Toc();
        cout << "Get features number: " << vKeyPoints.size() << endl
             << "run cost time: " << timerRun.aveCost() << " milliseconds" << endl
             << "detect cost time: " << timerDetect.aveCost() << " milliseconds" << endl;
        
        ShowKeypoints("press 'q' to exit", image, vKeyPoints);
        cout << endl;
        command = cv::waitKey();
    }
    cv::destroyAllWindows();

    // detect full dataset
    {   
        timerDetect.clearBuff();
        timerRun.clearBuff();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            DetectOnlyLocal(image, vKeyPoints, localDescriptors, settings->nFeatures(), settings->threshold(), settings->nNMSRadius());
            timerDetect.Toc();
        }
        cout << "Only detect the local keypoints: " << endl
             << "run cost time: " << timerRun.aveCost() << " milliseconds" << endl
             << "detect cost time: " << timerDetect.aveCost() << " milliseconds" << endl;
    }
    {
        HFextractor extractor = HFextractor(settings->nFeatures(),settings->nNMSRadius(),settings->threshold(),pModel);

        timerDetect.clearBuff();
        timerRun.clearBuff();
        vKeyPoints.clear();
        for (const string& file : files)
        {
            image = imread(strDatasetPath + file, IMREAD_GRAYSCALE);
            if (settings->needToResize())
                cv::resize(image, image, settings->newImSize());
            timerDetect.Tic();
            extractor(image, vKeyPoints, localDescriptors, globalDescriptors);
            timerDetect.Toc();
        }
        cout << "Detect the full features with HFextractor: " << endl
             << "detect cost time: " << timerDetect.aveCost() << " milliseconds" << endl;
    }

    system("pause");

    return 0;
}
