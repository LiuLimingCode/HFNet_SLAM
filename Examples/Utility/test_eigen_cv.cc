/**
 * Result:
db size: 1000

cv2eigen: 
cost time: 6392

Eigen::Map: 
cost time: 0

Eigen copy: 
cost time: 6323

Eigen: 
cost time: 708

Eigen + for: 
cost time: 874

Eigen + parallel_for_: 
cost time: 2545

eigen2cv: 
cost time: 3087

OpenCV shallow copy: 
cost time: 0

OpenCV deep copy: 
cost time: 1385

OpenCV: 
cost time: 1209

OpenCV + for: 
cost time: 18378

OpenCV + parallel_for_: 
cost time: 3322

 * 
 */
#include <iostream>
#include <chrono>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

class ParallelEigen : public cv::ParallelLoopBody
{
public:
    ParallelEigen (const Eigen::MatrixXf &db, const Eigen::VectorXf &query, Eigen::MatrixXf &res)
        : mDb(db), mQuery(query), mRes(res) {}

    virtual void operator ()(const cv::Range& range) const CV_OVERRIDE
    {
        for (int index = range.start; index < range.end; ++index)
        {
            mRes(0, index) = (mQuery - mDb.col(index)).norm();
        }
    }

    ParallelEigen& operator=(const ParallelEigen &) {
        return *this;
    };
private:
    const Eigen::MatrixXf &mDb;
    const Eigen::VectorXf &mQuery;
    Eigen::MatrixXf &mRes;
};

class ParallelCV : public cv::ParallelLoopBody
{
public:
    ParallelCV (const cv::Mat &db, const cv::Mat &query, cv::Mat &res)
        : mDb(db), mQuery(query), mRes(res) {}

    virtual void operator ()(const cv::Range& range) const CV_OVERRIDE
    {
        for (int index = range.start; index < range.end; ++index)
        {
            mRes.col(index) = cv::norm(mQuery - mDb.col(index), cv::NORM_L2);
        }
    }

    ParallelCV& operator=(const ParallelCV &) {
        return *this;
    };
private:
    const cv::Mat &mDb;
    const cv::Mat &mQuery;
    cv::Mat &mRes;
};

int main(int argc, char** argv)
{
    const int dbSize = 1000;
    Eigen::MatrixXf dbEigen = Eigen::MatrixXf::Random(4096, dbSize);
    Eigen::VectorXf queryEigen = Eigen::VectorXf::Random(4096);
    cv::Mat dbCV, queryCV;
    cv::eigen2cv(queryEigen, queryCV);
    cv::eigen2cv(dbEigen, dbCV);

    cout << "db size: " << dbSize << endl << endl;

    {
        auto t1 = chrono::steady_clock::now();

        Eigen::MatrixXf res;
        cv::cv2eigen(dbCV, res);

        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "cv2eigen: " << endl
             << "cost time: " << t << endl << endl;
    }

    {
        auto t1 = chrono::steady_clock::now();

        float *data = dbCV.ptr<float>();
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> res(data, 4096, dbSize);

        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "Eigen::Map: " << endl
             << "cost time: " << t << endl << endl;
    }

    {
        auto t1 = chrono::steady_clock::now();

        Eigen::MatrixXf res = dbEigen;

        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "Eigen copy: " << endl
             << "cost time: " << t << endl << endl;
    }

    {
        auto t1 = chrono::steady_clock::now();

        Eigen::MatrixXf res = 2 * (Eigen::MatrixXf::Ones(1, dbSize) - queryEigen.transpose() * dbEigen);

        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "Eigen: " << endl
             << "cost time: " << t << endl << endl;
    }

    {
        auto t1 = chrono::steady_clock::now();

        Eigen::MatrixXf res(1, dbSize);
        for (int index = 0; index < dbSize; ++index)
        {
            res(0, index) = (queryEigen - dbEigen.col(index)).norm();
        }

        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "Eigen + for: " << endl
             << "cost time: " << t << endl << endl;
    }

    {
        auto t1 = chrono::steady_clock::now();

        Eigen::MatrixXf res(1, dbSize);
        ParallelEigen parallel(dbEigen, queryEigen, res);
        cv::parallel_for_(cv::Range(0, dbSize), parallel);

        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "Eigen + parallel_for_: " << endl
             << "cost time: " << t << endl << endl;
    }




    {
        auto t1 = chrono::steady_clock::now();

        cv::Mat res;
        cv::eigen2cv(dbEigen, res);

        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "eigen2cv: " << endl
             << "cost time: " << t << endl << endl;
    }

    {
        auto t1 = chrono::steady_clock::now();

        cv::Mat res = dbCV;

        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "OpenCV shallow copy: " << endl
             << "cost time: " << t << endl << endl;
    }

    {
        auto t1 = chrono::steady_clock::now();

        cv::Mat res;
        dbCV.copyTo(res);

        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "OpenCV deep copy: " << endl
             << "cost time: " << t << endl << endl;
    }

    {
        auto t1 = chrono::steady_clock::now();

        cv::Mat res = 2 * (1 - queryCV.t() * dbCV);

        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "OpenCV: " << endl
             << "cost time: " << t << endl << endl;
    }

    {
        auto t1 = chrono::steady_clock::now();

        cv::Mat res = cv::Mat_<float>(1, dbSize);
        for (int index = 0; index < dbSize; ++index)
        {
            res.col(index) = cv::norm(queryCV - dbCV.col(index), cv::NORM_L2);
        }

        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "OpenCV + for: " << endl
             << "cost time: " << t << endl << endl;
    }

    {
        auto t1 = chrono::steady_clock::now();

        cv::Mat res = cv::Mat_<float>(1, dbSize);
        ParallelCV parallel(dbCV, queryCV, res);
        cv::parallel_for_(cv::Range(0, dbSize), parallel);

        auto t2 = chrono::steady_clock::now();
        auto t = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
        cout << "OpenCV + parallel_for_: " << endl
             << "cost time: " << t << endl << endl;
    }

    system("pause");
}