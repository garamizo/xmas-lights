#include "ceres/ceres.h"
#include <iostream>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/sfm.hpp>

using namespace std;
using namespace Eigen;
using namespace cv::sfm;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

int main(int argc, char* argv[])
{
    MatrixXd mat(5, 3);
    cv::Mat cvMat;
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;

    cout << mat << endl;
    return 0;
}