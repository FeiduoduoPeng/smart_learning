#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

extern "C" {

const double PI = 3.1415926536;
const double deltaAng = 20.0/180.0*PI;

bool skeleton2angles(double *skeIn, double *angOut);
bool calcLeftArm(const Eigen::Vector3d &vec1, const Eigen::Vector3d &vec2, std::vector<double> &angs);
bool calcRightArm(const Eigen::Vector3d &vec1, const Eigen::Vector3d &vec2, std::vector<double> &angs);
bool calcHead(const Eigen::Vector3d &vec1, const Eigen::Vector3d &vec2, std::vector<double> &angs);
bool calcWaist(const std::vector<Eigen::Vector3d> &origPts, std::vector<double> &angs);
bool  DCM2Euler(const Eigen::Matrix3d &R, std::vector<double> &angs);
Eigen::Matrix3d  fuseR(double angY, double angZ);
double  calcAngBetweenVec(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2);
bool  checkVecZero(Eigen::Vector3d v);
std::vector<Eigen::Vector3d> correctPts(const std::vector<Eigen::Vector3d> &origPts);

}