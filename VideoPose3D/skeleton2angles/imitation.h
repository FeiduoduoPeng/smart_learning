/*
 * @Author: Boris.Peng
 * @Date: 2020-04-13 11:09:09
 * @LastEditors: Boris.Peng
 * @LastEditTime: 2020-10-20 16:12:25
 */ 
#ifndef __IMITATION_H__
#define __IMITATION_H__

#include <iostream>
#include <fstream>
#include <deque>
#include <string>
#include <vector>

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/msg.h>
#include <sys/ipc.h>
#include <sys/types.h>
#include <assert.h>
#include <csignal>
#include <errno.h>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>


const double PI = 3.1415926536;

const std::string JointNames[] = {
    "Knee_X", "Back_Z", "Back_X", "Back_Y",
    // 4
    "Neck_Z", "Neck_X", "Head_Y",
    // 7
    "Left_Shoulder_X", "Left_Shoulder_Y", "Left_Elbow_Z", "Left_Elbow_X", "Left_Wrist_Z",
    "Left_Wrist_Y", "Left_Wrist_X",
    // 14
    "Right_Shoulder_X", "Right_Shoulder_Y", "Right_Elbow_Z", "Right_Elbow_X", "Right_Wrist_Z",
    "Right_Wrist_Y", "Right_Wrist_X"
};

const double anglesStriction[21][2] = {
    // 0
    // {-0.38, 0.38}, {-1.54, 1.54}, {-0.2, 0.5}, {-0.38, 0.38},
    {-0.38, 0.38}, {-1.0, 1.0}, {-0.15, 0.2}, {-0.2, 0.2},
    // 4
    {-1.54, 1.54}, {-0.8, 0.60}, {-0.38, 0.38},
    // 7
    {-3.14, 0.60}, {-0.38, 1.54}, {-1.54, 1.54}, {-2.10, 0.2}, {-1.54, 1.54}, {-0.38, 0.38}, {-0.60, 0.38},
    // 14
    {-3.14, 0.60}, {-1.54, 0.38}, {-1.54, 1.54}, {-2.10, 0.2}, {-1.54, 1.54}, {-0.38, 0.38}, {-0.38, 0.60}
};

class imitation{
public:
    imitation();
    ~imitation();
public:
    void skeletonCb(double *p, int n);

    double calcAngBetweenVec(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2);
    bool angleRestrict(std::vector<double> &angles);
    bool checkVecZero(Eigen::Vector3d v);

    std::vector<double> points2Angles(const std::vector<Eigen::Vector3d> &pts);
    std::vector<Eigen::Vector3d> correctPts(const std::vector<Eigen::Vector3d> &pts);
    std::vector<Eigen::Vector3d> correctFrame(const std::vector<Eigen::Vector3d> &pts);
    bool DCM2Euler(const Eigen::Matrix3d &R, std::vector<double> &angs);
    bool anglesFilter(const std::vector<double> &vec, std::vector<double> &res);

    Eigen::Matrix3d fuseR(double angY, double angZ);
    double restrictArmAndForward(int LR, std::vector<double> vv, Eigen::Vector3d aim);   //0 for L, 1 for R

    //Left arm
    bool chooseLeftShoulder(double angY, double angZ, const Eigen::Vector3d& vec);
    bool calcLeftArm(const Eigen::Vector3d& vec1, const Eigen::Vector3d& vec2, std::vector<double>& angs);
    bool checkLeftResults(const std::vector<double>& angs, const Eigen::Vector3d& vec1, const Eigen::Vector3d& vec2);
    //Right arm
    bool chooseRightShoulder(double angY, double angZ, const Eigen::Vector3d& vec);
    bool calcRightArm(const Eigen::Vector3d& vec1, const Eigen::Vector3d& vec2, std::vector<double>& angs);
    bool checkRightResults(const std::vector<double>& angs, const Eigen::Vector3d& vec1, const Eigen::Vector3d& vec2);
    //Head
    bool calcHead(const Eigen::Vector3d &vec1, const Eigen::Vector3d &vec2, std::vector<double>&angs);
    bool checkHeadResults(const std::vector<double> &angs, const Eigen::Vector3d &vec1, const Eigen::Vector3d &vec2);
    //Waist
    bool calcWaist(const std::vector<Eigen::Vector3d> &pts, std::vector<double> &angs);
    bool calcWaist(const Eigen::Vector3d &vec1, const Eigen::Vector3d &vec2, const Eigen::Vector3d &vec3, std::vector<double> &angs);
    bool checkWaistResults(const std::vector<double> &angs);

private:
    std::vector<Eigen::Vector3d> pts;
    bool runningFlag = true;
    msg_pts ptsIn;
    msg_angs angsOut;
    int msgInPtsKey = 1003;
    int msgOutAngsKey = 1004;
    int frame = 0;

private:
    const double deltaAng = 20.0/180.0*PI;
    double alpha = 0.8;         //lower pass coefficient
    double angThresh = 1.0*PI/3.0;
    int winSize = 5;
    std::string storeFile = "/home/cloud/dance_data.txt";
};


#endif
