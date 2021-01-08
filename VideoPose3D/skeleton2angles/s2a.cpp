/*
 * @Author: Boris.Peng
 * @Date: 2020-10-20 16:19:04
 * @LastEditors: Boris.Peng
 * @LastEditTime: 2020-10-28 18:32:19
 */

#include "ske2ang.h"

extern "C" {

bool skeleton2angles(double *skeIn, double *angOut) {
    //construct points from plain data
    std::vector<Eigen::Vector3d> origPts;
    for (int i=0; i<51; i++){
        if( i%3==0 ){
            origPts.push_back( Eigen::Vector3d(skeIn[i], skeIn[i+1], skeIn[i+2]) );
        }
    }

    // align waist
    const std::vector<Eigen::Vector3d> pts = correctPts(origPts);

    std::vector<double> angsHead(3,0), angsLeft(4,0), angsRight(4,0), angsWaist(3,0);
    bool rtn =  false;

    // calculate waist
    rtn = calcWaist(origPts, angsWaist);
    if (!rtn) {
        std::cout<<"calcWaist fail"<<std::endl;
    }
    angsWaist[0] = angsWaist[0];
    angsWaist[1] = -angsWaist[1];
    angsWaist[2] = angsWaist[2];

    // calculate head
    rtn  = calcHead(pts[9]-pts[8], pts[10]-pts[9], angsHead);
    if (!rtn) {
        std::cout<<"calcHead fail"<<std::endl;
    }
    angsHead[0] = angsHead[0];
    angsHead[1] = -angsHead[1];
    angsHead[2] = angsHead[2];

    // calculate left
    rtn = calcLeftArm(pts[12]-pts[11], pts[13]-pts[12], angsLeft);
    if (!rtn) {
        std::cout<<"calcLeftArm fail"<<std::endl;
    }
    angsLeft[0] = -angsLeft[0];
    angsLeft[1] = angsLeft[1];
    angsLeft[2] = -angsLeft[2];
    angsLeft[3] = -angsLeft[3];

    // calculate right
    rtn = calcRightArm(pts[15]-pts[14], pts[16]-pts[15], angsRight);
    if (!rtn) {
        std::cout<<"calcRightArm fail"<<std::endl;
    }
    angsRight[0] = -angsRight[0];
    angsRight[1] = angsRight[1];
    angsRight[2] = -angsRight[2];
    angsRight[3] = -angsRight[3];

    // assemble the angles
    std::vector<double> aggregate(1, 0);
    aggregate.insert(aggregate.end(), angsWaist.begin(), angsWaist.end());
    aggregate.insert(aggregate.end(), angsHead.begin(), angsHead.end());
    aggregate.insert(aggregate.end(), angsLeft.begin(), angsLeft.end());
    for (int i=0; i<3; i++) {
        aggregate.push_back(0.0);
    }
    aggregate.insert(aggregate.end(), angsRight.begin(), angsRight.end());
    for(int i=0; i<3; i++){
        aggregate.push_back(0.0);
    }
    
    //return angle through pointer
    for (int i=0; i<aggregate.size(); i++) {
        *(angOut+i) = aggregate[i];
    }
    return true;
}

bool calcLeftArm(const Eigen::Vector3d &vec1, const Eigen::Vector3d &vec2, std::vector<double> &angs) {
    angs.clear();
    angs.resize(4);
    
	Eigen::Matrix3d tilt;
    Eigen::Vector3d nPos(-cos(deltaAng), sin(deltaAng), 0);

    // calculate shoulder
    Eigen::Vector3d v1 = vec1;
    Eigen::Vector3d v2 = vec2;

    v1(2) = v1(2)<0 ? 0 : v1(2);
    v1 = v1 / v1.norm();
    v2(2) = v2(2)<0 ? 0 : v2(2);
    v2 = v2 / v2.norm();
    
    Eigen::Vector3d v1ProjXZ( v1(0), 0, v1(2));
    
    angs[0] = PI - calcAngBetweenVec(v1ProjXZ, Eigen::Vector3d(1,0,0));
    Eigen::Matrix3d rY;
    rY << cos(angs[0]), 0, sin(angs[0]),
        0, 1, 0,
        -sin(angs[0]), 0, cos(angs[0]);

    Eigen::Vector3d v1RotXY = rY.inverse()*v1;
    angs[1] = calcAngBetweenVec( v1RotXY, nPos);
    angs[1] = v1RotXY(1) > nPos(1) ? -angs[1] : angs[1];

    Eigen::Matrix3d rZ;
    rZ << cos(angs[1]), -sin(angs[1]), 0, 
          sin(angs[1]), cos(angs[1]), 0, 
          0,0,1;
    Eigen::Matrix3d R = rY*rZ;
    if( ( R.inverse()*v1- nPos).norm() > 1e-2 ){
        std::cout<< "left check failed: \n"<< R.inverse()*v1 - nPos<<std::endl;
    }

    // calculate elbow
    angs[3] = calcAngBetweenVec(v1, v2);
    if (angs[3]<5.0/180.0*PI) {
        angs[2] = 0.0;
        return true;
    }

    Eigen::Vector3d v2Rot= R.inverse()*v2;
    Eigen::Vector3d v2RotProjYZ(0, v2Rot(1), v2Rot(2));

    angs[2] = calcAngBetweenVec(v2RotProjYZ, Eigen::Vector3d(0,0,1) );    
    angs[2] = v2RotProjYZ(1) > 0 ? -angs[2] : angs[2];
	return true;
}

bool calcRightArm(const Eigen::Vector3d &vec1, const Eigen::Vector3d &vec2, std::vector<double> &angs){
    angs.clear();
    angs.resize(4);
    
	Eigen::Matrix3d tilt;
    Eigen::Vector3d nPos(-cos(deltaAng), -sin(deltaAng), 0);

    // calculate shoulder
    Eigen::Vector3d v1 = vec1;
    Eigen::Vector3d v2 = vec2;

    v1(2) = v1(2)<0 ? 0 : v1(2);
    v1 = v1 / v1.norm();
    v2(2) = v2(2)<0 ? 0 : v2(2);
    v2 = v2 / v2.norm();
    
    Eigen::Vector3d v1ProjXZ( v1(0), 0, v1(2));
    
    angs[0] = PI - calcAngBetweenVec(v1ProjXZ, Eigen::Vector3d(1,0,0));
    Eigen::Matrix3d rY;
    rY << cos(angs[0]), 0, sin(angs[0]),
        0, 1, 0,
        -sin(angs[0]), 0, cos(angs[0]);

    Eigen::Vector3d v1RotXY = rY.inverse()*v1;
    angs[1] = calcAngBetweenVec( v1RotXY, nPos);
    angs[1] = v1RotXY(1) > nPos(1) ? -angs[1] : angs[1];
    Eigen::Matrix3d rZ;
    rZ << cos(angs[1]), -sin(angs[1]), 0, 
          sin(angs[1]), cos(angs[1]), 0, 
          0,0,1;
    Eigen::Matrix3d R = rY*rZ;
    if( (R.inverse()*v1 - nPos).norm() > 1e-2){
        std::cout<< "right check failed: \n"<< R.inverse()*v1 - nPos<<std::endl;
    }

    // calculate elbow
    angs[3] = calcAngBetweenVec(v1, v2);
    if (angs[3]<5.0/180.0*PI) {
        angs[2] = 0.0;
        return true;
    }


    Eigen::Vector3d v2Rot= R.inverse()*v2;
    Eigen::Vector3d v2RotProjYZ(0, v2Rot(1), v2Rot(2));

    angs[2] = calcAngBetweenVec(v2RotProjYZ, Eigen::Vector3d(0,0,1) );    
    angs[2] = v2RotProjYZ(1) > 0 ? -angs[2] : angs[2];
	return true;
}

bool calcWaist(const std::vector<Eigen::Vector3d> &pts, std::vector<double> &angs){
    Eigen::Matrix3d chestR, legR;

    Eigen::Vector3d chestV1 = pts[11]-pts[0];
    Eigen::Vector3d chestV2 = pts[14]-pts[0];
    // Eigen::Vector3d chestV3 = pts[1]-pts[4];

    Eigen::Vector3d chestZ = chestV1.cross(chestV2);
    chestZ = chestZ /  chestZ.norm();
    Eigen::Vector3d chestY = chestV2-chestV1;
    chestY = chestY / chestY.norm();
    Eigen::Vector3d chestX = chestY.cross(chestZ);
    chestR << chestX(0), chestY(0), chestZ(0),
                chestX(1), chestY(1), chestZ(1),
                chestX(2), chestY(2), chestZ(2);

    bool rtn = DCM2Euler(chestR, angs);
    if (!rtn) {
        std::cout<<"waist DCM2Euler check fail"<<std::endl;
    }
    return true;
}

bool calcHead(const Eigen::Vector3d &vec1, const Eigen::Vector3d &vec2, std::vector<double> &angs){
    // Eigen::Vector3d xBe = vec2/vec2.norm();
    Eigen::Vector3d xBe = vec1/vec1.norm();
    Eigen::Vector3d yBe = vec1.cross(vec2);
    yBe = yBe / yBe.norm();
    Eigen::Vector3d zBe = xBe.cross(yBe);

    Eigen::Matrix3d R;
    R << xBe(0), yBe(0), zBe(0),
         xBe(1), yBe(1), zBe(1),
         xBe(2), yBe(2), zBe(2);

    bool rtn = DCM2Euler(R, angs); 
    if (!rtn) {
        std::cout<<"head DCM2Euler check fail"<<std::endl;
    }
    return rtn;
}

std::vector<Eigen::Vector3d>  correctPts(const std::vector<Eigen::Vector3d> &pts){
    Eigen::Vector3d v1 = pts[11] - pts[0];
    Eigen::Vector3d v2 = pts[14] - pts[0];
    Eigen::Vector3d yBe = pts[14] - pts[11];
    Eigen::Vector3d zBe = v1.cross(v2);
    Eigen::Vector3d xBe = yBe.cross(zBe);
    xBe = xBe / xBe.norm();
    yBe = yBe / yBe.norm();
    zBe = zBe / zBe.norm();

    Eigen::Matrix3d R;
    R<< xBe(0), yBe(0), zBe(0),
        xBe(1), yBe(1), zBe(1),
        xBe(2), yBe(2), zBe(2);
    
    std::vector<Eigen::Vector3d> ptsC;
    for( auto pt : pts ){
        ptsC.push_back( R.inverse() * pt );
    }
    return ptsC;
}

bool  DCM2Euler(const Eigen::Matrix3d &R, std::vector<double> &angs){
    Eigen::Vector3d euler1 = R.eulerAngles(0,1,2);
    Eigen::Vector3d euler2(euler1(0)-PI, PI-euler1(1), euler1(2)-PI);

    double dist1 = fabs(cos(euler1(0))-1) + fabs(cos(euler1(1))-1) + fabs(cos(euler1(2))-1);
    double dist2 = fabs(cos(euler2(0))-1) + fabs(cos(euler2(1))-1) + fabs(cos(euler2(2))-1);

    Eigen::Vector3d euler = dist1<dist2 ? euler1 : euler2;
    for (int i=0; i<3; i++) {
        if(euler[i]>PI){
            euler[i] -= 2*PI;
        } else if (euler[i]<-PI) {
            euler[i] += 2*PI;
        }
    }
    angs.clear();
    for (int i=0; i<3; i++) {
        angs.push_back(euler(i));
    }

    return true;
}

Eigen::Matrix3d  fuseR(double angY, double angZ) {
	Eigen::Matrix3d Ry, Rz;
	Ry << cos(angY), 0, sin(angY),
			0,1,0,
			-sin(angY), 0, cos(angY);
	Rz << cos(angZ), -sin(angZ), 0,
			sin(angZ), cos(angZ), 0,
			0, 0, 1;
	return Ry * Rz;
}

double  calcAngBetweenVec(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
    if( v1.norm() < 1e-5 || v2.norm()<1e-5 ){
        return 0;
    }
    double cosValue = v1.dot(v2) / v1.norm() / v2.norm();
    if(cosValue>1){
        cosValue=1;
    } else if(cosValue<-1){
        cosValue=-1;
        return PI;
    }
    return acos(cosValue);
}

bool  checkVecZero(Eigen::Vector3d v){
    if( v(0)<1e-7 && v(1)<1e-7 && v(2)<1e-7 ){
        return true;
    }else{
        return false;
    }
}

}