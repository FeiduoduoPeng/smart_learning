#include "imitation.h"
#include <random>

extern "C" {

void  skeletonCb(double *p, int n){
    std::vector<double> data;
    for (int i=0; i<n; i++){
        data.push_back(*(p+i));
    }

    if ( data.size()%3 != 0 ) {
        std::cout<<"number of data is error"<<std::endl;
        return;
    }
    pts.clear();
    for(int i=0; i<data.size(); i++){
        if( i%3==0 ){
            pts.push_back( Eigen::Vector3d(data[i], data[i+1], data[i+2]) );
        }
    }
    if (pts.size()!=17) {
        std::cout<<"number of pts is error"<<std::endl;
        return;
    }

    int idx = 0;
    std::vector<double> jointsValue = points2Angles(pts);
    if (jointsValue.size()!=21) {
        std::cout<<"number of jointsValue error: " << jointsValue.size()<<std::endl;
        return;
    }

    bool rtn = angleRestrict(jointsValue);
    anglesFilter(jointsValue, jointsValue);

    // limit the waist
    for (int i=0; i<3; i++) {
        jointsValue[i] *= 2.0;
        jointsValue[i] = sin(jointsValue[i]);
        jointsValue[i] /= 2.0;
    }

    for (int i=0; i<jointsValue.size(); i++){
        angsOut.angs[i] = jointsValue[i]; 
    }

    std::cout << "*****************" << frame++ << "******************" << std::endl;    // done
}

std::vector<double>  points2Angles(const std::vector<Eigen::Vector3d> &points){
    assert(points.size() == 17);

    // const std::vector<Eigen::Vector3d> alignFrame = correctFrame(points);
    const std::vector<Eigen::Vector3d> &alignFrame = points;
    const std::vector<Eigen::Vector3d> pts = correctPts(alignFrame);

    std::vector<double> angsHead(3,0), angsLeft(4,0), angsRight(4,0), angsWaist(3,0);
    bool rtn =  false;

    angsHead.clear();
    rtn  = calcHead(pts[9]-pts[8], pts[10]-pts[9], angsHead);
    if (!rtn) {
        std::cout<<"calcHead fail"<<std::endl;
    }
    angsHead[0] = angsHead[0];
    angsHead[1] = -angsHead[1];
    angsHead[2] = angsHead[2];

    angsWaist.clear();
    rtn = calcWaist(alignFrame, angsWaist);
    if (!rtn) {
        std::cout<<"calcWaist fail"<<std::endl;
    }
    angsWaist[0] = angsWaist[0];
    angsWaist[1] = -angsWaist[1];
    angsWaist[2] = angsWaist[2];

    angsLeft.clear();
    rtn = calcLeftArm(pts[12]-pts[11], pts[13]-pts[12], angsLeft);
    if (!rtn) {
        std::cout<<"calcLeftArm fail"<<std::endl;
    }
    angsLeft[0] = -angsLeft[0];
    angsLeft[1] = angsLeft[1];
    angsLeft[2] = -angsLeft[2];
    angsLeft[3] = -angsLeft[3];

    angsRight.clear();
    rtn = calcRightArm(pts[15]-pts[14], pts[16]-pts[15], angsRight);
    if (!rtn) {
        std::cout<<"calcRightArm fail"<<std::endl;
    }
    angsRight[0] = -angsRight[0];
    angsRight[1] = angsRight[1];
    angsRight[2] = -angsRight[2];
    angsRight[3] = -angsRight[3];

    /* assemble the angles, then return them */
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
    return aggregate;
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
    // check
    Eigen::Matrix3d Rx, Ry, Rz;
    Rx << 1,0,0,  0, cos(angs[0]), -sin(angs[0]), 0, sin(angs[0]), cos(angs[0]);
    Ry << cos(angs[1]), 0, sin(angs[1]), 0,1,0, -sin(angs[1]), 0, cos(angs[1]);
    Rz << cos(angs[2]), -sin(angs[2]), 0, sin(angs[2]), cos(angs[2]),0,  0, 0, 1 ;
    Eigen::Matrix3d delta = Rx*Ry*Rz - R;
    for (int i=0; i<delta.cols(); i++) {
        for (int j=0; j<delta.rows(); j++) {
            if (fabs(delta(i,j))>1e-5) {
                std::cout<<delta<<std::endl;
                return false;
            }
        }
    }
    return true;
}

bool  anglesFilter(const std::vector<double> &vec, std::vector<double> &res){
    static std::deque<std::deque<double>> slideWin;
    int vecSize = vec.size(); 
    std::deque<double> temp(std::begin(vec), std::begin(vec)+vecSize);
    slideWin.push_back(temp);
    if (slideWin.size()>winSize) {
        slideWin.pop_front();
    }

    int actualSize = slideWin.size();
    // std::cout<<"actualSize: "<<actualSize<<std::endl;
    double sum[vecSize] = {0};
    
    for (int i=0; i<actualSize; i++) {
        for (int j=0; j<vecSize; j++) {
            sum[j] += slideWin[i][j];
        }
    }
    res.clear();
    for(auto item:sum) {
        res.push_back(item/actualSize);
    }
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

//0 for L, 1 for R
double  restrictArmAndForward(int LR, std::vector<double> vv, Eigen::Vector3d aim) {
    //construct 
    double xBottom = LR ?  -0.6 : -0.6;
    double xUp = LR ?  3.14 : 3.14;
    double yBottom = LR ? -1.54 : -0.38;
    double yUp = LR ? 0.38 : 1.54;

    if (vv[0] > xUp) {
        vv[0] = xUp;
    } else if (vv[0]<xBottom){
        vv[0] = xBottom;
    }
    if (vv[1] > yUp) {
        vv[1] = yUp;
    } else if (vv[1]<yBottom){
        vv[1] = yBottom;
    }
    auto R = fuseR(vv[0], vv[1]);
    Eigen::Vector3d naturePos = LR ? Eigen::Vector3d(-cos(deltaAng), -sin(deltaAng), 0) : 
                                        Eigen::Vector3d(-cos(deltaAng), sin(deltaAng), 0);
    Eigen::Vector3d forwardVec = R*naturePos;
    return calcAngBetweenVec(forwardVec, aim);
}

/*********************************Left********************************************/
bool  calcLeftArm(const Eigen::Vector3d& vec1, const Eigen::Vector3d &vec2, std::vector<double>& angs) {
    angs.clear();
    angs.resize(4);
    
	Eigen::Matrix3d tilt;
	tilt << cos(deltaAng), -sin(deltaAng), 0,
			sin(deltaAng), cos(deltaAng), 0,
			0, 0, 1;

    //calculate shoulder
	Eigen::Vector3d v1 = tilt.inverse() * vec1 / vec1.norm();
	Eigen::Vector3d v2 = tilt.inverse() * vec2 / vec2.norm();
    v1(2) = (v1(2)<0) ? 0 : v1(2);
    v1 = v1 / v1.norm();
    
    Eigen::Vector3d v1Proj(v1(0), 0, v1(2));
    angs[0] = calcAngBetweenVec(v1Proj, Eigen::Vector3d(-1,0,0));
    Eigen::Matrix3d R1;
    R1 << cos(angs[0]), 0, sin(angs[0]),
            0, 1, 0,
            -sin(angs[0]), 0, cos(angs[0]);
    Eigen::Vector3d R1Vec = R1 * tilt.inverse() * Eigen::Vector3d(-1,0,0);
    angs[1] =  calcAngBetweenVec(R1Vec, v1);
    double temp1 = calcAngBetweenVec(v1, Eigen::Vector3d(0,1,0));
    double temp2 = PI/2.0 - deltaAng;
    angs[1] = (temp1>temp2) ? angs[1] : -angs[1];

    //calculate elbow
    angs[3] = calcAngBetweenVec(vec1, vec2);

	const Eigen::Matrix3d R = fuseR(angs[0], angs[1]);
	Eigen::Vector3d v1_inv = R.inverse() * v1;
	Eigen::Vector3d v2_inv = R.inverse() * v2;

	Eigen::Vector3d temp_vec = tilt * v1_inv;
    if ((temp_vec-Eigen::Vector3d(-1, 0, 0)).norm() > 1e-5) {
        std::cout<< "left test on v1 fail"<<std::endl;
        std::cout<<temp_vec<<std::endl;
        return false;
    }

	temp_vec = tilt * v2_inv;
	temp_vec(0) = 0.0;
	temp_vec(2) = temp_vec(2)<0 ? 0.0 : temp_vec(2);
	double ang2 = calcAngBetweenVec(Eigen::Vector3d(0,1,0), temp_vec);

    angs[2] = -(PI/2.0 - ang2);
	return true;
}

/*********************************Right********************************************/
bool  calcRightArm(const Eigen::Vector3d& vec1, const Eigen::Vector3d &vec2, std::vector<double>& angs) {
    angs.clear();
    angs.resize(4);
	Eigen::Matrix3d tilt;
	tilt << cos(-deltaAng), -sin(-deltaAng), 0,
			sin(-deltaAng), cos(-deltaAng), 0,
			0, 0, 1;

    //calculate shoulder
	Eigen::Vector3d v1 = tilt.inverse() * vec1 / vec1.norm();
	Eigen::Vector3d v2 = tilt.inverse() * vec2 / vec2.norm();
    v1(2) = v1(2)<0 ? 0 : v1(2);
    v1 = v1 / v1.norm();
    
    Eigen::Vector3d v1Proj(v1(0), 0, v1(2));
    // std::cout<<"temp out:\n"<<v1Proj<<std::endl;

    angs[0] = calcAngBetweenVec(v1Proj, Eigen::Vector3d(-1,0,0));
    Eigen::Matrix3d R1;
    R1 << cos(angs[0]), 0, sin(angs[0]),
            0, 1, 0,
            -sin(angs[0]), 0, cos(angs[0]);
    Eigen::Vector3d R1Vec = R1 * tilt.inverse() * Eigen::Vector3d(-1,0,0);
    angs[1] =  calcAngBetweenVec(R1Vec, v1);
    double temp1 = calcAngBetweenVec(v1, Eigen::Vector3d(0,1,0));
    double temp2 = PI/2.0 + deltaAng;
    angs[1] = (temp1>temp2) ? angs[1] : -angs[1];

    //calculate elbow
    angs[3] = calcAngBetweenVec(vec1, vec2);

	const Eigen::Matrix3d R = fuseR(angs[0], angs[1]);
	Eigen::Vector3d v1_inv = R.inverse() * v1;
	Eigen::Vector3d v2_inv = R.inverse() * v2;

	Eigen::Vector3d temp_vec = tilt * v1_inv;
    if ((temp_vec-Eigen::Vector3d(-1, 0, 0)).norm() > 1e-5) {
        std::cout<< "right test on v1 fail"<<std::endl;
        return false;
    }

	temp_vec = tilt * v2_inv;
	temp_vec(0) = 0.0;
	temp_vec(2) = temp_vec(2)<0 ? 0.0 : temp_vec(2);
	double ang2 = calcAngBetweenVec(Eigen::Vector3d(0,1,0), temp_vec);

    angs[2] = -(PI/2.0 - ang2);
	return true;	
}

/*********************************Head********************************************/
bool  calcHead(const Eigen::Vector3d &vec1, const Eigen::Vector3d &vec2, std::vector<double>&angs){
    // Eigen::Vector3d xBe = vec2/vec2.norm();
    Eigen::Vector3d xBe = vec1/vec1.norm() + vec2/vec2.norm();
    xBe = xBe/xBe.norm();
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

/*********************************Waist********************************************/
bool  calcWaist(const std::vector<Eigen::Vector3d> &pts, std::vector<double> &angs){
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


    Eigen::Vector3d legY = pts[1] - pts[4];
    legY[0] = 0;
    legY = legY / legY.norm();
    Eigen::Vector3d legX = Eigen::Vector3d::UnitX();
    Eigen::Vector3d legZ = legX.cross(legY);
    legR << legX(0), legY(0), legZ(0),
            legX(1), legY(1), legZ(1),
            legX(2), legY(2), legZ(2);

    // bool rtn = DCM2Euler(legR.inverse()*chestR, 1, angs);
    bool rtn = DCM2Euler(chestR, angs);
    if (!rtn) {
        std::cout<<"waist DCM2Euler check fail"<<std::endl;
    }
    return true;
}

/* correctPts makes points from world coordinator frame to chest frame, beacause head and arm is depend on chest frame
** x-up, y-right, z-front
*/
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

std::vector<Eigen::Vector3d>  correctFrame(const std::vector<Eigen::Vector3d> &pts){
    std::vector<Eigen::Vector3d> result;
    for (auto pt : pts) {
        result.push_back( Eigen::Vector3d(pt(2), pt(0), pt(1)) );
    }
    return result;
}

bool  angleRestrict(std::vector<double> &angles){
    bool inRange = true;
    assert(angles.size() == 21);
    for(auto &item : angles){
        if(item > 2*PI){
            item -= 2*PI;
        }else if(item < -2*PI){
            item += 2*PI;
        }
    }
    for(int i=0; i<angles.size(); i++){
        if(angles[i] > anglesStriction[i][1] ){
            inRange = false;
            std::cout<<"angle ceiling:\t"<<JointNames[i]<<"\t"<<angles[i]<<std::endl;
            angles[i] = anglesStriction[i][1];
        }else if( angles[i] < anglesStriction[i][0] ){
            inRange = false;
            std::cout<<"angle floor:\t"<<JointNames[i]<<"\t"<<angles[i]<<std::endl;
            angles[i] = anglesStriction[i][0];
        }
    }
    return inRange;
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

