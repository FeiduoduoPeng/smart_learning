#include "ske2ang.h"
int main() {
    double skeIn[51], angOut[21];
    for (int i=0; i<51; i++){
        skeIn[i] = log(i+1)*0.1+0.1;
    }
    skeleton2angles(skeIn, angOut);

    for (int i=0; i<21; i++) {
        std::cout<<angOut[i]<<"  ";
    }
    std::cout<<std::endl;
    return 0;
}