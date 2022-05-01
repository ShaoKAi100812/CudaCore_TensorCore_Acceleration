#include <stdio.h>
#include <iostream>
using namespace std;


int main(){

    std::cout<<"measure the runtime of im2col cpu"<<std::endl;
    //im2col_cpu(int height,int width, int channels_in, int kernel_h, int kernel_w,  int pad_h = 1 , int pad_w = 1, int stride_h = 1, int stride_w = 1 )
    for(int r=1, s=1; r<=11; r+=2, s+=2){
        cout << r << '\t'<< s << endl;
        // im2col_cpu(32, 32, 1, r, s, 0, 0, 1, 1)
    }
    
}