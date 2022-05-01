#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <iostream>
#include <random>
#include <cuda_fp16.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>


using namespace nvcuda;

//


// read codes in LabAssignments/helper.h to study how to meaure the runtime of a CPU function (e.g. gemmCPU), and then measure yourself.

template<typename datatype>
void im2col_cpu(datatype* image_in, datatype* col_out,  int height, int width, int channels_in, int kernel_h, int kernel_w,  int pad_h = 1 , int pad_w = 1, int stride_h = 1, int stride_w = 1 ){

    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int channels_col = channels_in * kernel_h * kernel_w; 

    // col : [height_col* width_col , channels_col]
    for (int c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / (kernel_h * kernel_w);
    
        for (int h = 0; h < height_col; ++h) {
          for (int w = 0; w < width_col; ++w) {
            int h_pad = h*stride_h - pad_h + h_offset;
            int w_pad = w*stride_w - pad_w + w_offset;
            if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width) {
              col_out[(c*height_col+h) * width_col + w] =
                image_in[(c_im * height + h_pad) * width + w_pad];
            } else {
              col_out[(c*height_col+h) * width_col + w] = 0;
            }
          }
        }
      }

};




void check_errors(float* h_out, float* d_out, int size){

    float errors = 0;
    for(int i =0;i < size;i++){
        float abs_error_=  abs(h_out[i] - d_out[i]);
        if(abs_error_ >0.001 ){
            std::cout << "abs_error_ = " << abs_error_ << " at " << i <<std::endl;
            std::cout << "CPU ref result = " << h_out[i] << ";, GPU result = "<< d_out[i] <<std::endl;
            std::cout << " error: Check your CUDA implementation! the result is not numerically correct compared to C program" << std::endl;
            return;
        }
        errors += abs_error_;
    }
    float avg_err = errors/size;
    //std::cout << "average errors = " << errors/size<<std::endl;
    if(avg_err > 0.001){
        std::cout << "average errors = " << avg_err<<std::endl;
        std::cout << " error: Check your CUDA implementation! the result is not numerically correct compared to C program" << std::endl;
        
    }else{
        std::cout << "numeric check passed " <<std::endl;
    }

};


// Todo , intializaton of activation tensor 
// B: batch size, H/W height/width of activation. C: #channel

// template<typename datatype>
// void init_host_activation(datatype *Activation, int B, int H, int W, int C){
// // TODO
  
// };

// // Todo, intializaton of weight tensor 
// // K: #output channels, R/S: Height/Withd of kernel, C:#input channel
// template<typename datatype>
// void init_host_weight(datatype *Weight, int K, int R, int S, int C){
// // TODO

  
// };
