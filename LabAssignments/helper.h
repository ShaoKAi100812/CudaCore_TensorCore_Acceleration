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
#include <chrono>

#include "configs.h"

#pragma once
using namespace std::chrono;
#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)



// initialize the host matrix
template<typename datatype>
void init_host_matrices(datatype *a, int Rows, int Cols) {

    for (int i = 0; i < Rows; i++) {
      for (int j = 0; j < Cols; j++) {
        a[i * Cols + j] =  datatype( (i * Cols + j) % 3); 
      }
    }
}

template<typename datatype>
void print_matrix(datatype *a,int Rows, int Cols){

    for(int i=0;i<Rows;i++){
        for(int j=0; j< Cols; j ++){
            std::cout<<a[i * Cols + j] <<", ";
        }
        std::cout<<std::endl;
    }

}




void check_errors(float* h_out, float* d_out, int size){

    float errors = 0;
    for(int i =0;i < size; i++){
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

}



//CPU baseline, for numeric check
template<typename datatype>
void gemmCPU(float* D, float* C, datatype* A, datatype* B, int gemmM, int gemmN ,int gemmK, float alpha, float beta){
    //A*B
    //Matrix A is row-major, Matrix B is col-major 
    for (unsigned int i = 0; i < gemmM; ++i)
    {
        for (unsigned int j = 0; j < gemmN; ++j) {
            float sum = 0.0;
            for (unsigned int k = 0; k < gemmK; ++k) {
                float a = A[i * gemmK + k];
                float b = B[j * gemmK + k];
                // float b = B[k * gemmN + j];     //wrong position of row and column
                sum += a * b;
            }
            D[i * gemmN + j] = alpha * sum + beta * C[i * gemmN + j];
        }
    }
}


float measure_CPU_runtime(){
    cpu_type *A_h = nullptr; 
    cpu_type *B_h = nullptr;
    float *C_h = nullptr;
    float *D_cpu_ref = nullptr;

    A_h = (cpu_type *)malloc(sizeof(cpu_type) * KgemmM * KgemmK);
    B_h = (cpu_type *)malloc(sizeof(cpu_type) * KgemmK * KgemmN);
    C_h = (float *)malloc(sizeof(float) * KgemmM * KgemmN);
    D_cpu_ref = (float *)malloc(sizeof(float) * KgemmM * KgemmN);


    init_host_matrices(A_h, KgemmM, KgemmK);
    init_host_matrices(B_h, KgemmK, KgemmN);
    init_host_matrices(C_h, KgemmM, KgemmN);


    float alpha = 1.0;
    float beta = 1.0;

    // warm up machine
    for (int i = 0;i < WARMUP_ROUNDS;i++ ){

        gemmCPU(D_cpu_ref, C_h, A_h,  B_h,  KgemmM, KgemmN, KgemmK, alpha, beta);

    }

    float total = 0;
    for (int i = 0;i < TEST_ROUNDS;i++ ){

        auto start = high_resolution_clock::now();


       
        gemmCPU(D_cpu_ref, C_h, A_h,  B_h,  KgemmM, KgemmN, KgemmK, alpha, beta);
    
        auto stop = high_resolution_clock::now();
    
    
        
        
        auto milliseconds = duration_cast<std::chrono::milliseconds>(stop - start);
        //cudaEventElapsedTime(&milliseconds, start, stop);
        total += milliseconds.count();

    }


    float average_time = total/TEST_ROUNDS;

    free(A_h);
    free(B_h);
    free(C_h);
    free(D_cpu_ref);

    return average_time;

}

void transpose(fpu_type *src, fpu_type *dst, const int N, const int M) {
    #pragma omp parallel for        // open threads on cpu, omg
    for(int n = 0; n<N*M; n++) {
        int i = n/N;
        int j = n%N;
        dst[n] = src[M*j + i];
    }
}