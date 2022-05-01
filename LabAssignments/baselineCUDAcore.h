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
#include "helper.h"

#pragma once

using namespace nvcuda;


/*
Naive CUDA Core (floating point unit(fpu) or SIMT units) implementation for D=alpha * A*B + beta*C
*/
__global__ void simple_fpu_gemm(fpu_type* A, fpu_type* B, float* C, float* D, int gemmM, int gemmN ,int gemmK, float alpha, float beta){
    int m = blockIdx.y * blockDim.y + threadIdx.y;      //row
    int n = blockIdx.x * blockDim.x + threadIdx.x;      //column
    fpu_type accu = 0.0;
    for (int k=0; k<gemmK; ++k){
        accu += alpha * static_cast<float>(A[m*gemmK+k]) * static_cast<float>(B[n*gemmK+k]);
    }
    D[m*gemmN+n] = static_cast<float>(accu) + beta * C[m*gemmN+n];
}


float run_simple_fpu_gemm(){
    // pointer to host
    fpu_type *A_h = nullptr;
    fpu_type *B_h = nullptr;
    float *C_h = nullptr;
    float *D_h = nullptr;

    float *D_cpu_ref = nullptr;


    // pointer for device data
    fpu_type *A_d = nullptr;
    fpu_type *B_d = nullptr;
    float *C_d = nullptr;
    float *D_d = nullptr;
    



    A_h = (fpu_type *)malloc(sizeof(fpu_type) * KgemmM * KgemmK);
    B_h = (fpu_type *)malloc(sizeof(fpu_type) * KgemmK * KgemmN);
    C_h = (float *)malloc(sizeof(float) * KgemmM * KgemmN);
    D_h = (float *)malloc(sizeof(float) * KgemmM * KgemmN);

    D_cpu_ref = (float *)malloc(sizeof(float) * KgemmM * KgemmN);

    // initialize host daa
    init_host_matrices(A_h, KgemmM, KgemmK);
    init_host_matrices(B_h, KgemmK, KgemmN);
    init_host_matrices(C_h, KgemmM, KgemmN);
    
    


    // create CUDA global memory
    checkKernelErrors(cudaMalloc( reinterpret_cast<void **>(&A_d), KgemmM * KgemmK * sizeof(fpu_type) ) );
    checkKernelErrors(cudaMalloc( reinterpret_cast<void **>(&B_d), KgemmK * KgemmN * sizeof(fpu_type) ) );
    checkKernelErrors(cudaMalloc( reinterpret_cast<void **>(&C_d), KgemmM * KgemmN * sizeof(float) )); 
    checkKernelErrors(cudaMalloc( reinterpret_cast<void **>(&D_d), KgemmM * KgemmN * sizeof(float) )); 


    auto start_1 = high_resolution_clock::now();  // Timer start
    checkKernelErrors(cudaMemcpy(A_d, A_h, sizeof(fpu_type) * KgemmM * KgemmK, cudaMemcpyHostToDevice));
    checkKernelErrors(cudaMemcpy(B_d, B_h, sizeof(fpu_type) * KgemmK * KgemmN, cudaMemcpyHostToDevice));
    checkKernelErrors(cudaMemcpy(C_d, C_h, sizeof(float) * KgemmM * KgemmN, cudaMemcpyHostToDevice));
    checkKernelErrors(cudaMemset(D_d, 0, sizeof(float) * KgemmM * KgemmN));
    auto stop_1 = high_resolution_clock::now();   // Timer stop
    auto milliseconds_1 = duration_cast<std::chrono::milliseconds>(stop_1 - start_1);
    std::cout << "\nRuntime of memory copying from host to device = " <<  milliseconds_1.count() << " ms"<< std::endl;

    float alpha = 1.0;
    float beta = 1.0;

    // kernel configs
    dim3 gridDim;
    dim3 blockDim;

    blockDim.x = N_tiles_CUDA;
    blockDim.y = M_tiles_CUDA;

    gridDim.x = KgemmN/N_tiles_CUDA; // gemmN must be divisible by N_tiles_CUDA
    gridDim.y = KgemmM/M_tiles_CUDA; // gemmM must be divisible by M_tiles_CUDA


   
    // warm up machine
    for (int i = 0;i < WARMUP_ROUNDS;i++ ){


        checkKernelErrors(
                     ( simple_fpu_gemm<<<gridDim,blockDim>>>(A_d , B_d , C_d , D_d , KgemmM , KgemmN , KgemmK , alpha , beta) )
                        );

    
        cudaDeviceSynchronize();

    }
    //create CUDA event to measure the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // test and take average time
    float total = 0;
    for (int i = 0;i < TEST_ROUNDS;i++ ){

        cudaEventRecord(start);



        checkKernelErrors(
                     ( simple_fpu_gemm<<<gridDim,blockDim>>>(A_d , B_d , C_d , D_d , KgemmM , KgemmN , KgemmK , alpha , beta) )
                        );

    
        cudaDeviceSynchronize();
    
        cudaEventRecord(stop);
    
    
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total += milliseconds;

    }

    
    float average_time = total/TEST_ROUNDS;
    //std::cout << "Tiled Kernel Execution time(without data transfer time) = " <<  average_time << " ms"<< std::endl;

    // Copy result back to host (CPU) 
    checkKernelErrors( cudaMemcpy(D_h, D_d, sizeof(float) * KgemmM * KgemmN, cudaMemcpyDeviceToHost) );

    // comput CPU baseline as ref
    #if( CPU_DEBUG == 1) 
    gemmCPU(D_cpu_ref, C_h, A_h,  B_h,  KgemmM, KgemmN, KgemmK, alpha, beta);
   
    //check numeric errors
    check_errors(D_cpu_ref,D_h,KgemmM*KgemmN);
    #endif


    free(A_h);
    free(B_h);
    free(C_h);
    free(D_h);
    free(D_cpu_ref);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(D_d);

    return average_time;
}