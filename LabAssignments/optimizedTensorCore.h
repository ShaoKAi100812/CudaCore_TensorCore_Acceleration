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


__global__ void optimized_wmma_gemm(half *a, half *b, float *c, float *d, int gemmM, int gemmN, int gemmK, float alpha, float beta){

    // implement the optimized tensor core kernel using shared memory and loop unrolling
    // some step hints for you. Feel free to follow different steps
    // step 1. creat shared memory buffer
    __shared__ half A_tile[16*16];
    __shared__ half B_tile[16*16];
    // shorten parameters for clean re-use
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // calculate current row and column of matrix C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Tile using a 2D grid
    // compute global warp id
    int warpM = ( blockIdx.x * blockDim.x + threadIdx.x) / 32; // warpSize = 32
    int warpN = ( blockIdx.y * blockDim.y + threadIdx.y);
    // Declare the fragments
    wmma :: fragment < wmma :: matrix_a , 16, 16, 16, half , wmma :: col_major > a_frag ;
    wmma :: fragment < wmma :: matrix_b , 16, 16, 16, half , wmma :: col_major > b_frag ;
    wmma :: fragment < wmma :: accumulator , 16, 16, 16, float > acc_frag ;
    wmma :: fragment < wmma :: accumulator , 16, 16, 16, float > c_frag ;
    wmma :: fill_fragment ( acc_frag , 0.0f );

    // step 2: main loop over K
    for (int i = 0; i < gemmK ; i += 16) {
        int aCol = i;
        int aRow = warpM * 16; // offset
        int bCol = warpN * 16; // offset
        int bRow = i;
        // step 1: load data from global mem to shared mem
        if (tx + 1 <= 16 && ty + 1 <= 16){
            A_tile[ty*16+tx] = a[row * gemmK + (i + tx)];
            B_tile[tx*16+ty] = b[col * gemmK + (i + ty)];
        }
        __syncthreads();
        // step 2: load data from shared mem to wmma fragments
        if ( aRow < gemmM && aCol < gemmK && bRow < gemmK && bCol < gemmN ){
            wmma :: load_matrix_sync (a_frag , A_tile , 16 );   // Bitwise operation
            wmma :: load_matrix_sync (b_frag , B_tile , 16 );
            // step 3: compute partial results, store immtermediate to shared mem if necessary  
            // Perform the matrix multiplication
            wmma :: mma_sync ( acc_frag , a_frag , b_frag , acc_frag );
        }
    }    
    //step 3: addtional computations: adding matrix C 
    int cCol = warpN * 16;
    int cRow = warpM * 16;
    if ( cRow < gemmM && cCol < gemmN ) {
        wmma :: load_matrix_sync (c_frag , c + cCol + cRow * gemmN , gemmN , wmma :: mem_row_major );
        for (int i = 0; i < c_frag.num_elements ; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
    //step 4: store back the final results to global memory
        // Store the output
        wmma :: store_matrix_sync (d + cCol + cRow * gemmN , c_frag , gemmN , wmma :: mem_row_major );
    }
}


float run_optimized_wmma_gemm(){

    // pointer to host
    half *A_h = nullptr;
    half *B_h = nullptr;
    float *C_h = nullptr;
    float *D_h = nullptr;

    float *D_cpu_ref = nullptr;


    // pointer for device data
    half *A_d = nullptr;
    half *B_d = nullptr;
    float *C_d = nullptr;
    float *D_d = nullptr;
    
    A_h = (half *)malloc(sizeof(half) * KgemmM * KgemmK);
    B_h = (half *)malloc(sizeof(half) * KgemmK * KgemmN);
    C_h = (float *)malloc(sizeof(float) * KgemmM * KgemmN);
    D_h = (float *)malloc(sizeof(float) * KgemmM * KgemmN);

    D_cpu_ref = (float *)malloc(sizeof(float) * KgemmM * KgemmN);

    // initialize host daa
    init_host_matrices(A_h, KgemmM, KgemmK);
    init_host_matrices(B_h, KgemmK, KgemmN);
    init_host_matrices(C_h, KgemmM, KgemmN);
    
    


    // create CUDA global memory
    checkKernelErrors(cudaMalloc( reinterpret_cast<void **>(&A_d), KgemmM * KgemmK * sizeof(half) ) );
    checkKernelErrors(cudaMalloc( reinterpret_cast<void **>(&B_d), KgemmK * KgemmN * sizeof(half) ) );
    checkKernelErrors(cudaMalloc( reinterpret_cast<void **>(&C_d), KgemmM * KgemmN * sizeof(float) )); 
    checkKernelErrors(cudaMalloc( reinterpret_cast<void **>(&D_d), KgemmM * KgemmN * sizeof(float) )); 

    // // Transpose
    // auto start_1 = high_resolution_clock::now();  // Timer start
    // transpose(B_h, B_h, KgemmN, KgemmK);
    // auto stop_1 = high_resolution_clock::now();   // Timer stop
    // auto milliseconds_1 = duration_cast<std::chrono::milliseconds>(stop_1 - start_1);
    // std::cout << "\nRuntime of matrix transpose = " <<  milliseconds_1.count() << " ms"<< std::endl;
    
    checkKernelErrors(cudaMemcpy(A_d, A_h, sizeof(half) * KgemmM * KgemmK, cudaMemcpyHostToDevice));
    checkKernelErrors(cudaMemcpy(B_d, B_h, sizeof(half) * KgemmK * KgemmN, cudaMemcpyHostToDevice));
    checkKernelErrors(cudaMemcpy(C_d, C_h, sizeof(float) * KgemmM * KgemmN, cudaMemcpyHostToDevice));
    checkKernelErrors(cudaMemset(D_d, 0, sizeof(float) * KgemmM * KgemmN));

    float alpha = 1.0;
    float beta = 1.0;

    // kernel configs
    // note the differences between tensor core and cuda core
    dim3 gridDim;
    dim3 blockDim;

    blockDim.x = 128; //M_tiles_TC/16 * 32; // 16: each wmma computes 16x16, 32: number of threads per warp is 32
    blockDim.y = 4;//N_tiles_TC/16; // how many warps among tiledim N


    gridDim.x = (KgemmM + (16 * blockDim.x / 32 - 1)) / (16 * blockDim.x / 32); //KgemmN/N_tiles_TC; // gemmN must be divisible by N_tiles_CUDA

    gridDim.y = (KgemmM + 16 * blockDim.y - 1) / (16 * blockDim.y); //KgemmM/M_tiles_TC; // gemmM must be divisible by M_tiles_CUDA

   
    // warm up machine
    for (int i = 0;i < WARMUP_ROUNDS;i++ ){


        //half *a, half *b, float *c, float *d, int gemmM,int gemmN, int gemmK, float alpha, float beta
        checkKernelErrors(
                     ( optimized_wmma_gemm<<<gridDim,blockDim>>>(A_d , B_d , C_d , D_d , KgemmM , KgemmN , KgemmK , alpha , beta) )
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

        //half* A, half* B, float* C, float* D, int gemmM, int gemmN ,int gemmK,float alpha, float beta
        checkKernelErrors(
                     ( optimized_wmma_gemm<<<gridDim,blockDim>>>(A_d , B_d , C_d , D_d , KgemmM , KgemmN , KgemmK , alpha , beta) )
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

    // std::cout<<"GPU matrix" <<std::endl;
    // print_matrix(D_h,KgemmM,KgemmN);
    
    // std::cout<<"CPU matrix" <<std::endl;
    // print_matrix(D_cpu_ref,KgemmM,KgemmN);
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
