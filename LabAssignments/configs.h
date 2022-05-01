#include <cuda_fp16.h>
#include <cuda.h>
#pragma once
// configs problem size, start with small size for debuging 
constexpr int KgemmM = 4096;        //4096
constexpr int KgemmN = 4096;
constexpr int KgemmK = 4096;

// 1 means enable numeric check, 0 means disable to save time,
// note, you have to do numeric check first with small proble size (e.g. 256)
// then you set this as 0 and test on larger problem size (e.g. 512,1024,2048,4096)
#define CPU_DEBUG 0

// configs test rounds
constexpr int TEST_ROUNDS = 1;  //20 
constexpr int WARMUP_ROUNDS = 0; //3




//configs CUDA core implementtaion
// each thread computes one output, so there are M_tiles_CUDA*N_tiles_CUDA threads for teach threadblock. The max threads of each block should be less than 1024.
constexpr int M_tiles_CUDA = 32;
constexpr int N_tiles_CUDA = 32;
// split over K to reduce the shared memory size
constexpr int K_tiles_CUDA = 32;


// configs tensor core implementation
// each warp computs a 16*16*16 tiled gemm. each warp has 32 threads. So in total there are M_tiles_TC/16 * N_tiles_TC/16 * 32 threads for each block 
constexpr int M_tiles_TC = 64; // 64/16 = 4
constexpr int N_tiles_TC = 64;  // 64/16 = 4 // 4 * 4 * 32 = 512
// split over K to reduce the shared memory size
constexpr int K_tiles_TC = 64; 

// switch between half and float for cpu datatype (Matrix A and B)
typedef half cpu_type; 
// typedef float cpu_type; 


// switch between half and float for CUDA core(FPU) datatype (Matrix A and B) 
typedef half fpu_type; 
// typedef float fpu_type; 
