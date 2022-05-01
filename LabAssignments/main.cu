#include "baselineCUDAcore.h"
#include "baselineTensorCore.h"
#include "optimizedCUDAcore.h"
#include "optimizedTensorCore.h"
#include <iostream>
#include "helper.h"

int main(){

    
    float time_cpu = 0.0;
    float time_simple_fpu = 0.0;
    float time_simple_wmma = 0.0;

    float time_optimized_fpu = 0.0;
    float time_optimized_wmma = 0.0;

    std::cout<<"Matrix Size : MxNxK : " <<KgemmM << "x" << KgemmN <<"x" <<KgemmK<<std::endl;

    // std::cout<< "Warm-up Rounds = " << WARMUP_ROUNDS<<std::endl;

    // std::cout <<"TEST_ROUNDS = "<< TEST_ROUNDS <<std::endl;

    // time_cpu = measure_CPU_runtime();
    // std::cout << "average time of CPU baseline = " <<  time_cpu << " ms"<< std::endl;

    auto start = high_resolution_clock::now();  // Timer start
    time_simple_fpu = run_simple_fpu_gemm();
    std::cout << "average time of kernel simple_fpu = " <<  time_simple_fpu << " ms"<< std::endl;
    auto stop = high_resolution_clock::now();   // Timer stop
    auto milliseconds = duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Total runtime of the CNN layer = " <<  milliseconds.count() << " ms"<< std::endl;

    time_simple_wmma = run_simple_wmma_gemm();
    std::cout << "\naverage time of kernel simple_wmma = " <<  time_simple_wmma << " ms"<< std::endl;

    time_optimized_fpu =  run_optimized_fpu_gemm();
    std::cout << "Tiling size = " << K_tiles_CUDA << std::endl;
    std::cout << "average time of kernel optimized_fpu = " <<  time_optimized_fpu << " ms"<< std::endl;

    time_optimized_wmma = run_optimized_wmma_gemm();
    std::cout << "\naverage time of kernel optimized_wmma = " <<  time_optimized_wmma << " ms"<< std::endl;

    

}