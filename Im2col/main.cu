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
#include "im2col.h"
#include <chrono>
using namespace std::chrono;

int TEST_ROUNDS = 20;

void init_host_matrices(float *a, int Rows, int Cols) {

    for (int i = 0; i < Rows; i++) {
      for (int j = 0; j < Cols; j++) {
        a[i * Cols + j] =  float( (i * Cols + j) % 3); 
      }
    }
}

int main(){
    float total = 0;
    float average_time = 0;
    // Initialize image_in
    float *image_in = nullptr; 
    image_in = (float *)malloc(sizeof(float) * 32 * 32);
    // Calculate the average time consumption of different R and S
    for(int r=1, s=1; r<=11; r+=2, s+=2){
        total = 0;
        average_time = 0;
        init_host_matrices(image_in, 32, 32);
        // Calculate the average of 20 times of im2col() transformation
        for (int i = 0 ;i<TEST_ROUNDS; i++){   
            auto start = high_resolution_clock::now();  // Timer start
            for (int n = 0; n < 64; n++){         // N = 64 
                // Initialize col_out
                float *col_out = nullptr;
                col_out = (float *)malloc(sizeof(float) * (32-r+1) * (32-s+1) * r * s);
                im2col_cpu(image_in, col_out, 32, 32, 1, r, s, 0, 0, 1, 1);  // channel_in = 1
                free(col_out);
            }
            auto stop = high_resolution_clock::now();   // Timer stop
            auto milliseconds = duration_cast<std::chrono::milliseconds>(stop - start);
            total += milliseconds.count();
        }
        average_time = total/TEST_ROUNDS;
        std::cout<<"measure the runtime of im2col cpu (R=S="<<r<<") is "<<average_time<<" milliseconds"<<std::endl;
    }
    free(image_in);
}
