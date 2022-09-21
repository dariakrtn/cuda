#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctime>

#define	N	(1024*1024)

__global__ void kernel( float* y )
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float x = 2.0f * 3.1415926f * (float)idx / (float)N;
    y[idx] = logf(x);
}

int main(int argc, char* argv[])
{
    //CPU

    int start2, time2;
    float* data2 = new float[N];

    start2 = clock();

    for (int idx2 = 0; idx2 < N; idx2++)
    {
        float x2 = 2.0f * 3.1415926f * (float)idx2 / (float)N;
        data2[idx2] = logf(x2);
        
    }
    
    time2 = clock() - start2;
    double time_CPU = time2;

    printf("\nCPU Time: %f milliseconds\n", time_CPU);

    //GPU

    float* a = new float[N];
    float* dev = NULL;

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaMalloc((void**)&dev, N * sizeof(float));

    kernel << <dim3((N / 512), 1), dim3(512, 1) >> > (dev);

    cudaMemcpy(a, dev, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("GPU Time: %.2f milliseconds\n", gpuTime);

    cudaFree(dev);


    return 0;

}
