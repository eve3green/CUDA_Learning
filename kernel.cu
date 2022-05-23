#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 2 //субматриця
int M = 5*16, K = 5*16;
int* A = new int[M * K]; // матриці в глобальній памя'ті
int* B = new int[M * K];
int* C = new int[M * K];

using namespace std;

__global__ void matrixAdd(int* A, int* B, int* C, int M, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = col * M + row;

    //множення
    if (col < M && row < K) {
        C[row * M + col] = 0;
        for (int k = 0; k < M; k++) {
            C[row * M + col] += A[row * M + k] * B[k * M + col];
        }
    }
}

int main() {

    A = new int[M * K];
    B = new int[M * K];
    C = new int[M * K];

    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++) {
            A[i * M + j] = 2;
            B[i * M + j] = 2;
            C[i * M + j] = 0;
        }

    int* dev_a, * dev_b, * dev_c; 

    int size = M * K * sizeof(int); //скільки треба виділити пам'яті

    cudaMalloc((void**)&dev_a, size); //виділення пам'яті
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);

    cudaMemcpy(dev_a, A, size, cudaMemcpyHostToDevice); //Перенос на пам'ять ГПУ
    cudaMemcpy(dev_b, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM); //число выделенных блоков
    dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x, (K + dimBlock.y - 1) / dimBlock.y); //размер и размерность сетки
    printf("dimGrid.x = %d, dimGrid.y = %d\n", dimGrid.x, dimGrid.y); //выводится размер сетки

    matrixAdd <<<dimGrid, dimBlock >>> (dev_a, dev_b, dev_c, M, K); //викликається ядро
    //cudaDeviceSynchronize(); 

    cudaMemcpy(C, dev_c, size, cudaMemcpyDeviceToHost);

    //вывод    результата
    printf("Result Matrix C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%d   ", C[i]);
        }
        printf("\n");
    }


    cudaFree(dev_a); //очистка пам'яті
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}
