#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void reverse(double *slice, int n) {
    int total = gridDim.x * blockDim.x;
    int begin = blockIdx.x * blockDim.x + threadIdx.x;

    for (int left = begin; left < n / 2; left += total) {
        printf("%d\n", left);
        int right = n - left - 1;
        double x = slice[left];
        slice[left] = slice[right];
        slice[right] = x;
    }
}

int main() {
    int n;
    if (scanf("%d", &n) != 1) {
        fprintf(stderr, "ERROR: read array size\n");
        return 0;
    }

    if (n < 0) {
        fprintf(stderr, "ERROR: invalid array size\n");
        return 0;
    }

    if (n == 0) {
        return 0;
    }

    double *slice = (double*) malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        if (scanf("%lf", &slice[i]) != 1) {
            fprintf(stderr, "ERROR: read array element");
            free(slice);
            return 0;
        }
    }

    double *cudaSlice;
    if (cudaMalloc((void**)&cudaSlice, n * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "ERROR: allocate GPU memory\n");
        free(slice);
        return 0;
    }

    cudaMemcpy(cudaSlice, slice, n * sizeof(double), cudaMemcpyHostToDevice);

    reverse<<<3, 3>>>(cudaSlice, n);

    cudaMemcpy(slice, cudaSlice, n * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%.10lf ", slice[i]);
    }
    printf("\n");

    free(slice);
    cudaFree(cudaSlice);

    return 0;
}