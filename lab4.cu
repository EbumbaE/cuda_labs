#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define CSC(call)  									                    \
do {											                              \
	cudaError_t res = call;							                  \
	if (res != cudaSuccess) {							                \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",	  \
				__FILE__, __LINE__, cudaGetErrorString(res)); 	\
		exit(0);								                            \
	}										                                  \
} while(0)

struct comparator {
	__host__ __device__ bool operator()(double a, double b) {
		return fabs(a) < fabs(b);
	}
};

__global__ void swapRows(double *matrix, int i, int j, int n) {
	int begin = blockDim.x * blockIdx.x + threadIdx.x;
	int total = blockDim.x * gridDim.x;

	double tmp;
	for (int idx = begin; idx < n + 1; idx += total) {
		tmp = matrix[i + n * idx];
		matrix[i + n * idx] = matrix[j + n * idx];
		matrix[j + n * idx] = tmp;
	}
}

__global__ void gaussian(double *matrix, int col, int n) {
	int beginI = blockDim.x * blockIdx.x + threadIdx.x + col + 1;
	int beginJ = blockDim.y * blockIdx.y + threadIdx.y + col + 1;
	int totalI = blockDim.x * gridDim.x;
	int totalJ = blockDim.y * gridDim.y;

    if (fabs(matrix[col + n * col]) < 1e-10) {
        return;
    }

	for (int i = beginI; i < n; i += totalI) {
		for (int j = beginJ; j <= n; j += totalJ) {
            double factor = matrix[i + n * col] / matrix[col + n * col];
			matrix[i + n * j] -= factor * matrix[col + n * j];
		}
	}
}

void solveGaussian(double *matrix, int n) {
    double *matrixCuda;
    CSC(cudaMalloc(&matrixCuda, n * (n + 1) * sizeof(double)));
    CSC(cudaMemcpy(matrixCuda, matrix, n * (n + 1) * sizeof(double), cudaMemcpyHostToDevice));

    for (int col = 0; col < n; col++) {
        thrust::device_ptr<double> startPtr(matrixCuda + (col + n * col));
        thrust::device_ptr<double> endPtr(matrixCuda + (n + n * col));

        auto iter = thrust::max_element(startPtr, endPtr, comparator());
        int pivot = (iter - startPtr) + col;
        if (pivot != col) {
            swapRows<<<1024, 32>>>(matrixCuda, pivot, col, n);
            CSC(cudaGetLastError());
            CSC(cudaDeviceSynchronize());
        }

        gaussian<<<dim3(32, 32), dim3(32, 32)>>>(matrixCuda, col, n);
        CSC(cudaGetLastError());
        CSC(cudaDeviceSynchronize());
    }

    CSC(cudaMemcpy(matrix, matrixCuda, n * (n + 1) * sizeof(double), cudaMemcpyDeviceToHost));
    CSC(cudaFree(matrixCuda));
}

int main() {
    int n;
    scanf("%d", &n);
    if (n <= 0) {
        fprintf(stderr, "ERROR: invalid n");
        return 0;
    }

    double *matrix = (double*) malloc(n * (n + 1) * sizeof(double));
    if (matrix == NULL) {
        fprintf(stderr, "ERROR: malloc matrix");
        return 0;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &matrix[i + n * j]);
        }
    }

    for (int i = 0; i < n; i++) {
        scanf("%lf", &matrix[i + n * n]);
    }

    solveGaussian(matrix, n);

    double *result = (double*) malloc(n * sizeof(double));
    if (result == NULL) {
        fprintf(stderr, "ERROR: malloc result");
        free(matrix);
        return 0;
    }

    for (int i = n - 1; i >= 0; i--) {
		result[i] = matrix[i + n * n];
		for (int j = i + 1; j < n; j++) {
			result[i] -= result[j] * matrix[i + j * n];
		}
        if (fabs(matrix[i + i * n]) < 1e-10) {
            result[i] = 0.0;
        } else {
    		result[i] /= matrix[i + i * n];
        }
	}

    for (int i = 0; i < n; i++) {
        printf("%.10e", result[i]);
        if (i < n - 1) {
            printf(" ");
        }
    }
    printf("\n");

    free(matrix);
    free(result);
    return 0;
}