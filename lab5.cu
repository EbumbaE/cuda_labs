#include "stdio.h"
#include "stdlib.h"

#define SHARED_SIZE 512
#define AXIS_BLOCKS 32768
#define BLOCK_SIZE 256
#define NUM_BLOCKS 256

#define uint int

#define CSC(call)                               \
do {                                         \
 cudaError_t res = call;                         \
 if (res != cudaSuccess) {                       \
  fprintf(stderr, "ERROR in %s:%d. Message: %s\n",   \
    __FILE__, __LINE__, cudaGetErrorString(res));  \
  exit(0);                                    \
 }                                            \
} while(0)

__global__ void upDownSweep(uint *input, uint *out, uint size) {
    extern __shared__ uint shared[];

    uint sliceEnd = SHARED_SIZE;
    uint thrOffset = 2 * (blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x);
    uint sweepIdx = 1;
    uint leftIdx = threadIdx.x;
    uint rightIdx = threadIdx.x + sliceEnd / 2;

    if (thrOffset + threadIdx.x + sliceEnd / 2 > size) {
        return;
    }

    shared[leftIdx] = input[thrOffset + leftIdx];
    shared[rightIdx] = input[thrOffset + rightIdx];

    for (uint d = sliceEnd/2; d > 0; d /= 2) {
        __syncthreads();
        if (threadIdx.x < d) {
            uint leftIdx = sweepIdx * (2 * threadIdx.x + 1) - 1;
            uint rightIdx = sweepIdx * (2 * threadIdx.x + 2) - 1;

            shared[rightIdx] += shared[leftIdx];
        }
        sweepIdx *= 2;
    }

    if (threadIdx.x == 0) {
        shared[sliceEnd - 1] = 0;
    }

    for (uint d = 1; d < sliceEnd; d *= 2) {
        sweepIdx /= 2;
        __syncthreads();
        if (threadIdx.x < d) {
            uint leftIdx = sweepIdx * (2 * threadIdx.x + 1) - 1;
            uint rightIdx = sweepIdx * (2 * threadIdx.x + 2) - 1;

            uint left = shared[leftIdx];
            shared[leftIdx] = shared[rightIdx];
            shared[rightIdx] += left;
        }
    }
    
    __syncthreads();

    out[thrOffset + rightIdx] = shared[rightIdx];
    if (threadIdx.x != 0) {
        out[thrOffset + leftIdx] = shared[leftIdx];
    }

    if (threadIdx.x == sliceEnd / 2 - 1) {
        out[thrOffset + rightIdx + 1] = out[thrOffset + rightIdx] + input[thrOffset + rightIdx];
    }
}

__global__ void getLastElements(uint *slice, uint *lastElements, uint n, uint size, uint border) {
    uint total = blockDim.x * gridDim.x;
    uint begin = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint idx = begin; idx < n; idx += total) {    
        lastElements[idx] = slice[min(idx * size + size, border - 1)];
    }
}

__global__ void blellochSub(uint *slice, uint *sub, uint n) {
    uint total = blockDim.x * gridDim.x;
    uint begin = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint idx = begin; idx < n; idx += total) {    
        slice[idx] = slice[idx] - sub[idx];
    }
}

__global__ void blellochSum(uint *prefSum, uint *diff, uint n) {
    uint diffIdx = blockIdx.y * gridDim.x + blockIdx.x;
    uint idx = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        prefSum[idx] += diff[diffIdx];
    }
}

void blellochScan(uint *bits, uint *prefSum, uint n) {
    if (n < SHARED_SIZE) {
        CSC(cudaMemset(prefSum, 0, (SHARED_SIZE + 1) * sizeof(uint)));
        CSC(cudaMemset(bits + n, 0, (SHARED_SIZE - n % SHARED_SIZE) * sizeof(uint)));
    
        upDownSweep<<<dim3(1, 1), SHARED_SIZE / 2, SHARED_SIZE * sizeof(uint)>>>(bits, prefSum, SHARED_SIZE);
        CSC(cudaGetLastError());
        return;
    } 
    if (n == SHARED_SIZE) {
        upDownSweep<<<dim3(1, 1), SHARED_SIZE / 2, n * sizeof(uint)>>>(bits, prefSum, SHARED_SIZE);
        CSC(cudaGetLastError());
        return;
    }

    uint blocksAmount = (SHARED_SIZE + n - 1) / SHARED_SIZE;
    uint gridY = blocksAmount / AXIS_BLOCKS;
    if (blocksAmount % AXIS_BLOCKS != 0) {
        gridY++;
    }
    uint gridX;
    if (blocksAmount <= AXIS_BLOCKS) {
        gridX = blocksAmount;
    } else {
        gridX = AXIS_BLOCKS;
    }

    if (n % SHARED_SIZE == 0) {
        upDownSweep<<<dim3(gridX, gridY), SHARED_SIZE / 2, SHARED_SIZE * sizeof(uint)>>>(bits, prefSum, n);
        CSC(cudaGetLastError());
    } else {
        uint newN = n + (SHARED_SIZE - n % SHARED_SIZE);
        CSC(cudaMemset(prefSum, 0, (newN + 1) * sizeof(uint)));
        CSC(cudaMemset(bits + n, 0, (newN - n) * sizeof(uint)));
        
        upDownSweep<<<dim3(gridX, gridY), SHARED_SIZE / 2, SHARED_SIZE * sizeof(uint)>>>(bits, prefSum, newN);
        CSC(cudaGetLastError());
    }

    uint *lastElements;
    CSC(cudaMalloc(&lastElements, (blocksAmount + (SHARED_SIZE - blocksAmount % SHARED_SIZE) + 1) * sizeof(uint)));

    getLastElements<<<BLOCK_SIZE, NUM_BLOCKS>>>(prefSum, lastElements, blocksAmount, SHARED_SIZE, n);
    CSC(cudaGetLastError());
    
    uint *newPrefSum;
    CSC(cudaMalloc(&newPrefSum, (blocksAmount + SHARED_SIZE - blocksAmount % SHARED_SIZE + 1) * sizeof(uint)));
    CSC(cudaMemset(newPrefSum, 0, (blocksAmount + SHARED_SIZE - blocksAmount % SHARED_SIZE + 1) * sizeof(uint)));

    blellochScan(lastElements, newPrefSum, blocksAmount);

    blellochSub<<<BLOCK_SIZE, NUM_BLOCKS>>>(newPrefSum + 1, lastElements, blocksAmount);
    CSC(cudaGetLastError());

    blellochSum<<<dim3(gridX, gridY), SHARED_SIZE>>>(prefSum + 1, newPrefSum + 1, n);
    CSC(cudaGetLastError());

    CSC(cudaFree(newPrefSum));
    CSC(cudaFree(lastElements));
}

__global__ void getBits(uint *slice, uint *bits, uint bit, uint n) {
    uint total = blockDim.x * gridDim.x;
    uint begin = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint idx = begin; idx < n; idx += total) {
        uint val = slice[idx];
        uint b = (val >> bit) & 1U;
        bits[idx] = b;
    }
}

__global__ void buildPosition(
    uint *prefSum, 
    uint *bits, 
    uint n
) {
    uint total = blockDim.x * gridDim.x;
    uint begin = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint idx = begin; idx < n; idx += total) {
        uint newPos;
        if (bits[idx] == 0) {
            newPos = idx - prefSum[idx];
        } else {
            newPos = n + prefSum[idx] - prefSum[n];
        }

        bits[idx] = newPos;
    }
}

__global__ void applyPosition(
    uint *inputSlice, 
    uint *outputSlice, 
    uint *positions, 
    uint n
) {
    uint total = blockDim.x * gridDim.x;
    uint begin = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint idx = begin; idx < n; idx += total) {
        outputSlice[positions[idx]] = inputSlice[idx];
    }
}

void radixSort(uint *data, uint n) {
    uint *prefSum;
    uint *bits;
    CSC(cudaMalloc(&prefSum, (n + SHARED_SIZE - n % SHARED_SIZE + 1) * sizeof(uint)));
    CSC(cudaMalloc(&bits, (n + SHARED_SIZE - n % SHARED_SIZE) * sizeof(uint)));

    uint *cudaSlice;
    CSC(cudaMalloc(&cudaSlice, n * sizeof(uint)));
    CSC(cudaMemcpy(cudaSlice, data, n * sizeof(uint), cudaMemcpyHostToDevice));
    
    for (uint bit = 0; bit < 32; bit++) {
        getBits<<<BLOCK_SIZE, NUM_BLOCKS>>>(cudaSlice, bits, bit, n);
        CSC(cudaGetLastError());

        CSC(cudaMemset(prefSum, 0, (n + 1) * sizeof(uint)));
        blellochScan(bits, prefSum, n);

        buildPosition<<<BLOCK_SIZE, NUM_BLOCKS>>>(prefSum, bits, n);
        CSC(cudaGetLastError());

        CSC(cudaMemcpy(prefSum, cudaSlice, n * sizeof(uint), cudaMemcpyDeviceToDevice));

        applyPosition<<<BLOCK_SIZE, NUM_BLOCKS>>>(prefSum, cudaSlice, bits, n);
        CSC(cudaGetLastError());
    }

    CSC(cudaMemcpy(data, cudaSlice, n * sizeof(uint), cudaMemcpyDeviceToHost));

    CSC(cudaFree(prefSum));
    CSC(cudaFree(bits));
    CSC(cudaFree(cudaSlice));
}

int main() {
    uint n;
    if (fread(&n, sizeof(uint), 1, stdin) != 1) {
        fprintf(stderr, "ERROR: scan N");
        return 0;
    }

    if (n < 1) {
      return 0;
    }

    uint *data = (uint*)malloc(n * sizeof(uint));
    if (!data) {
        fprintf(stderr, "ERROR: malloc data");
        return 0;
    }

    if (fread(data, sizeof(uint), n, stdin) != n) {
        fprintf(stderr, "ERROR: read array");
        free(data);
        return 0;
    }

    radixSort(data, n);

    fwrite(data, sizeof(uint), n, stdout);

    free(data);

    return 0;
}