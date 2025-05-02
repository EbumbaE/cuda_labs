#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CSC(call)  									                    \
do {											                       \
	cudaError_t res = call;							                   \
	if (res != cudaSuccess) {							               \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",	           \
				__FILE__, __LINE__, cudaGetErrorString(res)); 	       \
		exit(0);								                       \
	}										                           \
} while(0)

__global__ void sobel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int totalX = gridDim.x * blockDim.x;
    int totalY = gridDim.y * blockDim.y;
    int totalThr = totalX * totalY;

    int thrIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int thrIdy = blockDim.y * blockIdx.y + threadIdx.y;
    int thrId = thrIdy * totalX + thrIdx;

    int pixels = w * h;
    int pixelPerThr = (pixels + totalThr - 1) / totalThr;

    int start = thrId * pixelPerThr;
    int end = min(start + pixelPerThr, pixels);

    // printf("%d, %d, %d | %d, %d, %d | %d, %d | %d, %d \n", totalX, totalY, totalThr, thrIdx, thrIdy, thrId, pixels, pixelPerThr, start, end);

    int mx[3][3] = {
      {-1, 0, 1},
      {-2, 0, 2},
      {-1, 0, 1},
    };
    int my[3][3] = {
      {-1, -2, -1},
      { 0,  0,  0},
      { 1,  2,  1},
    };

    for (int k = start; k < end; k++) {
      int x = k % w;
      int y = k / w;

      float gradX = 0, gradY = 0;
      uchar4 p;

      for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            int px = max(min(x + i, w - 1), 0);
            int py = max(min(y + j, h - 1), 0);
            p = tex2D<uchar4>(tex, px, py);

            float yuv = 0.299 * p.x + 0.587 * p.y + 0.114 * p.z;
            gradX += yuv * mx[j + 1][i + 1];
            gradY += yuv * my[j + 1][i + 1];
        }
      }

      float grad = sqrt(gradX * gradX + gradY * gradY);
      grad = max(min(grad, 255.0), 0.0);

      out[y * w + x] = make_uchar4(grad, grad, grad, 0);
    }
}

int main() {
    char input[1024], output[1024];
    scanf("%s", input);
    scanf("%s", output);

    FILE *fp = fopen(input, "rb");
    if (fp == NULL) {
        fprintf(stderr, "ERROR: open input file");
        return -1;
    }

    int w, h;
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);

    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeMirror;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    sobel<<< dim3(3, 3), dim3(1, 1) >>>(tex, dev_out, w, h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    fp = fopen(output, "wb");
    if (fp == NULL) {
        fprintf(stderr, "ERROR: open output file");
        free(data);
        return -1;
    }
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}