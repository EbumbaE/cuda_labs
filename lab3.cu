#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CSC(call)  									                    \
do {											                              \
	cudaError_t res = call;							                  \
	if (res != cudaSuccess) {							                \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",	  \
				__FILE__, __LINE__, cudaGetErrorString(res)); 	\
		exit(0);								                            \
	}										                                  \
} while(0)

struct Color {
  float x, y, z;
};

__global__ void spectralMethod(uchar4 *pixels, int w, int h, int nc, const Color *cudaNormalized) {
    int total = gridDim.x * blockDim.x;
    int begin = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = begin; i < w * h; i += total) {
        uchar4 pixel = pixels[i];
        float x = (float)pixel.x;
        float y = (float)pixel.y;
        float z = (float)pixel.z;
        float len = sqrt(x * x + y * y + z * z);

        // printf("%d, %d| %d \n", total, begin, i);

        float norm = x * cudaNormalized[0].x + y * cudaNormalized[0].y + z * cudaNormalized[0].z;
        float similar = norm / len;
        float maxSimilar = similar;
        int maxClass = 0;

        for (int cl = 1; cl < nc; cl++) {
            norm = x * cudaNormalized[cl].x + y * cudaNormalized[cl].y + z * cudaNormalized[cl].z;
            similar = norm / len;

            if (similar > maxSimilar) {
                maxSimilar = similar;
                maxClass = cl;
            }
        }

        pixels[i].w = maxClass;
    }
}

int main() {
    char input[1024], output[1024];
    scanf("%s", input);
    scanf("%s", output);

    FILE *fp = fopen(input, "rb");
    if (fp == NULL) {
        fprintf(stderr, "ERROR: open input file");
        return 0;
    }

    int nc;
    scanf("%d", &nc);

    if (nc <= 0) {
        fprintf(stderr, "ERROR: invalid class count");
        fclose(fp);
        return 0;
    }

    int w, h;
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);

    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    if (data == NULL) {
        fprintf(stderr, "ERROR: malloc data");
        fclose(fp);
        return 0;
    }

    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    Color *normalized = (Color *)malloc(sizeof(Color) * nc);
     if (data == NULL) {
        fprintf(stderr, "ERROR: malloc normalized");
        fclose(fp);
        free(data);
        return 0;
    }

    int np;
    for (int cl = 0; cl < nc; cl++) {
        scanf("%d", &np);

        float avgX = 0.0, avgY = 0.0, avgZ = 0.0;
        int pixels = np;
        while(pixels--) {
            int x, y;
            scanf("%d %d", &x, &y);
            
            int i = y * w + x;
            avgX += data[i].x;
            avgY += data[i].y;
            avgZ += data[i].z;
        }

        avgX /= (float)np;
        avgY /= (float)np;
        avgZ /= (float)np;

        float len = sqrt(avgX * avgX + avgY * avgY + avgZ * avgZ);
        normalized[cl].x = avgX / len;
        normalized[cl].y = avgY / len;
        normalized[cl].z = avgZ / len;
    }

    Color *cudaNormalized;
    CSC(cudaMalloc(&cudaNormalized, sizeof(Color) * nc));
    CSC(cudaMemcpy(cudaNormalized, normalized, nc * sizeof(Color), cudaMemcpyHostToDevice));

    uchar4 *cudaData;
    CSC(cudaMalloc(&cudaData, sizeof(uchar4) * w * h));
    CSC(cudaMemcpy(cudaData, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    spectralMethod<<<1024, 256>>>(cudaData, w, h, nc, cudaNormalized);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, cudaData, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    fp = fopen(output, "wb");
    if (fp == NULL) {
        fprintf(stderr, "ERROR: open output file");
        free(data);
        CSC(cudaFree(cudaData));
        return 0;
    }

    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    CSC(cudaFree(cudaData));

    return 0;
}