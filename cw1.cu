#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include <vector>
#include <math.h>

#define CSC(call)                                                 \
    do                                                            \
    {                                                             \
        cudaError_t res = call;                                   \
        if (res != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(res)); \
            exit(0);                                              \
        }                                                         \
    } while (0)

using namespace std;

struct vec3 {
    float x;
    float y;
    float z;
};

struct material {
    float specular;
    uchar4 color;
    float diffus;
    float spec;
    float refl;
};

struct trig {
    vec3 a;
    vec3 b;
    vec3 c;
    material mat;
    vec3 normal;
};

struct lightSource {
    vec3 pos;
    float intensity;
    vec3 color;
};

__host__ __device__ float dot(const vec3 &a, const vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ vec3 prod(const vec3 &a, const vec3 &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__host__ __device__ vec3 norm(const vec3 &v) {
    float l = sqrt(dot(v, v));
    return {v.x / l, v.y / l, v.z / l};
}

__host__ __device__ vec3 diff(const vec3 &a, const vec3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ vec3 add(const vec3 &a, const vec3 &b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ vec3 mult(const vec3 &a, const vec3 &b, const vec3 &c, const vec3 &v) {
    return {
        a.x * v.x + b.x * v.y + c.x * v.z,
        a.y * v.x + b.y * v.y + c.y * v.z,
        a.z * v.x + b.z * v.y + c.z * v.z
    };
}

__host__ __device__ vec3 multFloat(const vec3 &a, float c) {
    return {a.x * c, a.y * c, a.z * c};
}

__host__ __device__ int closestTrig(const vec3 &pos, const vec3 &dir, float &ts_min, trig *trigs, int n) {
    int k_min = -1;
    for (int k = 0; k < n; k++) {
        vec3 e1 = diff(trigs[k].b, trigs[k].a);
        vec3 e2 = diff(trigs[k].c, trigs[k].a);
        vec3 p = prod(dir, e2);
        double div = dot(p, e1);
        if (fabs(div) < 1e-10) {
            continue;
        }
        vec3 t = diff(pos, trigs[k].a);
        double u = dot(p, t) / div;
        if ((u < 0.0) || (u > 1.0)) {
            continue;
        }
        vec3 q = prod(t, e1);
        double v = dot(q, dir) / div;
        if ((v < 0.0) || (v + u > 1.0)) {
            continue;
        }
        double ts = dot(q, e2) / div;
        if (ts < 0.0) {
            continue;
        }
        if (k_min == -1 || ts < ts_min) {
            k_min = k;
            ts_min = ts;
        }
    }
    return k_min;
}

__host__ __device__ vec3 reflect(const vec3 &I, const vec3 &N) {
    return diff(I, multFloat(multFloat(N, 2), dot(I, N)));
}

void Hexahedron(float rad, float centX, float centY, float centZ, const material &mat, vector<trig> &figures) {
    vec3 pointA = {centX - rad, centY - rad, centZ - rad}; // -1 -1 -1
    vec3 pointB = {centX - rad, centY - rad, rad + centZ}; // -1 -1 1
    vec3 pointC = {centX - rad, centY + rad, centZ - rad}; // -1 1 -1
    vec3 pointD = {centX - rad, centY + rad, rad + centZ}; // -1 1 1
    vec3 pointE = {centX + rad, centY - rad, centZ - rad}; // 1 -1 -1
    vec3 pointF = {centX + rad, centY - rad, rad + centZ}; // 1 -1 1
    vec3 pointG = {centX + rad, centY + rad, centZ - rad}; // 1 1 -1
    vec3 pointH = {centX + rad, centY + rad, rad + centZ}; // 1 1 1

    figures.push_back({pointA, pointB, pointD, mat});
    figures.push_back({pointA, pointC, pointD, mat});
    figures.push_back({pointB, pointF, pointH, mat});
    figures.push_back({pointB, pointD, pointH, mat});
    figures.push_back({pointE, pointF, pointH, mat});
    figures.push_back({pointE, pointG, pointH, mat});
    figures.push_back({pointA, pointE, pointG, mat});
    figures.push_back({pointA, pointC, pointG, mat});
    figures.push_back({pointA, pointB, pointF, mat});
    figures.push_back({pointA, pointE, pointF, mat});
    figures.push_back({pointC, pointD, pointH, mat});
    figures.push_back({pointC, pointG, pointH, mat});
}

void Octahedron(float rad, float centX, float centY, float centZ, const material &mat, vector<trig> &figures) {
    vec3 pointA = {centX - rad, centY + rad, centZ}; // -1 1 0
    vec3 pointB = {centX + rad, centY + rad, centZ}; // 1 1 0
    vec3 pointC = {centX + rad, centY - rad, centZ}; // 1 -1 0
    vec3 pointD = {centX - rad, centY - rad, centZ}; // -1 -1 0
    vec3 pointE = {centX, centY, centZ + rad}; // 0 0 1
    vec3 pointF = {centX, centY, centZ - rad}; // 0 0 -1

    figures.push_back({pointA, pointB, pointE, mat});
    figures.push_back({pointB, pointD, pointE, mat});
    figures.push_back({pointB, pointC, pointE, mat});
    figures.push_back({pointA, pointD, pointE, mat});
    figures.push_back({pointA, pointB, pointF, mat});
    figures.push_back({pointB, pointD, pointF, mat});
    figures.push_back({pointB, pointC, pointF, mat});
    figures.push_back({pointA, pointD, pointF, mat});
}

void Icosahedron(float rad, float centX, float centY, float centZ, const material &mat, vector<trig> &figures) {
    vec3 pointA = {centX, centY - rad, centZ + rad}; // 0 -1 1
    vec3 pointB = {centX, centY + rad, centZ + rad}; // 0 1 1
    vec3 pointC = {centX - rad, centY, centZ + rad}; // -1 0 1
    vec3 pointD = {centX + rad, centY, centZ + rad}; // 1 0 1
    vec3 pointE = {centX - rad, centY + rad, centZ}; // -1 1 0
    vec3 pointF = {centX + rad, centY + rad, centZ}; // 1 1 0
    vec3 pointG = {centX + rad, centY - rad, centZ}; // 1 -1 0
    vec3 pointH = {centX - rad, centY - rad, centZ}; // -1 -1 0
    vec3 pointI = {centX - rad, centY, centZ - rad}; // -1 0 -1
    vec3 pointJ = {centX + rad, centY, centZ - rad}; // 1 0 -1
    vec3 pointK = {centX, centY - rad, centZ - rad}; // 0 -1 -1
    vec3 pointL = {centX, centY + rad, centZ - rad}; // 0 1 -1

    figures.push_back({pointA, pointB, pointC, mat});
    figures.push_back({pointB, pointA, pointD, mat});
    figures.push_back({pointA, pointC, pointH, mat});
    figures.push_back({pointC, pointB, pointE, mat});
    figures.push_back({pointE, pointB, pointF, mat});
    figures.push_back({pointG, pointA, pointH, mat});
    figures.push_back({pointD, pointA, pointG, mat});
    figures.push_back({pointB, pointD, pointF, mat});
    figures.push_back({pointE, pointF, pointL, mat});
    figures.push_back({pointG, pointH, pointK, mat});
    figures.push_back({pointD, pointG, pointJ, mat});
    figures.push_back({pointF, pointD, pointJ, mat});
    figures.push_back({pointH, pointC, pointI, mat});
    figures.push_back({pointC, pointE, pointI, mat});
    figures.push_back({pointJ, pointK, pointL, mat});
    figures.push_back({pointK, pointI, pointL, mat});
    figures.push_back({pointF, pointJ, pointL, mat});
    figures.push_back({pointJ, pointG, pointK, mat});
    figures.push_back({pointH, pointI, pointK, mat});
    figures.push_back({pointI, pointE, pointL, mat});
}

void Floor(const trig &fl1, const trig &fl2, const material &mat, vector<trig> &figures) {
    figures.push_back({fl1.a, fl1.b, fl1.c, mat, {0.0, 0.0, 1.0}});
    figures.push_back({fl2.a, fl2.b, fl2.c, mat, {0.0, 0.0, 1.0}});
}

void buildSpace(vector<trig> &figures, const vec3 &cent1, const vec3 &cent2, const vec3 &cent3, const trig &fl1, const trig &fl2, const vec3 &rads, material *materials) {
    Floor(fl1, fl2, materials[3], figures);

    Hexahedron(rads.y, cent1.x, cent1.y, cent1.z, materials[0], figures);
    Octahedron(rads.x, cent2.x, cent2.y, cent2.z, materials[1], figures);
    Icosahedron(rads.z, cent3.x, cent3.y, cent3.z, materials[2], figures);
}

__host__ __device__ uchar4 ray(vec3 pos, vec3 dir, int depth, uchar4 *floor, int floorSize, int sceneSize, trig *trigs, lightSource *lights, int lightsNum, int maxRayDeep, int n) {
    float floatMax = FLT_MAX;

    uchar4 reflectColor = {0, 0, 0, 0};

    if (depth > maxRayDeep) {
        return {0, 0, 0, 0};
    }

    int currTrig = closestTrig(pos, dir, floatMax, trigs, n);
    if (currTrig == -1) {
        return {0, 0, 0, 0};
    }

    vec3 inter = add(pos, multFloat(dir, floatMax));

    material interMat = trigs[currTrig].mat;
    uchar4 matColor = make_uchar4(interMat.color.x, interMat.color.y, interMat.color.z, interMat.color.w);
    if (currTrig < 2) { // пол
        int x = (inter.x / floorSize * sceneSize + sceneSize / 2);
        int y = (inter.y / floorSize * sceneSize + sceneSize / 2);
        matColor = floor[x * sceneSize + y];
    }

    vec3 interNormal = trigs[currTrig].normal;
    vec3 reflectDir = norm(reflect(dir, interNormal));
    vec3 reflectPos;
    if (dot(reflectDir, interNormal) < 0) {
        reflectPos = diff(inter, multFloat(interNormal, 1e-6));
    } else {
        reflectPos = add(inter, multFloat(interNormal, 1e-6));
    }

    reflectColor = ray(reflectPos, reflectDir, depth + 1, floor, floorSize, sceneSize, trigs, lights, lightsNum, maxRayDeep, n);

    vec3 sumColor = {0.0, 0.0, 0.0};

    for (int i = 0; i < lightsNum; i++) {
        vec3 lightDir = norm(diff(lights[i].pos, inter));
        int lightTrig = closestTrig(lights[i].pos, multFloat(lightDir, -1.0), floatMax, trigs, n);

        float diffuseIntens = 0.0;
        float specularIntens = 0.0;
        if (lightTrig == currTrig) { // не тень
            diffuseIntens = lights[i].intensity * max(0.0, dot(lightDir, interNormal));
            specularIntens = pow(max(0.0, dot(multFloat(reflect(multFloat(lightDir, -1), interNormal), -1), dir)), (double)interMat.specular) * lights[i].intensity;
        }

        sumColor.x += lights[i].color.x * matColor.x * diffuseIntens * interMat.diffus + 255.0 * specularIntens * interMat.spec;
        sumColor.y += lights[i].color.y * matColor.y * diffuseIntens * interMat.diffus + 255.0 * specularIntens * interMat.spec;
        sumColor.z += lights[i].color.z * matColor.z * diffuseIntens * interMat.diffus + 255.0 * specularIntens * interMat.spec;
    }

    return make_uchar4(
        min(0.1 * matColor.x + sumColor.x + interMat.refl * reflectColor.x, 255.0),
        min(0.1 * matColor.y + sumColor.y + interMat.refl * reflectColor.y, 255.0),
        min(0.1 * matColor.z + sumColor.z + interMat.refl * reflectColor.z, 255.0),
        matColor.w
    );
}

void ssaa(uchar4 *data, uchar4 *out, int w, int h, int sqrtRayNum) {
    int wScale = w * sqrtRayNum;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int yScale = y * sqrtRayNum;
            int xScale = x * sqrtRayNum;
            uint4 sumColor = make_uint4(0, 0, 0, 0);
            
            for (int n = yScale; n < yScale + sqrtRayNum; n++) {
                for (int m = xScale; m < xScale + sqrtRayNum; m++) {
                    sumColor.x += data[n * wScale + m].x;
                    sumColor.y += data[n * wScale + m].y;
                    sumColor.z += data[n * wScale + m].z;
                }
            }
            
            float d = sqrtRayNum * sqrtRayNum;
            out[y * w + x] = make_uchar4(sumColor.x / d, sumColor.y / d, sumColor.z / d, sumColor.w);
        }
    }
}

__global__ void kernelSsaa(uchar4 *data, uchar4 *out, int w, int h, int sqrtRayNum) {
    int beginX = blockIdx.x * blockDim.x + threadIdx.x;
    int beginY = blockIdx.y * blockDim.y + threadIdx.y;
    int totalX = blockDim.x * gridDim.x;
    int totalY = blockDim.y * gridDim.y;

    int wScale = w * sqrtRayNum;
    for (int y = beginX; y < h; y += totalX) {
        for (int x = beginY; x < w; x += totalY) {
            int yScale = y * sqrtRayNum;
            int xScale = x * sqrtRayNum;
            uint4 sumColor = make_uint4(0, 0, 0, 0);
            
            for (int n = yScale; n < yScale + sqrtRayNum; n++) {
                for (int m = xScale; m < xScale + sqrtRayNum; m++) {
                    sumColor.x += data[n * wScale + m].x;
                    sumColor.y += data[n * wScale + m].y;
                    sumColor.z += data[n * wScale + m].z;
                }
            }
            
            float d = sqrtRayNum * sqrtRayNum;
            out[y * w + x] = make_uchar4(sumColor.x / d, sumColor.y / d, sumColor.z / d, sumColor.w);
        }
    }
}

void render(vec3 pc, vec3 pv, int w, int h, double angle, uchar4 *data, uchar4 *floor, int floorSize, int sceneSize, trig *trigs, lightSource *lights, int lightsNum, int maxRayDeep, int n) {
    float stepW = 2.0 / (w - 1);
    float stepH = 2.0 / (h - 1);
    float dist = 1.0 / tan(angle * M_PI / 360.0);
    
    vec3 viewDir = norm(diff(pv, pc));
    vec3 rightVec = norm(prod(viewDir, {0.0, 0.0, 1.0}));
    vec3 upVec = prod(rightVec, viewDir);
    
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            vec3 pixel = {-1 + stepW * i, (-1 + stepH * j) * h / w, dist};
            vec3 rayDir = norm(mult(rightVec, upVec, viewDir, pixel));
            
            data[(h - 1 - j) * w + i] = ray(pc, rayDir, 0, floor, floorSize, sceneSize, trigs, lights, lightsNum, maxRayDeep, n);
        }
    }
}

__global__ void kernelRender(vec3 pc, vec3 pv, int w, int h, double angle, uchar4 *data, uchar4 *floor, int floorSize, int sceneSize, trig *trigs, lightSource *lights, int lightsNum, int maxRayDeep, int n) {
    int beginX = blockIdx.x * blockDim.x + threadIdx.x;
    int beginY = blockIdx.y * blockDim.y + threadIdx.y;
    int totalX = blockDim.x * gridDim.x;
    int totalY = blockDim.y * gridDim.y;

    float stepW = 2.0 / (w - 1.0);
    float stepH = 2.0 / (h - 1.0);
    float dist = 1.0 / tan(angle * M_PI / 360.0);
    
    vec3 viewDir = norm(diff(pv, pc));
    vec3 rightVec = norm(prod(viewDir, {0.0, 0.0, 1.0}));
    vec3 upVec = prod(rightVec, viewDir);
    
    for (int j = beginY; j < h; j += totalY) {
        for (int i = beginX; i < w; i += totalX) {
            vec3 pixel = {-1 + stepW * i, (-1 + stepH * j) * h / w, dist};
            vec3 rayDir = norm(mult(rightVec, upVec, viewDir, pixel));
            
            data[(h - 1 - j) * w + i] = ray(pc, rayDir, 0, floor, floorSize, sceneSize, trigs, lights, lightsNum, maxRayDeep, n);
        }
    }
}

int main(int argc, char *argv[]) {
    string cmd = argv[1];
    if (cmd != "--cpu" && cmd != "--gpu" && cmd != "--default") {
        return 0;
    }

    bool onGpu = cmd != "--cpu";

    int frameNum = 120;
    char pathToResult[100] = "res/%d.data";
    int w = 640, h = 480, angle = 120;
    float rc0 = 7.0, zc0 = 3.0, phic0 = 0.0, acr = 2.0, acz = 1.0, wcr = 2.0, wcz = 6.0, wcphi = 1.0, pcr = 0.0, pcz = 0.0;
    float rn0 = 2.0, zn0 = 0.0, phin0 = 0.0, anr = 0.5, anz = 0.1, wnr = 1.0, wnz = 4.0, wnphi = 1.0, pnr = 0.0, pnz = 0.0;

    float centerX1 = 3.0, centerY1 = -2.0, centerZ1 = 2.5, colorR1 = 1.0, colorG1 = 0.0, colorB1 = 1.0, r1 = 1.5, refl1 = 0.8, emptyRefr1 = 0.0;
    float centerX2 = -1.0, centerY2 = -1.5, centerZ2 = 1.5, colorR2 = 0.2, colorG2 = 1.0, colorB2 = 0.2, r2 = 1.5, refl2 = 0.7, emptyRefr2 = 0.0;
    float centerX3 = -2.5, centerY3 = 2.5, centerZ3 = 2.5, colorR3 = 0.0, colorG3 = 1.0, colorB3 = 1.0, r3 = 1.5, refl3 = 0.6, emptyRefr3 = 0.0;
    int empty = 0;

    float floorX1 = -5.0, floorY1 = -5.0, floorZ1 = 0.0, floorX2 = -5.0, floorY2 = 5.0, floorZ2 = 0.0, floorX3 = 5.0, floorY3 = 5.0, floorZ3 = 0.0, floorX4 = 5.0, floorY4 = -5.0, floorZ4 = 0.0;
    char floorPath[100] = "board.data";
    float floorR = 1.0, floorG = 1.0, floorB = 1.0, floorRefl = 0.7;

    int lightsNum = 2;
    float lightX1 = 5.0, lightY1 = 5.0, lightZ1 = 5.0;
    float lightR1 = 1.0, lightG1 = 1.0, lightB1 = 1.0, lightIntens1 = 0.8;
    float lightX2 = -5.0, lightY2 = 5.0, lightZ2 = 5.0;
    float lightR2 = 1.0, lightG2 = 1.0, lightB2 = 1.0, lightIntens2 = 0.8;

    int maxRayDeep = 2, sqrtRayNum = 4;
    lightSource *lights;

    if (cmd == "--default") {
      cout << frameNum << '\n';
      cout << pathToResult << '\n';
      cout << w << ' ' << h << ' ' << angle << '\n';
      cout << rc0 << ' ' << zc0 << ' ' << phic0 << ' ' << acr << ' ' << acz << ' ' << wcr << ' ' << wcz << ' ' << wcphi << ' ' << pcr << ' ' << pcz << '\n';
      cout << rn0 << ' ' << zn0 << ' ' << phin0 << ' ' << anr << ' ' << anz << ' ' << wnr << ' ' << wnz << ' ' << wnphi << ' ' << pnr << ' ' << pnz << '\n';
      cout << centerX1 << ' ' << centerY1 << ' ' << centerZ1 << ' ' << colorR1 << ' ' << colorG1 << ' ' << colorB1 << ' ' << r1 << ' ' << refl1 << ' ' << emptyRefr1 << ' ' << empty << '\n';
      cout << centerX2 << ' ' << centerY2 << ' ' << centerZ2 << ' ' << colorR2 << ' ' << colorG2 << ' ' << colorB2 << ' ' << r2 << ' ' << refl2 << ' ' << emptyRefr2 << ' ' << empty << '\n';
      cout << centerX3 << ' ' << centerY3 << ' ' << centerZ3 << ' ' << colorR3 << ' ' << colorG3 << ' ' << colorB3 << ' ' << r3 << ' ' << refl3 << ' ' << emptyRefr3 << ' ' << empty << '\n';
      cout << floorX1 << ' ' << floorY1 << ' ' << floorZ1 << ' ' << floorX2 << ' ' << floorY2 << ' ' << floorZ2 << ' ' << floorX3 << ' ' << floorY3 << ' ' << floorZ3 << ' ' << floorX4 << ' ' << floorY4 << ' ' << floorZ4 << '\n';
      cout << floorPath << '\n';
      cout << floorR << ' ' << floorG << ' ' << floorB << ' ' << floorRefl << '\n';
      cout << lightsNum << '\n';
      cout << lightX1 << ' ' << lightY1 << ' ' << lightZ1 << '\n';
      cout << lightR1 << ' ' << lightG1 << ' ' << lightB1 << '\n';
      cout << lightX2 << ' ' << lightY2 << ' ' << lightZ2 << '\n';
      cout << lightR2 << ' ' << lightG2 << ' ' << lightB2 << '\n';

      lights = (lightSource *)malloc(sizeof(lightSource) * lightsNum);
      lights[0] = {{lightX1, lightY1, lightZ1}, lightIntens1, {lightR1, lightG1, lightB1}};
      lights[1] = {{lightX2, lightY2, lightZ2}, lightIntens2, {lightR2, lightG2, lightB2}};

      cout << maxRayDeep << ' ' << sqrtRayNum << '\n';

      return 0;
    }

    cin >> frameNum;
    cin >> pathToResult;
    cin >> w >> h >> angle;
    cin >> rc0 >> zc0 >> phic0 >> acr >> acz >> wcr >> wcz >> wcphi >> pcr >> pcz;
    cin >> rn0 >> zn0 >> phin0 >> anr >> anz >> wnr >> wnz >> wnphi >> pnr >> pnz;
    cin >> centerX1 >> centerY1 >> centerZ1 >> colorR1 >> colorG1 >> colorB1 >> r1 >> refl1 >> emptyRefr1 >> empty;
    cin >> centerX2 >> centerY2 >> centerZ2 >> colorR2 >> colorG2 >> colorB2 >> r2 >> refl2 >> emptyRefr2 >> empty;
    cin >> centerX3 >> centerY3 >> centerZ3 >> colorR3 >> colorG3 >> colorB3 >> r3 >> refl3 >> emptyRefr3 >> empty;
    cin >> floorX1 >> floorY1 >> floorZ1 >> floorX2 >> floorY2 >> floorZ2 >> floorX3 >> floorY3 >> floorZ3 >> floorX4 >> floorY4 >> floorZ4;
    cin >> floorPath;
    cin >> floorR >> floorG >> floorB >> floorRefl;
    cin >> lightsNum;

    lights = (lightSource*) malloc(sizeof(lightSource) * lightsNum);
    float lightX, lightY, lightZ;
    float lightR, lightG, lightB;
    for (int i = 0; i < lightsNum; i++) {
        cin >> lightX >> lightY >> lightZ;
        cin >> lightR >> lightG >> lightB;
        lights[i] = {{lightX, lightY, lightZ}, lightIntens1, {lightR, lightG, lightB}};
    }
    cin >> maxRayDeep >> sqrtRayNum;

    char buff[256];
    uchar4 *data = (uchar4*) malloc(sizeof(uchar4) * w * h * sqrtRayNum * sqrtRayNum);
    uchar4 *shortData = (uchar4*) malloc(sizeof(uchar4) * w * h);
    vector<trig> figures;

    uchar4 *cudaData;
    uchar4 *cudaShortData;
    trig *cudaFigures;
    lightSource *cudaLights;
    uchar4 *cudaFloor;

    vec3 pc, pv;

    int w1, h1;
    FILE *fp = fopen(floorPath, "rb");
    if (fp == NULL) {
        fprintf(stderr, "ERROR: open floor file");
        return 0;
    }
    fread(&w1, sizeof(int), 1, fp);
    fread(&h1, sizeof(int), 1, fp);
    uchar4 *floor = (uchar4 *)malloc(sizeof(uchar4) * w1 * h1);
    fread(floor, sizeof(uchar4), w1 * h1, fp);
    fclose(fp);

    for (int i = 0; i < w1 * h1; i++) {
        floor[i] = make_uchar4(floorR * floor[i].x, floorG * floor[i].y, floorB * floor[i].z, floor[i].w);
    }
    int floorSize = (int)abs(floorX1 - floorX3);

    trig fl1 = {{floorX1, floorY1, floorZ1}, {floorX2, floorY2, floorZ2}, {floorX3, floorY3, floorZ3}};
    trig fl2 = {{floorX1, floorY1, floorZ1}, {floorX3, floorY3, floorZ3}, {floorX4, floorY4, floorZ4}};
    material *materials = (material*) malloc(sizeof(material) * 4);
    materials[0] = {125.0, make_uchar4(colorR1 * 255, colorG1 * 255, colorB1 * 255, 0), 0.2, 0.4, refl1}; // purple
    materials[1] = {125.0, make_uchar4(colorR2 * 255, colorG2 * 255, colorB2 * 255, 0), 0.2, 0.3, refl2}; // blue
    materials[2] = {125.0, make_uchar4(colorR3 * 255, colorG3 * 255, colorB3 * 255, 0), 0.7, 0.5, refl3}; // green
    materials[3] = {125.0, make_uchar4(floorR * 255, floorG * 255, floorB * 255, 0), 0.6, 0.7, floorRefl};

    buildSpace(figures, {centerX1, centerY1, centerZ1}, {centerX2, centerY2, centerZ2}, {centerX3, centerY3, centerZ3}, fl1, fl2, {r1, r2, r3}, materials);

    if (onGpu) {
        CSC(cudaMalloc(&cudaData, sizeof(uchar4) * w * h * sqrtRayNum * sqrtRayNum));
        CSC(cudaMalloc(&cudaShortData, sizeof(uchar4) * w * h));
        CSC(cudaMalloc((trig**)(&cudaFigures), figures.size() * sizeof(trig)));
        CSC(cudaMalloc(&cudaLights, sizeof(lightSource) * lightsNum));
        CSC(cudaMalloc(&cudaFloor, sizeof(uchar4) * w1 * h1));

        CSC(cudaMemcpy(cudaFigures, figures.data(), figures.size() * sizeof(trig), cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(cudaLights, lights, sizeof(lightSource) * lightsNum, cudaMemcpyHostToDevice));
        CSC(cudaMemcpy(cudaFloor, floor, sizeof(uchar4) * w1 * h1, cudaMemcpyHostToDevice));
    }

    float gpuTime = 0.0, sumGpuTime = 0.0;
    for (int k = 0; k < frameNum; k++) {
        float step = 2 * M_PI * k / frameNum;

        float rct = rc0 + acr * sin(wcr * step + pcr);
        float zct = zc0 + acz * sin(wcz * step + pcz);
        float phict = phic0 + wcphi * step;

        float rnt = rn0 + anr * sin(wnr * step + pnr);
        float znt = zn0 + anz * sin(wnz * step + pnz);
        float phint = phin0 + wnphi * step;

        pc = {rct * cos(phict), rct * sin(phict), zct};
        pv = {rnt * cos(phint), rnt * sin(phint), znt};

        cudaEvent_t begin, end;
        gpuTime = 0.0;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
        cudaEventRecord(begin, 0);

        if (onGpu) {
            kernelRender<<<dim3(8, 32), dim3(8, 32)>>>(pc, pv, w * sqrtRayNum, h * sqrtRayNum, angle, cudaData, cudaFloor, floorSize, w1, cudaFigures, cudaLights, lightsNum, maxRayDeep, figures.size());
            CSC(cudaGetLastError());

            kernelSsaa<<<dim3(8, 32), dim3(8, 32)>>>(cudaData, cudaShortData, w, h, sqrtRayNum);
            CSC(cudaGetLastError());

            CSC(cudaMemcpy(data, cudaShortData, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
        } else {
            render(pc, pv, w * sqrtRayNum, h * sqrtRayNum, angle, data, floor, floorSize, w1, figures.data(), lights, lightsNum, maxRayDeep, figures.size());
            ssaa(data, shortData, w, h, sqrtRayNum);
            memcpy(data, shortData, sizeof(uchar4) * w * h);
        }

        cudaEventRecord(end, 0);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&gpuTime, begin, end);
        sumGpuTime += gpuTime;

        cout << "Кадр-" << k << '\t' << "Время: " << gpuTime << "ms" << '\t' << "Отражений: " << w * h * sqrtRayNum * sqrtRayNum << '\n';

        sprintf(buff, pathToResult, k);
        FILE *out = fopen(buff, "wb");
        fwrite(&w, sizeof(int), 1, out);
        fwrite(&h, sizeof(int), 1, out);
        fwrite(data, sizeof(uchar4), w * h, out);
        fclose(out);
    }

    cout << "Итоговое время: " << sumGpuTime << " ms Среднее время: " << sumGpuTime / float(frameNum) << " ms\n";

    free(data);
    free(shortData);
    free(floor);
    free(materials);

    if (onGpu) {
        CSC(cudaFree(cudaData));
        CSC(cudaFree(cudaShortData));
        CSC(cudaFree(cudaFigures));
        CSC(cudaFree(cudaLights));
        CSC(cudaFree(cudaFloor));
    }

    return 0;
}