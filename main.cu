#include <iostream>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <SDL.h>
#include <cuda_runtime.h>

#include "sdl.hpp"
#include "rays.hpp"

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

double sx1;
double sy1;

vec3f origin = {0, -4, 0};

void doInput(char *click) {
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                exit(0);
                break;
            case SDL_MOUSEBUTTONDOWN:
                (*click)++;
                printf("(x, y) (%d, %d)\n", event.button.x, event.button.y);
                sy1 = event.button.y;
                sx1 = event.button.x;
                break;
            case SDL_MOUSEWHEEL:
                if (event.wheel.y > 0)
                    origin.y += 0.1;
                if (event.wheel.y < 0)
                    origin.y -= 0.1;
                printf("%f %f %f\n", origin.x, origin.y, origin.z);
                break;
            default:
                break;
        }
    }
}

__global__ void drawRay(vec3f origin, vec3f *vertices, int *triangles, int tr, double fovx, double fovy, int *buffer) {
    int h = blockIdx.y;
    int w = blockIdx.x;

    double xr = -tan(w*1.0/SCREEN_WIDTH*fovy - fovy/2);
    double yr = -1;
    double zr = -tan(fovx/2 - h*1.0/SCREEN_HEIGHT*fovx);

    vec3f dir = {xr, yr, zr};

    vec3f e1, e2;
    vec3f tvec, pvec, qvec;

    for(int it = 0; it < tr; it++) {
        // Copy three points of a triangle
        vec3f v0 = vertices[triangles[3*it + 0]];
        vec3f v1 = vertices[triangles[3*it + 1]];
        vec3f v2 = vertices[triangles[3*it + 2]];

        int intersect = 0;

        double epsilon = 0.0001;

        sub(e1, v1, v0);
        sub(e2, v2, v0);

        sub(tvec, origin, v0);
        cross(pvec, dir, e2);
        cross(qvec, tvec, e1);

        double det = dot(pvec, e1);

        if(det < epsilon && det > -epsilon) {
            intersect = 2;
        } else {
            double inv_det = 1.0 / det;
            double u = dot(pvec, tvec) * inv_det;
            if(u < 0 || u > 1) {
                intersect = 0;
            } else {
                double v = dot(qvec, dir) * inv_det;
                if(v < 0 || v + u > 1)
                    intersect = 0;
                else
                    intersect = 1;
            }
        }

        if(intersect) {
            ((int *)buffer)[w + h*SCREEN_WIDTH] = 0xff0000;
            it = tr;
        } else {
            ((int *)buffer)[w + h*SCREEN_WIDTH] = 0x00f000 + w;
        }
    }
}

void draw(vec3f* vBuf, int* tBuf, int *gBuf, int tr, double fovx, double fovy, int *buffer) {
    dim3 blocks(SCREEN_WIDTH, SCREEN_HEIGHT);

    printf("Hello\n");
    drawRay<<<blocks, 1>>>(origin, vBuf, tBuf, tr, fovx, fovy, gBuf);
    cudaDeviceSynchronize();
    printf("Hello\n");

    CHECK(cudaMemcpy(buffer, gBuf, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(int), cudaMemcpyDeviceToHost));
}

int main(int argc, char *argv[]) {
    App *app = new App();
    app->LDS_initSDL();
    int* buffer = new int[SCREEN_WIDTH * SCREEN_HEIGHT];

    vec3f* vertices = new vec3f[1000];
    int** triangles = new int*[1000];

    int tr;
    loadmesh(triangles, vertices, &tr);

    int frame = 0;
    char click = 0;

    double fovy = 1.04;
    double fovx = 1.04*SCREEN_HEIGHT/SCREEN_WIDTH;

    vec3f* vBuf = NULL;
    int *tBuf = NULL, *gBuf = NULL;
    CHECK(cudaMalloc((void**)&vBuf, 1000 * sizeof(vec3f)));
    CHECK(cudaMalloc((void**)&tBuf, 3 * tr * sizeof(int)));
    CHECK(cudaMalloc((void**)&gBuf, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(int)));
    CHECK(cudaMemcpy(vBuf, vertices, 1000 * sizeof(vec3f), cudaMemcpyHostToDevice));
    for (int t = 0; t < tr; t++)
        CHECK(cudaMemcpy(tBuf + t*3, triangles[t], 3 * sizeof(int), cudaMemcpyHostToDevice));

    while(1) {
        if(click) {
            // Debug
            break;
        }


        auto start = std::chrono::high_resolution_clock::now();
        draw(vBuf, tBuf, gBuf, tr, fovx, fovy, buffer);
        auto end = std::chrono::high_resolution_clock::now();
        int ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        printf("%dms\n", ms);

        app->LDS_prepareScene(frame, buffer);
        doInput(&click);
        app->LDS_presentScene();
        frame += 1;
    }

    for(int i = 0; i < tr; i++)
        delete triangles[i];

    delete[] buffer;
    delete[] triangles;
    delete[] vertices;

    CHECK(cudaFree(vBuf));
    CHECK(cudaFree(tBuf));
    CHECK(cudaFree(gBuf));

    return 0;
}
