#include <iostream>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <SDL.h>
#include <cuda_runtime.h>

#include "sdl.hpp"
#include "rays.hpp"

#define PI 3.14159265

double sx1;
double sy1;

int animate = 0;

int ang = 0;

vec3f origin = {0, -10, 0};

void doInput(char *click) {
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                exit(0);
                break;
            case SDL_MOUSEBUTTONDOWN:
                (*click)++;
                //printf("(x, y) (%d, %d)\n", event.button.x, event.button.y);
                sy1 = event.button.y;
                sx1 = event.button.x;
                break;
            case SDL_MOUSEWHEEL:
                //printf("%f %f %f\n", origin.x, origin.y, origin.z);
                if (event.wheel.y > 0)
                    origin.y += 0.1;
                if (event.wheel.y < 0)
                    origin.y -= 0.1;
                break;
            case SDL_KEYDOWN:
                switch(event.key.keysym.sym) {
                    case 'a':
                        animate = !animate;
                        break;
                }
                break;
            default:
                break;
        }
    }
}

void draw(vec3f* vBuf, int* tBuf, int *gBuf, int tn, int vn, double fovx, double fovy, int *buffer) {
    dim3 blocks(SCREEN_WIDTH/8, SCREEN_HEIGHT/8);
    dim3 threads(8, 8);

    CHECK(cudaMemset(gBuf, 0, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(int)));

    drawRay<<<blocks, threads>>>(origin, vBuf, tBuf, tn, fovx, fovy, gBuf);

    CHECK(cudaMemcpy(buffer, gBuf, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(int), cudaMemcpyDeviceToHost));
}

int main(int argc, char *argv[]) {
    App *app = new App();
    app->LDS_initSDL();
    int* buffer = new int[SCREEN_WIDTH * SCREEN_HEIGHT];

    vec3f* vertices;
    int* triangles;

    int tn = 0;
    int vn = 0;
    //loadmesh(&triangles, &vertices, &tr);
    loadObjFile(&triangles, &vertices, &tn, &vn);

    int frame = 0;
    char click = 0;

    double fovy = 1.04;
    double fovx = 1.04*SCREEN_HEIGHT/SCREEN_WIDTH;

    vec3f* vBuf = NULL;
    int *tBuf = NULL, *gBuf = NULL;
    CHECK(cudaMalloc((void**)&vBuf, vn * sizeof(vec3f)));
    CHECK(cudaMalloc((void**)&tBuf, 3 * tn * sizeof(int)));
    CHECK(cudaMalloc((void**)&gBuf, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(int)));
    CHECK(cudaMemcpy(vBuf, vertices, vn * sizeof(vec3f), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(tBuf, triangles, 3 * tn * sizeof(int), cudaMemcpyHostToDevice));

    while(1) {
        if(click) {
            // Debug
            break;
        }

        if(animate) {
            ang = (ang + 1) % 360;
            origin.y = -10*sin(ang*PI/180);
            origin.x = -10*cos(ang*PI/180);
        }

        auto start = std::chrono::high_resolution_clock::now();
        draw(vBuf, tBuf, gBuf, tn, vn, fovx, fovy, buffer);
        auto end = std::chrono::high_resolution_clock::now();
        int ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        printf("%dms\n", ms);

        app->LDS_prepareScene(frame, buffer);
        doInput(&click);
        app->LDS_presentScene();
        frame += 1;
    }

    delete[] buffer;
    delete[] triangles;
    delete[] vertices;

    CHECK(cudaFree(vBuf));
    CHECK(cudaFree(tBuf));
    CHECK(cudaFree(gBuf));

    return 0;
}
