#include "tracer.hpp"
#include "rays.hpp"
#include <SDL.h>
#include <cuda_runtime.h>

Tracer::Tracer(App *app) {
    buffer = new int[SCREEN_WIDTH * SCREEN_HEIGHT];


    vertices = std::vector<vec3f>();
    triangles = std::vector<vec3i>();

    this->frame = 0;

    this->app = app;

    // TODO: Fix to support window resizing
    this->fovy = 1.04;
    this->fovx = 1.04*SCREEN_HEIGHT/SCREEN_WIDTH;

    loadObjFile(triangles, vertices);

    this->ang = 0;

    this->origin = {0, -10, 0};
    this->click = 0;

    vBuf = NULL;
    tBuf = NULL;
    gBuf = NULL;

    CHECK(cudaMalloc((void**)&vBuf, vertices.size() * sizeof(vec3f)));
    CHECK(cudaMalloc((void**)&tBuf, 3 * triangles.size() * sizeof(int)));
    CHECK(cudaMalloc((void**)&gBuf, SCREEN_HEIGHT * SCREEN_WIDTH * sizeof(int)));
    CHECK(cudaMemcpy(vBuf, vertices.data(), vertices.size() * sizeof(vec3f), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(tBuf, triangles.data(), 3 * triangles.size() * sizeof(int), cudaMemcpyHostToDevice));
}

Tracer::~Tracer() {
    delete[] buffer;

    CHECK(cudaFree(vBuf));
    CHECK(cudaFree(tBuf));
    CHECK(cudaFree(gBuf));
}

void Tracer::draw() const {
    dim3 blocks(SCREEN_WIDTH/8, SCREEN_HEIGHT/8);
    dim3 threads(8, 8);

    CHECK(cudaMemset(gBuf, 0, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(int)));

    drawRay<<<blocks, threads>>>(origin, PI*ang/180, vBuf, tBuf, triangles.size(), fovx, fovy, gBuf);

    CHECK(cudaMemcpy(buffer, gBuf, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(int), cudaMemcpyDeviceToHost));
}

void Tracer::run() {
    while(!click) {
        CHECK(cudaMemcpy(vBuf, vertices.data(), vertices.size() * sizeof(vec3f), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(tBuf, triangles.data(), 3 * triangles.size() * sizeof(int), cudaMemcpyHostToDevice));

        auto start = CURRENT_TIME;
        draw();
        auto end = CURRENT_TIME;
        int ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        printf("%dms x: %f, y: %f, z: %f, ang: %d\n", ms, origin.x, origin.y, origin.z, ang);

        app->LDS_prepareScene(frame, buffer);
        getInput();
        app->LDS_presentScene();
        frame += 1;
    }
}

void Tracer::getInput() {
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                exit(0);
                break;
            case SDL_MOUSEBUTTONDOWN:
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
                    case SDLK_LEFT:
                        origin.x -= 1.0*cos(ang*PI/180);
                        origin.z -= 1.0*sin(ang*PI/180);
                        break;
                    case 'd':
                    case SDLK_RIGHT:
                        origin.x += 1.0*cos(ang*PI/180);
                        origin.z += 1.0*sin(ang*PI/180);
                        break;
                    case 'w':
                    case SDLK_UP:
                        origin.z += 1.0*cos(ang*PI/180);
                        origin.x += 1.0*sin(ang*PI/180);
                        break;
                    case 's':
                    case SDLK_DOWN:
                        origin.z -= 1.0*cos(ang*PI/180);
                        origin.x -= 1.0*sin(ang*PI/180);
                        break;
                    case 'q':
                        ang += 5;
                        break;
                    case 'e':
                        ang -= 5;
                        break;
                }
                break;
            default:
                break;
        }
    }
}
