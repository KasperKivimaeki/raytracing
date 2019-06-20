#include <stdio.h>
#include <math.h>
#include "sdl.h"
#include "rays.h"

double sx1;
double sy1;

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
            default:
                break;
        }
    }
}

int main(int argc, char *argv[]) {
    memset(&app, 0, sizeof(app));
    LDS_initSDL();
    void* buffer = malloc(SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(int));

    vec3f* vertices = malloc(sizeof(vec3f)*1000);
    int** triangles = malloc(sizeof(int*)*1000);

    int tr;
    loadmesh(triangles, vertices, &tr);

    int frame = 0;
    char click = 0;

    vec3f origin = {0, -5, 0};
    double fov = 1.04;
    double fov_ = 1.04*SCREEN_HEIGHT/SCREEN_WIDTH;

    // For each pixel  (ray)
    for(int h = 0; h < SCREEN_HEIGHT; h++) {
        for(int w = 0; w < SCREEN_WIDTH; w++) {
            // For each object (triangle)
            for(int it = 0; it < tr; it++) {
                // Copy three points of a triangle
                vec3f v0 = vertices[triangles[it][0]];
                vec3f v1 = vertices[triangles[it][1]];
                vec3f v2 = vertices[triangles[it][2]];

                double xr = -tan(w*1.0/SCREEN_WIDTH*fov - fov/2);
                double yr = -1;
                double zr = -tan(fov_/2 - h*1.0/SCREEN_HEIGHT*fov_);

                vec3f d = {xr, yr, zr};

                int intersect = intersects(origin, d, v0, v1, v2);

                if(intersect == 1) {
                    ((int *)buffer)[w + h*SCREEN_WIDTH] = 0xff0000;
                    break;
                } else if(intersect) {
                    ((int *)buffer)[w + h*SCREEN_WIDTH] = 0x0000ff;
                    break;
                } else {
                    ((int *)buffer)[w + h*SCREEN_WIDTH] = 0x00f000 + w;
                }
                       
            }
        }
    }
    while(1) {
        if(click) {
            // Debug
            break;
        }
        LDS_prepareScene(frame, buffer);
        doInput(&click);
        LDS_presentScene();
        SDL_Delay(16);

        frame += 1;
    }

    free(buffer);
    for(int i = 0; i < tr; i++)
        free(triangles[i]);
    free(triangles);
    free(vertices);
    return 0;
}
