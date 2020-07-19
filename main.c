#include <stdio.h>
#include <math.h>
#include "sdl.h"
#include "rays.h"

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

void draw(vec3f *vertices, int **triangles, int tr, double fovx, double fovy, void *buffer) {
    // For each pixel  (ray)
    for(int h = 0; h < SCREEN_HEIGHT; h++) {
        for(int w = 0; w < SCREEN_WIDTH; w++) {
            // For each object (triangle)
            for(int it = 0; it < tr; it++) {
                // Copy three points of a triangle
                vec3f v0 = vertices[triangles[it][0]];
                vec3f v1 = vertices[triangles[it][1]];
                vec3f v2 = vertices[triangles[it][2]];

                double xr = -tan(w*1.0/SCREEN_WIDTH*fovy - fovy/2);
                double yr = -1;
                double zr = -tan(fovx/2 - h*1.0/SCREEN_HEIGHT*fovx);

                vec3f d = {xr, yr, zr};

                int intersect = intersects(origin, d, v0, v1, v2);

                if(intersect) {
                    ((int *)buffer)[w + h*SCREEN_WIDTH] = COLOR_PICK(it);
                    break;
                } else {
                    ((int *)buffer)[w + h*SCREEN_WIDTH] = 0x00f000 + w;
                }
            }
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

    double fovy = 1.04;
    double fovx = 1.04*SCREEN_HEIGHT/SCREEN_WIDTH;

    while(1) {
        if(click) {
            // Debug
            break;
        }
        draw(vertices, triangles, tr, fovx, fovy, buffer);
        LDS_prepareScene(frame, buffer);
        doInput(&click);
        LDS_presentScene();
        frame += 1;
    }

    free(buffer);
    for(int i = 0; i < tr; i++)
        free(triangles[i]);
    free(triangles);
    free(vertices);
    return 0;
}
