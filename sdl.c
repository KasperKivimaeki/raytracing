#include "sdl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void LDS_prepareScene(const int frame, const void* buffer) {
    SDL_UpdateTexture(app.texture, NULL, buffer, SCREEN_WIDTH * sizeof(int));
    SDL_RenderClear(app.renderer);
}

void LDS_presentScene() {
    SDL_RenderCopy(app.renderer, app.texture, NULL, NULL);
    SDL_RenderPresent(app.renderer);
}

// Nice init function by parallelrealities.co.uk
void LDS_initSDL() {
    int rendererFlags, windowFlags;

    rendererFlags = SDL_RENDERER_ACCELERATED;
    windowFlags = 0;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("Couldn't initialize SDL: %s\n", SDL_GetError());
        exit(1);
    }

    app.window = SDL_CreateWindow("Mandelbrot set", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, windowFlags);

    if (!app.window) {
        printf("Failed to open %d x %d window: %s\n", SCREEN_WIDTH, SCREEN_HEIGHT, SDL_GetError());
        exit(1);
    }

    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");

    app.renderer = SDL_CreateRenderer(app.window, -1, rendererFlags);

    if (!app.renderer) {
        printf("Failed to create renderer: %s\n", SDL_GetError());
        exit(1);
    }

    app.texture = SDL_CreateTexture(app.renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STATIC, SCREEN_WIDTH, SCREEN_HEIGHT);
}
