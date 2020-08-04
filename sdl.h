#ifndef SDL_HL
#define SDL_HL

#include <SDL.h>

#define SCREEN_WIDTH 1920/4
#define SCREEN_HEIGHT 1080/4

void LDS_prepareScene(int, const void*);
void LDS_presentScene();
void LDS_initSDL();

typedef struct {
    SDL_Renderer *renderer;
    SDL_Window *window;
    SDL_Texture *texture;
} App;

App app;

extern const int *COLOR_ARR[]; 

#define COLOR_PICK(i) COLOR_ARR[i % 19]

#endif
