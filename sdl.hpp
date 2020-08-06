#ifndef SDL_HLLL
#define SDL_HLLL

#include <SDL.h>

#define SCREEN_WIDTH 1920
#define SCREEN_HEIGHT 1080


class App {
public:
    App() {
    }

    void LDS_prepareScene(int, const void*);
    void LDS_presentScene();
    void LDS_initSDL();

private:
    SDL_Renderer *renderer;
    SDL_Window *window;
    SDL_Texture *texture;
};

const int COLOR_ARR[] = {
    0xc5c8c6, 0x1daf21, 0x0538c6, 0x282a2e, 0x373b41,
    0xa54242, 0xcc6666, 0x8c9440, 0xb5bd68, 0xde935f,
    0xf0c674, 0x5f819d, 0x81a2be, 0x85678f, 0xb294bb,
    0x5e8d87, 0x8abeb7, 0x707880, 0xc5c8c6 
};


#define COLOR_PICK(i) COLOR_ARR[i % 19]

#endif
