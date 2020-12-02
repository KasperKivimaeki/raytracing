#include "sdl.hpp"
#include "tracer.hpp"

int main(int argc, char *argv[]) {
    App *app = new App();
    Tracer *t = new Tracer(app);

    t->run();

    return 0;
}
