#ifndef TRACER_HPP
#define TRACER_HPP

#include "sdl.hpp"
#include "rays.hpp"

#include <chrono>
#include <vector>

#define PI 3.14159265

#define CURRENT_TIME std::chrono::high_resolution_clock::now()

class Tracer {
public:
    Tracer(App *);
    ~Tracer();

    void run();

private:
    void draw() const;
    void getInput();

    int frame;

    int click;

    std::vector<vec3f> vertices;
    std::vector<vec3i> triangles;

    int* buffer;

    int *gBuf, *tBuf;
    vec3f *vBuf;

    float fovx;
    float fovy;

    double sx1;
    double sy1;

    int ang;

    vec3f origin;

    App *app;
};

#endif
