#ifndef RAYS_H
#define RAYS_H

#include <cuda_runtime.h>
#include <vector>

struct VEC3F {
    double x;
    double y;
    double z;
};

struct VEC3I {
    int x;
    int y;
    int z;
};

void check(cudaError_t err, const char* context);
#define CHECK(x) check(x, #x)

#define EPSILON 0.0000001

#define vec3f struct VEC3F
#define vec3i struct VEC3I

#define cross(dest, a, b)\
    dest.x = a.y*b.z - a.z*b.y;\
    dest.y = a.z*b.x - a.x*b.z;\
    dest.z = a.x*b.y - a.y*b.x;\

#define dot(a, b) (a.x*b.x + a.y*b.y + a.z*b.z)

#define sub(dest, a, b)\
    dest.x = a.x - b.x;\
    dest.y = a.y - b.y;\
    dest.z = a.z - b.z;\

#define add(dest, a, b)\
    dest.x = a.x + b.x;\
    dest.y = a.y + b.y;\
    dest.z = a.z + b.z;\

#define print3(s, x) printf("%s %f %f %f", s, x[0], x[1], x[2]);

// Triangles, vertices, amount
void loadObjFile(std::vector<vec3i> &triangles, std::vector<vec3f> &vertices);

__global__ void drawRay(vec3f origin, float, vec3f *vertices, int *triangles, int tr, double fovx, double fovy, int *buffer);

#endif
