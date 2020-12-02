#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#include "rays.hpp"
#include "sdl.hpp"

using std::vector;

void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

void loadObjFile(vector<vec3i> &triangles, vector<vec3f> &vertices) {
    std::ifstream ifile("obj/tetrahedron.obj");

    std::string nil;

    std::string line;
    while (std::getline(ifile, line)) {
        std::istringstream iss(line);
        switch(line[0]) {
            case 'v':
                vec3f vertex;
                iss >> nil >> vertex.x >> vertex.y >> vertex.z;
                vertices.push_back(vertex);
                //std::cout << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
                break;
            case 'f':
                vec3i triangle;
                iss >> nil >> triangle.x >> triangle.y >> triangle.z;
                triangles.push_back(triangle);
                //std::cout << p1 << " " << p2 << " " << p3 << std::endl;
                break;
            default:
                break;
        }
    }

    std::cout << "v: " << vertices.size() << std::endl;
    std::cout << "t: " << triangles.size()/3 << std::endl;
}

__global__ void drawRay(vec3f origin, float ang, vec3f *vertices, int *triangles, int tr, double fovx, double fovy, int *buffer) {
    int h = blockIdx.y * 8 + threadIdx.y;
    int w = blockIdx.x * 8 + threadIdx.x;

    double xr = tan(w*1.0/SCREEN_WIDTH*fovy - fovy/2);
    double yr = 1;
    double zr = tan(fovx/2 - h*1.0/SCREEN_HEIGHT*fovx);

    double s = abs(xr) + abs(yr) + abs(zr);
    xr /= s;
    yr /= s;
    zr /= s;

    xr += cos(ang);

    vec3f dir = {xr, yr, zr};

    vec3f e1, e2;
    vec3f tvec, pvec, qvec;

    float mt = 0;

    for(int it = 0; it < tr; it++) {
        // Copy three vertices of a triangle
        vec3f v0 = vertices[triangles[3*it + 0]];
        vec3f v1 = vertices[triangles[3*it + 1]];
        vec3f v2 = vertices[triangles[3*it + 2]];

        int intersect = 0;
        double u, v, t, det, inv_det;

        sub(e1, v1, v0);
        sub(e2, v2, v0);

        cross(pvec, dir, e2);

        det = dot(pvec, e1);

        if (det > -EPSILON && det < EPSILON) {
            intersect = 0;
        } else {
            inv_det = 1.0 / det;
            sub(tvec, origin, v0);
            u = dot(tvec, pvec) * inv_det;
            if(u < 0.0 || u > 1.0) {
                intersect = 0;
            } else {
                cross(qvec, tvec, e1);
                v = dot(dir, qvec) * inv_det;
                if(v < 0.0 || u + v > 1.0)
                    intersect = 0;
                else {
                    t = dot(e2, qvec) * inv_det;
                    intersect = t > 0; 
                }
            }
        }

        if (intersect && t > mt) {
            mt = t;

            int a = (int)(mt*mt);
            int s = 0xff;
            ((int *)buffer)[w + h*SCREEN_WIDTH] = (a << 16) + (a << 8) + a;

            switch(it) {
                case 0:
                    ((int *)buffer)[w + h*SCREEN_WIDTH] |= s << 16;
                    break;
                case 1:
                    ((int *)buffer)[w + h*SCREEN_WIDTH] |= s << 8;
                    break;
                case 2:
                    ((int *)buffer)[w + h*SCREEN_WIDTH] |= s;
                    break;
                case 3:
                    ((int *)buffer)[w + h*SCREEN_WIDTH] |= (s << 8) + s;
                    break;
            }
        }
    }
}
