#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

#include "rays.hpp"
#include "sdl.hpp"

void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

void loadObjFile(void *t0, void *v0, int *tn, int *vn) {
    std::ifstream ifile("tetrahedron.obj");

    std::string nil;

    std::vector<vec3f> vertices;
    std::vector<int> triangles;

    std::string line;
    while (std::getline(ifile, line)) {
        std::istringstream iss(line);
        switch(line[0]) {
            case 'v':
                vec3f vertex;
                iss >> nil >> vertex.x >> vertex.y >> vertex.z;
                std::cout << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
                vertices.push_back(vertex);
                break;
            case 'f':
                int p1, p2, p3;
                iss >> nil >> p1 >> p2 >> p3;
                triangles.push_back(p1);
                triangles.push_back(p2);
                triangles.push_back(p3);
                std::cout << p1 << " " << p2 << " " << p3 << std::endl;
                break;
            default:
                break;
        }
    }

    *(vec3f**)v0 = new vec3f[vertices.size()];
    memcpy(*(vec3f**)v0, vertices.data(), vertices.size() * sizeof(vec3f));

    *(int**)t0 = new int[triangles.size()];
    memcpy(*(int**)t0, triangles.data(), triangles.size() * sizeof(int));

    std::cout << "v: " << vertices.size() << std::endl;
    std::cout << "t: " << triangles.size()/3 << std::endl;

    *tn = triangles.size()/3;
    *vn = vertices.size();
}

void loadmesh(void *t0, void *v0, int *tn) {
    vec3f *vertices = new vec3f[1000];
    int **triangles = new int*[1000];

    double vert[7][3] = {
        {0, 0, 0}, // MIDDLE
        {1, 0, 0}, // RIGHT
        {0, 0, 1}, // UP
        {1, 0, 1}, // RIGHT UP
        {0, 0, 0}, // MIDDLE
        {-1, 0, 0}, // LEFT
        {-0.5, 0, 0.5} // LEFT UP
    };

    for(int i = 0; i < 7; i++) {
        vertices[i].x = vert[i][0];
        vertices[i].y = vert[i][1];
        vertices[i].z = vert[i][2];
    }

    int triang[3][3] = {
        {0, 2, 1},
        {1, 2, 3},
        {5, 6, 4},
    };

    for(int i = 0; i < 3; i++) {
        triangles[i] = new int[3];
        for(int j = 0; j < 3; j++) {
            triangles[i][j] = triang[i][j];
        }
    }

    *tn = 3;
    *(int***)t0 = triangles;
    *(vec3f**)v0 = vertices;
}

__global__ void drawRay(vec3f origin, vec3f *vertices, int *triangles, int tr, double fovx, double fovy, int *buffer) {
    int h = blockIdx.y * 8 + threadIdx.y;
    int w = blockIdx.x * 8 + threadIdx.x;

    double xr = -tan(w*1.0/SCREEN_WIDTH*fovy - fovy/2);
    double yr = 1;
    double zr = -tan(fovx/2 - h*1.0/SCREEN_HEIGHT*fovx);

    double s = abs(xr) + abs(yr) + abs(zr);
    xr /= s;
    yr /= s;
    zr /= s;

    vec3f dir = {xr, yr, zr};

    vec3f e1, e2;
    vec3f tvec, pvec, qvec;

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
                    intersect = 1; 
                }
            }
        }

        if (intersect) {
            switch(it) {
                case 0:
                    ((int *)buffer)[w + h*SCREEN_WIDTH] = 0xff0000;
                    break;
                case 1:
                    ((int *)buffer)[w + h*SCREEN_WIDTH] = 0x00ff00;
                    break;
                case 2:
                    ((int *)buffer)[w + h*SCREEN_WIDTH] = 0x0000ff;
                    break;
                case 3:
                    ((int *)buffer)[w + h*SCREEN_WIDTH] = 0x00ffff;
                    break;
            }
            return;
        }
    }
}
