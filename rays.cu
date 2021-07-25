#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <assert.h>

#include "rays.hpp"
#include "sdl.hpp"

#define IFDEBUG(...)

using std::vector;

__device__ mat3f rotation3d(float ang, int l, int m, int n) {
    mat3f mat;
    mat.a0 = l*l*(1 - cos(ang)) + cos(ang);
    mat.b0 = m*l*(1 - cos(ang)) - sin(ang) * n;
    mat.c0 = n*l*(1 - cos(ang)) + sin(ang) * m;
    mat.a1 = l*m*(1 - cos(ang)) + sin(ang) * n;
    mat.b1 = m*m*(1 - cos(ang)) + cos(ang);
    mat.c1 = n*m*(1 - cos(ang)) - sin(ang) * l;
    mat.a2 = l*n*(1 - cos(ang)) - sin(ang) * m;
    mat.b2 = m*n*(1 - cos(ang)) + sin(ang) * l;
    mat.c2 = n*n*(1 - cos(ang)) + cos(ang);
    return mat;
}

__device__ vec3f multvec3f(vec3f v, mat3f m) {
    vec3f r;

    r.x = v.x * m.a0 + v.y * m.b0 + v.z * m.c0;
    r.y = v.x * m.a1 + v.y * m.b1 + v.z * m.c1;
    r.z = v.x * m.a2 + v.y * m.b2 + v.z * m.c2;

    return r;
}

void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

void loadObjFile(vector<vec3i> &triangles, vector<vec3f> &vertices, vector<vec3f> &normals) {
    std::ifstream ifile("obj/teapot.obj");

    std::string nil;

    std::string line;
    while (std::getline(ifile, line)) {
        std::istringstream iss(line);
        switch(line[0]) {
            case 'v':
                vec3f vertex;
                iss >> nil >> vertex.x >> vertex.y >> vertex.z;
                vertices.push_back(vertex);
                break;
            case 'f':
                vec3i triangle;
                iss >> nil >> triangle.x >> triangle.y >> triangle.z;
                triangles.push_back(triangle);
                break;
            case 'n':
                vec3f normal;
                iss >> nil >> normal.x >> normal.y >> normal.z;
                normals.push_back(normal);
                break;
            default:
                break;
        }
    }

    std::cout << "v: " << vertices.size() << std::endl;
    std::cout << "n: " << normals.size() << std::endl;
    std::cout << "t: " << triangles.size()/3 << std::endl;
}

__global__ void drawRay(vec3f camera, float ang, const vec3f *vertices, const vec3f *normals, int vn, const int *triangles, int tr, double fovx, double fovy, int *buffer) {
    int h = blockIdx.y * 8 + threadIdx.y;
    int w = blockIdx.x * 8 + threadIdx.x;

    // POINT IN IMAGE PLANE
    vec3f p;
    p.x = w*2.0f/SCREEN_WIDTH - 1.f;
    p.y = 1.0f/(tan(fovx/2));
    p.z = h*2.0f/SCREEN_HEIGHT - 1.f;

    mat3f camRot = rotation3d(ang, 1, 0, 0); // l=0, m=0, n=1

    IFDEBUG(abs(p.x) < 1.0f || h || w);
    IFDEBUG(abs(p.z) < 1.0f || h || w);

    // A RAY
    vec3f ray = multvec3f(p, camRot);

    vec3f e1, e2;
    vec3f tvec, pvec, qvec;

    float mt = 0;

    for(int it = 0; it < tr; it++) {
        assert(3*it + 2 < tr*3);

        int t0 = triangles[3*it + 0];
        int t1 = triangles[3*it + 1];
        int t2 = triangles[3*it + 2];

        IFDEBUG(t0 < vn);
        IFDEBUG(t1 < vn);
        IFDEBUG(t2 < vn);

        // Copy three vertices of a triangle
        vec3f v0 = vertices[t0];
        vec3f v1 = vertices[t1];
        vec3f v2 = vertices[t2];

        int intersect = 0;
        double u, v, t, det, inv_det;

        sub(e1, v1, v0);
        sub(e2, v2, v0);

        cross(pvec, ray, e2);

        det = dot(pvec, e1);

        if (det > -EPSILON && det < EPSILON) {
            intersect = 0;
        } else {
            inv_det = 1.0 / det;
            sub(tvec, camera, v0);
            u = dot(tvec, pvec) * inv_det;
            if(u < 0.0 || u > 1.0) {
                intersect = 0;
            } else {
                cross(qvec, tvec, e1);
                v = dot(ray, qvec) * inv_det;
                if(v < 0.0 || u + v > 1.0)
                    intersect = 0;
                else {
                    t = dot(e2, qvec) * inv_det;
                    intersect = t > 0; 
                }
            }
        }

        if (intersect && t > mt) {
            vec3f vn = normals[t0];

            float mul = dot(vn, ray) / 3.0f;
            int col = (int)(255.0 * mul) & 0xff;

            mt = t;

            buffer[w + h*SCREEN_WIDTH] = col | (col << 8) | (col << 16); 
        }
    }
}
