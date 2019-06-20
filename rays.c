#include "rays.h"
#include <stdio.h>
#include <stdlib.h>

int intersects(vec3f origin, vec3f dir, vec3f v0, vec3f v1, vec3f v2) {

    vec3f e1, e2;
    vec3f tvec, pvec, qvec;

    double epsilon = 0.0001;

    sub(e1, v1, v0);
    sub(e2, v2, v0);

    sub(tvec, origin, v0);
    cross(pvec, dir, e2);
    cross(qvec, tvec, e1);

    double det = dot(pvec, e1);

    if(det < epsilon && det > -epsilon) {
        return 2;
    }

    double inv_det = 1.0 / det;

    double u = dot(pvec, tvec) * inv_det;
    if(u < 0 || u > 1)
        return 0;

    double v = dot(qvec, dir) * inv_det;
    if(v < 0 || v + u > 1)
        return 0;

    return 1;
}

void loadmesh(int** triangles, vec3f* vertices, int* tr) {
    double vert[7][3] = {
        {0, 0, 0},
        {1, 0, 0},
        {0, 0, 1},
        {1, 0, 1},
        {0, 0, 0},
        {-1, 0, 0},
        {-0.5, 0, 0.5}
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
        triangles[i] = malloc(sizeof(int) * 3);
        for(int j = 0; j < 3; j++) {
            triangles[i][j] = triang[i][j];
        }
    }

    *tr = 3;
}
