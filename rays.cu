#include "rays.h"
#include <stdio.h>
#include <stdlib.h>

void loadmesh(int** triangles, vec3f* vertices, int* tr) {
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

    *tr = 3;
}
