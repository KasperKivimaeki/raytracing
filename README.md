# raytracing

This is my experimental project where I try to incorporate the things I have learned from Aalto University's courses Computer Graphics and Parallel Programming Computers.


### Building

To build the project you would need to have [CUDA tools](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#installing-cuda-development-tools) installed. 

You also need to install SDL and g++. 

```bash
make
```

This should settle things.

### Current project status

When I have time and motivation I will implement a proper working camera and BVH so drawing a single frame wont have the time complexity of O(T) where T is the number of triangles in the world. 
