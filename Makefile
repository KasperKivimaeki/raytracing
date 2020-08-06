GXX=nvcc

SDL = -Xcompiler "-Wall --std=c++11 -lm -ldl -lasound -lm -ldl -lpthread -lpulse-simple -lpulse -lX11 -lXext -lXcursor -lXinerama -lXi -lXrandr -lXss -lXxf86vm -lwayland-egl -lwayland-client -lwayland-cursor -lxkbcommon -lpthread -lrt -I/usr/include/SDL2 -D_REENTRANT" -lSDL2

%.o: %.cu
	$(GXX) $(SDL) -c $<

engine: main.o sdl.o rays.o
	$(GXX) $+ -o engine $(SDL)
clean: 
	rm -f *.o engine

