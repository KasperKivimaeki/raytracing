SDL = -lGL -lSDL2 -I /usr/include/SDL2/ -D_REENTRANT -L/usr/lib -pthread -lm

engine: main.o sdl.o rays.o
	gcc $(SDL) $? -o engine

main.o : main.c
	gcc $(SDL) -c $^

sdl.o: sdl.c
	gcc $(SDL) -c $^

rays.o: rays.c
	gcc $(SDL) -c $^

clean: 
	rm -f *.o engine
