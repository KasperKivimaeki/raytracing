SDL = -lSDL2 -I lib -lm

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
