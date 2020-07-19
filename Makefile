SDL = `sdl2-config --static-libs --cflags`

engine: main.o sdl.o rays.o
	gcc $+ -o engine $(SDL)

main.o : main.c
	gcc $(SDL) -c $^

sdl.o: sdl.c
	gcc $(SDL) -c $^

rays.o: rays.c
	gcc $(SDL) -c $^

clean: 
	rm -f *.o engine
