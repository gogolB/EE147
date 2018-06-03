CC=gcc
CPP=g++
INC=-I./deps/glm -I./deps/glew/include -I./deps/glfw/include
main.o: main.c
	$(CPP) $(INC) main.c

final_project: main.o
	$(CPP) -o main.o final_proj

clean:
	-rm *.o

