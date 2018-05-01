

CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_32


all: graph.o virtual_graph.o argument_parsing.o timer.o tigr_utilities.o

graph.o: graph.cpp
	$(CC) -c graph.cpp -o graph.o $(CFLAGS)

virtual_graph.o: virtual_graph.cpp
	$(CC) -c virtual_graph.cpp -o virtual_graph.o $(CFLAGS)
	
argument_parsing.o: argument_parsing.cpp
	$(CC) -c argument_parsing.cpp -o argument_parsing.o $(CFLAGS)

timer.o: timer.cpp
	$(CC) -c timer.cpp -o timer.o $(CFLAGS)

tigr_utilities.o: tigr_utilities.cpp
	$(CC) -c tigr_utilities.cpp -o tigr_utilities.o $(CFLAGS)

clean:
	rm *.o 
