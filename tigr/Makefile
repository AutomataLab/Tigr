
CC=g++
NC=nvcc
CFLAGS=-std=c++11
NFLAGS=-arch=sm_32

SHARED=../shared

all: sssp.o bfs.o cc.o pr.o sswp.o bc.o

sssp.o: sssp.cu
	$(NC) -c sssp.cu $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o  $(CFLAGS) $(NFLAGS)
	
bfs.o: bfs.cu
	$(NC) -c bfs.cu $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o $(CFLAGS) $(NFLAGS)

cc.o: cc.cu
	$(NC) -c cc.cu $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o $(CFLAGS) $(NFLAGS)
	
pr.o: pr.cu
	$(NC) -c pr.cu $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o $(CFLAGS) $(NFLAGS)
	
sswp.o: sswp.cu
	$(NC) -c sswp.cu $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o $(CFLAGS) $(NFLAGS)
	
bc.o: bc.cu
	$(NC) -c bc.cu $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o $(CFLAGS) $(NFLAGS)
	
clean:
	rm  *.o 
