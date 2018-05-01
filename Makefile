

CC=g++
NC=nvcc
CFLAGS=-std=c++11 -O3
NFLAGS=-arch=sm_32

SHARED=shared
TIGR=tigr

all: make1 make2 sssp bfs cc pr sswp

make1:
	make -C $(SHARED)

make2:
	make -C $(TIGR)

sssp: $(TIGR)/sssp.o $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o
	$(NC) $(TIGR)/sssp.o $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o -o sssp $(CFLAGS) $(NFLAGS)
	
bfs: $(TIGR)/bfs.o $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o
	$(NC) $(TIGR)/bfs.o $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o -o bfs $(CFLAGS) $(NFLAGS)

cc: $(TIGR)/cc.o $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o
	$(NC) $(TIGR)/cc.o $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o -o cc $(CFLAGS) $(NFLAGS)
	
pr: $(TIGR)/pr.o $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o
	$(NC) $(TIGR)/pr.o $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o -o pr $(CFLAGS) $(NFLAGS)
	
sswp: $(TIGR)/sswp.o $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o
	$(NC) $(TIGR)/sswp.o $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o -o sswp $(CFLAGS) $(NFLAGS)
	
bc: $(TIGR)/bc.o $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o
	$(NC) $(TIGR)/bc.o $(SHARED)/graph.o $(SHARED)/virtual_graph.o $(SHARED)/argument_parsing.o $(SHARED)/timer.o $(SHARED)/tigr_utilities.o -o bc $(CFLAGS) $(NFLAGS)

clean:
	make -C $(SHARED) clean
	make -C $(TIGR) clean
	rm -f sssp bfs cc pr sswp bc
