
#include "../shared/timer.hpp"
#include "../shared/tigr_utilities.hpp"
#include "../shared/graph.hpp"
#include "../shared/virtual_graph.hpp"
#include "../shared/globals.hpp"
#include "../shared/argument_parsing.hpp"
#include "../shared/gpu_error_check.cuh"




__global__ void kernel(unsigned int numParts, 
							unsigned int *nodePointer, 
							PartPointer *partNodePointer,
							unsigned int *edgeList, 
							unsigned int *dist, 
							bool *finished,
							int level)
{
	unsigned int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
		unsigned int id = partNodePointer[partId].node;
		unsigned int part = partNodePointer[partId].part;

		if(dist[id] != level)
			return;

		unsigned int thisPointer = nodePointer[id];
		unsigned int degree = edgeList[thisPointer];
			
		unsigned int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;

		
		unsigned int end;

		unsigned int ofs = thisPointer + part + 1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts;
			
			if(dist[edgeList[end]] == DIST_INFINITY)
			{
				dist[edgeList[end]] = level + 1;
				*finished = false;
			}
		}
		
	}
}


__global__ void clearLabel(bool *label, unsigned int size)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < size)
		label[id] = false;
}


int main(int argc, char** argv)
{	
	ArgumentParser arguments(argc, argv, true, false);
	
	Graph graph(arguments.input, false);
	graph.ReadGraph();
	
	VirtualGraph vGraph(graph);
	
	vGraph.MakeUGraph();
	
	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;
	
	if(arguments.hasDeviceID)
		cudaSetDevice(arguments.deviceID);

	cudaFree(0);
	
	unsigned int *dist;
	dist  = new unsigned int[num_nodes];

	for(int i=0; i<num_nodes; i++)
	{
		dist[i] = DIST_INFINITY;
	}
	dist[arguments.sourceNode] = 0;
	

	unsigned int *d_nodePointer;
	unsigned int *d_edgeList;
	unsigned int *d_dist;
	PartPointer *d_partNodePointer; 
	
	bool finished;
	bool *d_finished;

	gpuErrorcheck(cudaMalloc(&d_nodePointer, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_edgeList, (num_edges + num_nodes) * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_dist, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_partNodePointer, vGraph.numParts * sizeof(PartPointer)));

	gpuErrorcheck(cudaMemcpy(d_nodePointer, vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_edgeList, vGraph.edgeList, (num_edges + num_nodes) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_dist, dist, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_partNodePointer, vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemcpyHostToDevice));

	Timer t;
	t.Start();

	int itr = 0;
	int level = 0;
	do
	{
		itr++;
		finished = true;
		gpuErrorcheck(cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice));
		if(itr % 2 == 1)
		{
			kernel<<< vGraph.numParts/512 + 1 , 512 >>>(vGraph.numParts, 
														d_nodePointer,
														d_partNodePointer,
														d_edgeList, 
														d_dist, 
														d_finished,
														level);
		}
		else
		{
			kernel<<< vGraph.numParts/512 + 1 , 512 >>>(vGraph.numParts, 
														d_nodePointer, 
														d_partNodePointer,
														d_edgeList, 
														d_dist, 
														d_finished,
														level);													
		}
	
		gpuErrorcheck( cudaPeekAtLastError() );
		gpuErrorcheck( cudaDeviceSynchronize() );
		
		gpuErrorcheck(cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost));
		
		level++;
		
	} while (!(finished));
	
	cout << "Number of iterations = " << itr << endl;

	float runtime = t.Finish();
	cout << "Processing finished in " << runtime << " (ms).\n";
		
	
	gpuErrorcheck(cudaMemcpy(dist, d_dist, num_nodes*sizeof(unsigned int), cudaMemcpyDeviceToHost));

	utilities::PrintResults(dist, 30);
	
	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, dist, num_nodes);
	

	gpuErrorcheck(cudaFree(d_nodePointer));
	gpuErrorcheck(cudaFree(d_edgeList));
	gpuErrorcheck(cudaFree(d_dist));
	gpuErrorcheck(cudaFree(d_finished));
	gpuErrorcheck(cudaFree(d_partNodePointer));

}
