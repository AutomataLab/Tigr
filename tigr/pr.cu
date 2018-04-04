
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
							float *pr1,
							float *pr2)
{
	int partId = blockDim.x * blockIdx.x + threadIdx.x;

	if(partId < numParts)
	{
		int id = partNodePointer[partId].node;
		int part = partNodePointer[partId].part;


		// int sourceWeight = dist[id];
		
		int thisPointer = nodePointer[id];
		int degree = edgeList[thisPointer];
		
		float sourcePR = (float) pr2[id] / degree;
		
			
		int numParts;
		if(degree % Part_Size == 0)
			numParts = degree / Part_Size ;
		else
			numParts = degree / Part_Size + 1;
		
		int end;
		// int w8;
		int ofs = thisPointer + part + 1;

		for(int i=0; i<Part_Size; i++)
		{
			if(part + i*numParts >= degree)
				break;
			end = ofs + i*numParts;
			// w8 = end + 1;

			atomicAdd(&pr1[edgeList[end]], sourcePR);
		}	
	}
}

__global__ void clearLabel(float *prA, float *prB, unsigned int num_nodes, float base)
{
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < num_nodes)
	{
		prA[id] = base + prA[id] * 0.85;
		prB[id] = 0;
	}
}

int main(int argc, char** argv)
{
	ArgumentParser arguments(argc, argv, false, true);
		
	Graph graph(arguments.input, false);
	graph.ReadGraph();
	
	VirtualGraph vGraph(graph);
	
	vGraph.MakeUGraph();
	
	uint num_nodes = graph.num_nodes;
	uint num_edges = graph.num_edges;
	

	if(arguments.hasDeviceID)
		cudaSetDevice(arguments.deviceID);	

	cudaFree(0);
	
	float *pr1, *pr2;
	pr1  = new float[num_nodes];
	pr2  = new float[num_nodes];
	
	float initPR = (float) 1 / num_nodes;
	cout << initPR << endl;
	
	for(int i=0; i<num_nodes; i++)
	{
		pr1[i] = 0;
		pr2[i] = initPR;
	}

	unsigned int *d_nodePointer;
	unsigned int *d_edgeList;
	PartPointer *d_partNodePointer; 
	float *d_pr1;
	float *d_pr2;
	

	gpuErrorcheck(cudaMalloc(&d_nodePointer, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_edgeList, (num_edges + num_nodes) * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_pr1, num_nodes * sizeof(float)));
	gpuErrorcheck(cudaMalloc(&d_pr2, num_nodes * sizeof(float)));
	gpuErrorcheck(cudaMalloc(&d_partNodePointer, vGraph.numParts * sizeof(PartPointer)));

	gpuErrorcheck(cudaMemcpy(d_nodePointer, vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_edgeList, vGraph.edgeList, (num_edges + num_nodes) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_pr1, pr1, num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_pr2, pr2, num_nodes * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrorcheck(cudaMemcpy(d_partNodePointer, vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer), cudaMemcpyHostToDevice));

	Timer t;
	t.Start();

	int itr = 0;
	// make it fast
	float base = (float)0.15/num_nodes;
	do
	{
		itr++;
		if(itr % 2 == 1)
		{
			kernel<<< vGraph.numParts/512 + 1 , 512 >>>(vGraph.numParts, 
														d_nodePointer,
														d_partNodePointer,
														d_edgeList, 
														d_pr1,
														d_pr2);
			clearLabel<<< num_nodes/512 + 1 , 512 >>>(d_pr1, d_pr2, num_nodes, base);
		}
		else
		{
			kernel<<< vGraph.numParts/512 + 1 , 512 >>>(vGraph.numParts, 
														d_nodePointer, 
														d_partNodePointer,
														d_edgeList,
														d_pr2,
														d_pr1);
			clearLabel<<< num_nodes/512 + 1 , 512 >>>(d_pr2, d_pr1, num_nodes, base);														
		}
	
		gpuErrorcheck( cudaPeekAtLastError() );
		gpuErrorcheck( cudaDeviceSynchronize() );
		
	} while(itr < arguments.numberOfItrs);
	
	cudaDeviceSynchronize();
	
	cout << "Number of iterations = " << itr << endl;

	float runtime = t.Finish();
	cout << "Processing finished in " << runtime << " (ms).\n";
		
	
	if(itr % 2 == 1)
	{
		gpuErrorcheck(cudaMemcpy(pr1, d_pr1, num_nodes*sizeof(float), cudaMemcpyDeviceToHost));
	}
	else
	{
		gpuErrorcheck(cudaMemcpy(pr1, d_pr2, num_nodes*sizeof(float), cudaMemcpyDeviceToHost));
	}

		utilities::PrintResults(pr1, 30);
		
	//float sum = 0;
	//for(int i=0; i<num_nodes; i++)
	//	sum = sum + pr1[i];
	//cout << sum << endl << endl;

	if(arguments.hasOutput)
		utilities::SaveResults(arguments.output, pr1, num_nodes);	

	gpuErrorcheck(cudaFree(d_nodePointer));
	gpuErrorcheck(cudaFree(d_edgeList));
	gpuErrorcheck(cudaFree(d_pr1));
	gpuErrorcheck(cudaFree(d_pr2));
	gpuErrorcheck(cudaFree(d_partNodePointer));

}
