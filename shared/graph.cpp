
#include "graph.hpp"



Graph::Graph(string graphFilePath, bool isWeighted)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
	graphLoaded = false;
	hasZeroID = false;
}


void Graph::ReadGraph()
{
	cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
		
	ifstream infile;
	infile.open(graphFilePath);
	
	stringstream ss;
	
	uint max = 0;

	if(graphLoaded == true)
	{
		edges.clear();
		weights.clear();
	}	
	
	graphLoaded = true;
	
	uint source;
	uint end;
	uint w8;
	uint i = 0;
	
	string line;
	
	Edge newEdge;
	
	unsigned long edgeCounter = 0;
	
	while(getline( infile, line ))
	{
		if(line[0] < '0' || line[0] > '9')
			continue;
			
		ss.str("");
		ss.clear();
		ss << line;
		
		ss >> newEdge.source;
		ss >> newEdge.end;
		
		edges.push_back(newEdge);
		
		if (newEdge.source == 0)
			hasZeroID = true;
		if (newEdge.end == 0)
			hasZeroID = true;			
		if(max < newEdge.source)
			max = newEdge.source;
		if(max < newEdge.end)
			max = newEdge.end;
		
		if (isWeighted)
		{
			if (ss >> w8)
				weights.push_back(w8);
			else
				weights.push_back(1);
		}
		
		edgeCounter++;
	}
	
	infile.close();
	
	graphLoaded = true;
	
	num_edges = edgeCounter;
	num_nodes = max;
	if (hasZeroID)
		num_nodes++;
		
	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;
}
