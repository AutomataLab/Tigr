## Tigr: Transforming Irregular Graphs for GPU-Friendly Graph Processing

Tigr is a lightweight graph transformation and processing framework for GPU platforms. 

In real-world graphs, the high irregularity of degree distribution acts
as a major barrier to their efficient processing on GPU architectures.
Tigr addresses the irregularity issue at its origin by transforming 
irregular graphs into more regular ones, meanwhile preserving the same
results as running on the original graphs.

#### Compilation

To compile Tigr, just run *make* in the root directory.

#### Running applications in Tigr

The applications take the input graph as input as well as some optional arguments. For example:

```
$ ./sssp --input path-to-input-graph
$ ./sssp --input path-to-input-graph --source 10
``` 

#### Input graph format

Input graphs should be in form of plain text files, containing the list of the edges of the graph. Each line is corresponding to an edge and is of the following form:

```
V1  V2  W
```

It specifies that there is an edge from node V1 to node V2 with weight W. The Wight value is optional and if it is omitted, it is set to 1. The node-ids can start from 0 or 1. It ignores any line starting with a character rather than a number.

Graphs in this format can be found in many public graph repositories, such as SNAP's. There are some graph datasets ready to download in *datasets* folder. To download, just run *make* in each folder.

#### Publications:

[ASPLOS'18] Amir Nodehi, Junqiao Qiu, Zhijia Zhao. Tigr: Transforming
Irregular Graphs for GPU-Friendly Graph Processing. In Proceedings of
The 23th International Conference on Architectural Support for
Programming Languages and Operating Systems, Williamsburg, VA, 2018. 15
pages

