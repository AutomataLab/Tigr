## Tigr: Transforming Irregular Graphs for GPU-Friendly Graph Processing

Tigr is a lightweight graph transformation and processing framework for GPU platforms. 

In real-world graphs, the high irregularity of degree distribution acts
as a major barrier to their efficient processing on GPU architectures.
Tigr addresses the irregularity issue at its origin by transforming 
irregular graphs into more regular ones, meanwhile preserving the same
results as running on the original graphs.

#### Compilation

To compile Tigr, just run make in the root directory.

#### Running applications in Tigr

The applications take the input graph as input as well as some optional arguments. For example:

```
$ ./sssp --input path-to-input-graph
$ ./sssp --input path-to-input-graph --source 10
``` 

#### Publications:

[ASPLOS'18] Amir Nodehi, Junqiao Qiu, Zhijia Zhao. Tigr: Transforming
Irregular Graphs for GPU-Friendly Graph Processing. In Proceedings of
The 23th International Conference on Architectural Support for
Programming Languages and Operating Systems, Williamsburg, VA, 2018. 15
pages

