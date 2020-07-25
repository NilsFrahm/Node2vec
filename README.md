# Implementation of Node2vec

This repository provides a new implementation of *node2vec*, which was presented in the following paper:<br>
> node2vec: Scalable Feature Learning for Networks.<br>
> Aditya Grover and Jure Leskovec.<br>
> Knowledge Discovery and Data Mining, 2016.<br>

## Requirements

* Python 3.6
* A recent version of Linux. Windows works as well, but the performance is much worse.

## Basic Usage

### Example

To run this implementation on Zachary's karate club network, the following command has to be executed:<br/>
	``python node2vec.py --input input/karate.edgelist``<br/>
The embedding is then saved in the output folder.

### Options

To explore all other available options, you can use the following command:<br/>
	``python src/main.py --help``
    
### Input

The input file has to be an edgelist, i.e. every line must have the following format:

	node1_id_int node2_id_int
		
The graph is assumed to be undirected and unweighted by default. If the --weighted flag was set, every line must have the following format:

    node1_id_int node2_id_int weight
    
### Output

The output file consists of *n+1*, where n is the amount of vertices in the input graph.
The format of the first line is the following:

	num_of_nodes dim_of_representation

The format of all other lines are as follows:
	
	node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by *node2vec*.