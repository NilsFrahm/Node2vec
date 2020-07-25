import argparse
import numpy as np
import datetime
import sys
import os
import pickle
from multiprocessing import Value, Lock, Process
from numba import jit
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim.models.word2vec import LineSentence
import time

#The lock ensures, that only one process writes to a file at a time.

lock = Lock()

def parse_Args():

    """
    Parses the arguments given by the user
    """
    
    parser = argparse.ArgumentParser(description="Run the Node2vec algorithm.")
    
    parser.add_argument("--input", default = "input/karate.edgelist", type = str, nargs = "?", help = "The source path of the intput graph. Default: input\karate")
    parser.add_argument("--output", default = "", type = str, nargs = "?", help = "The name of the output embedding data, which is saved in the output folder. Default: \"datetime + input path\"")
    parser.add_argument("--dimensions", default = 128, type = int, nargs = "?", help = "Number of dimensionss the output embedding is supposed to have. Default: 128")
    parser.add_argument("--num_walks", default = 10, type = int, nargs = "?", help = "Number of walks per node. Deafault: 10")
    parser.add_argument("--walk_length", default = 80, type = int, nargs = "?", help = "The maximum length of a random walk. Default: 80")
    parser.add_argument("--p", default = 1., type = float, nargs = "?", help = "Return parameter. Default: 1")
    parser.add_argument("--q", default = 1., type = float, nargs = "?", help = "Inout parameter. Default: 1")
    parser.add_argument("--weighted", action='store_true', help = "Specifies, if the given graph is weighted or not. Default: False")
    parser.add_argument("--directed", action='store_true', help = "Specifies, if the given graph is directed or not. Default: False")
    parser.add_argument("--window_size", default = 10, type = int, nargs = "?", help = "The windows size used in the Skip-Gram model. Default: 10")
    parser.add_argument("--iter", default = 1, type = int, nargs = "?", help = "Passes over the whole training set done by stochastic gradient descent in the Skip-Gram model. Default: 1")
    parser.add_argument("--cpu_cores", default = 4, type = int, nargs = "?", help = "Number of cores the random walk generation is supposed to use. Default: 4")
    parser.add_argument("--workers", default = 4, type = int, nargs = "?", help = "Number of cores the embedding process is supposed to use. Default: 4")
    parser.add_argument("--low_mem_walks", action='store_true', help = "Precomputes all probability vectors at the cost of RAM consumption. Default: False")
    parser.add_argument("--low_mem_emb", action='store_false', help = "The model is trained via a file on the disk to avoid an overflow of the RAM. Default: True")
    
    return parser.parse_args()
    
def further_parse_output_string(input_string, output_string):

    """
    Prevents giving an error at the very end, if an illegal name for the output file was given.
    """
  
    if output_string == "":

        output_string = "output/" + str(datetime.datetime.now().strftime("%Y_%m_%d %H_%M_%S")) + " " + input_string.replace("/", "_")

        return output_string

    else:

        print("Warning: Be careful with assigning an own name to the output data. If it contains illegal characters, no data will be saved.")
    
    return output_string

def precompute_Adjacency_Structures(input_path, weighted, directed):

    """
    Precomputing of the necessary data structures:
    By the help of the given edge list, some kinds of adjacency lists are computed. 
    adj_set refers to the dictionary 1 in the thesis, while adj_weighted refers to the dictionary 2.
    first_step_probs contains a dictionary, which contains all probability vectors for the first step.
    """  
    
    f = open(input_path)
    adj_set = {}
    adj_weighted = {} 
    first_step_probs = {}    
    count_edges = 0
    
    if weighted:

        edge_weight = {}

        if directed:
    
            for [a, b, c] in [x.split() for x in f]:
            
                count_edges += 1
                adj_set.setdefault(np.int32(a), set()).add(np.int32(b))
                adj_set.setdefault(np.int32(b), set()) 
                edge_weight[(np.int32(a), np.int32(b))] = np.float32(c)

            for a in list(adj_set.keys()):

                neighbour_list = np.array(list(adj_set[a]))
                weight_list = []
                
                for neighbour in neighbour_list:

                    weight_list.append(edge_weight[(a, neighbour)])

                weight_list = np.array(weight_list)
                adj_weighted[a] = (neighbour_list, weight_list) 
                first_step_probs[a] = (neighbour_list, weight_list/weight_list.sum())       
            
        else:
            
            for [a, b, c] in [x.split() for x in f]:
            
                count_edges += 1
                adj_set.setdefault(np.int32(a), set()).add(np.int32(b))
                adj_set.setdefault(np.int32(b), set()).add(np.int32(a))
                edge_weight[(np.int32(a), np.int32(b))] = np.float32(c)
                edge_weight[(np.int32(b), np.int32(a))] = np.float32(c)

            for a in list(adj_set.keys()):

                neighbour_list = np.array(list(adj_set[a]))
                weight_list = []
                
                for neighbour in neighbour_list:

                    weight_list.append(edge_weight[(a, neighbour)])

                weight_list = np.array(weight_list)
                adj_weighted[a] = (neighbour_list, weight_list) 
                first_step_probs[a] = (neighbour_list, weight_list/weight_list.sum()) 

            print(adj_weighted)  
            
    else:
    
        if directed:
        
            for [a, b] in [x.split() for x in f]:
            
                count_edges += 1
                adj_set.setdefault(np.int32(a), set()).add(np.int32(b))
                adj_set.setdefault(np.int32(b), set())                          #nodes without outgoing edges are still present in the dict

            for a in list(adj_set.keys()):
            
                neighbour_list = np.array(list(adj_set[a]))
                adj_weighted[a] = (neighbour_list, np.ones(neighbour_list.size)) 
                first_step_probs[a] = (neighbour_list, np.ones(neighbour_list.size)/neighbour_list.size)                
        
        else:
        
            for [a, b] in [x.split() for x in f]:
            
                count_edges += 1
                adj_set.setdefault(np.int32(a), set()).add(np.int32(b))
                adj_set.setdefault(np.int32(b), set()).add(np.int32(a))
                
            for a in list(adj_set.keys()):
            
                neighbour_list = np.array(list(adj_set[a]))
                adj_weighted[a] = (neighbour_list, np.ones(neighbour_list.size))
                first_step_probs[a] = (neighbour_list, np.ones(neighbour_list.size)/neighbour_list.size)
    
    f.close()
    
    all_nodes = np.array(list(adj_set.keys()))
    
    print("The input has been transformed to adjacency lists.")
    
    if directed:
    
        print("The graph has " + str(all_nodes.size) + " nodes and " + str(count_edges) + " directed edges.")

        return (adj_set, adj_weighted, first_step_probs, all_nodes, count_edges) 
        
    else:
    
        print("The graph has " + str(all_nodes.size) + " nodes and " + str(count_edges) + " undirected edges (corresponds to " + str(2 * count_edges) + " directed edges).")

        return (adj_set, adj_weighted, first_step_probs, all_nodes, 2 * count_edges)     

def precompute_Probs(adj_set, adj_weighted, nodes, p, q, file_number, edges_computed_global, edges_to_compute):

    """
    Transforms the two given dictionaries to a new one, which contains all probability vectors with their corresponding neighbours
    and writes the result to a file.
    """

    probs = {}
    computed_edges = 0
    
    for previous_node in nodes:
    
        neighbours = adj_weighted[previous_node][0]
    
        for current_node in neighbours:
        
            (neighbours, weights) = adj_weighted[current_node]
            prev_neighbours = adj_set[previous_node]
            probs[(previous_node, current_node)] = (compute_Transition_Probabilities(neighbours, prev_neighbours, weights, previous_node, current_node, p, q), neighbours)   

            computed_edges += 1
            edge_Progress(computed_edges, edges_computed_global, edges_to_compute)

    lock.acquire(block = True)
    file_name = "input/intermediate_results/intermediate_probs" + str(file_number.value) + ".data"
    file_number.value += 1
    f = open(file_name, "wb")
    pickle.dump(probs, f)
    f.close()
    lock.release()     
 
def compute_Transition_Probabilities(neighbours, prev_neighbours, weights, previous_node, current_node, p, q):

    """
    Computes the transtion probabilities according to the Node2vec algorithm.
    """

    length = neighbours.size
    probs = np.zeros(length, dtype = np.float32)
    
    if length == 0:
    
        return np.array(())
    
    for i in range(length):
    
        node = neighbours[i]
        weight = weights[i]
        
        if node == previous_node:
        
            probs[i] = weight/p
            
        elif node in prev_neighbours:
        
            probs[i] = weight
            
        else:
        
            probs[i] = weight/q
    
    normalization = np.sum(probs)
            
    return probs/normalization

@jit(nopython=True)
def faster_Sample(neighbours, prob_distr):

    """
    This is the implementation of sampling strategy 2.
    It samples one value uniformly in the interval [0.0,1.0) and iterates over the probability vector until the
    sum of all seen values is larger than the sample.
    It uses numba to compile this method once, such that it can be executed faster.
    """
    
    sample = np.random.random() 
    prob_sum = 0
    
    for x in np.arange(neighbours.size):
    
        prob_sum += prob_distr[x]
        
        if sample <= prob_sum:
        
            return neighbours[x]
            
    return neighbours[-1]
 
def random_Walks(probs, first_step_probs, nodes, walk_length, file_number, walks_computed_global, walks_to_compute):

    """
    Samples the random walks by iterating over the given list of start nodes in nodes
    and writes the result to a file.
    """

    computed_walks = 0
    walks = []
        
    for start_node in nodes:

        (neighbours, prob_distr) = first_step_probs[start_node]

        if neighbours.size == 0:

            walks.append(np.array([start_node]))
            continue  

        else:

            first_step = faster_Sample(neighbours, prob_distr)
            walk = -np.ones(walk_length, dtype = np.int32)
            walk[:2] = start_node, first_step
               
        for i in range(walk_length-2):
            
            (previous_node, current_node) = (walk[i], walk[i+1])
            (prob_distr, neighbours) = probs[(previous_node, current_node)]

            if prob_distr.size == 0:
                
                break
                    
            else:
                
                next = faster_Sample(neighbours, prob_distr)
                walk[i+2] = next

        if walk[i+2] != -1:
            
            walks.append(walk[:i+3])
            computed_walks += 1

            walk_Progress(computed_walks, walks_computed_global, walks_to_compute)
                
        else:
            
            walks.append(walk[:i+2])
            computed_walks += 1

            walk_Progress(computed_walks, walks_computed_global, walks_to_compute)

    while True:

        try:

            lock.acquire()
            file_name = "input/intermediate_results/intermediate_data" + str(file_number.value) + ".data"
            file_number.value += 1
            f = open(file_name, "wb")
            pickle.dump(walks, f)
            f.close()
            lock.release()
            break

        except:

            pass  

def edge_Progress(computed_edges, edges_computed_global, edges_to_compute):

    """
    This function monitors the progress of phase 1. 
    """

    if computed_edges % 25000 == 0:

        edges_computed_global.value += 25000

        if edges_computed_global.value % 100000 == 0:

            print("Probabilities of " + str(edges_computed_global.value) + "/" + str(edges_to_compute) + " edges have been precomputed.")

def walk_Progress(computed_walks, walks_computed_global, walks_to_compute):

    """
    This function monitors the progress of phase 2. 
    """

    if computed_walks % 25000 == 0:

        walks_computed_global.value += 25000

        if walks_computed_global.value % 100000 == 0:

            print(str(walks_computed_global.value) + " random walks of " + str(walks_to_compute) + " have been computed.")

def low_Mem_Random_Walks(adj_set, adj_weighted, nodes, walk_length, file_number, p, q, walks_computed_global, walks_to_compute):

    """
    Samples the random walks by iterating over the given list of start nodes in nodes
    and writes the result to a file.
    Instead of a dictionary of all probability vectors as the random_walks method,
    it computes a probability if needed and discards it afterwards.
    """
    
    computed_walks = 0
    walks = []
        
    for start_node in nodes:

        (neighbours, weights) = (adj_weighted[start_node])

        if neighbours.size == 0:

            walks.append(np.array([start_node]))
            continue  

        else:

            first_step = faster_Sample(neighbours, weights/weights.sum())
            walk = -np.ones(walk_length, dtype = np.int32)
            walk[:2] = start_node, first_step
               
        for i in range(walk_length-2):
            
            (previous_node, current_node) = (walk[i], walk[i+1])
            (neighbours, weights) = adj_weighted[current_node]
            prev_neighbours = adj_set[previous_node]
            prob_distr = compute_Transition_Probabilities(neighbours, prev_neighbours, weights, previous_node, current_node, p, q)

            if prob_distr.size == 0:
                
                break
                    
            else:
                
                next = faster_Sample(neighbours, prob_distr)
                walk[i+2] = next

        if walk[i+2] != -1:
            
            walks.append(walk[:i+3])
            computed_walks += 1

            walk_Progress(computed_walks, walks_computed_global, walks_to_compute)

        else:
            

            walks.append(walk[:i+2])
            computed_walks += 1

            walk_Progress(computed_walks, walks_computed_global, walks_to_compute)

    while True:

        try:

            lock.acquire()
            file_name = "input/intermediate_results/intermediate_data" + str(file_number.value) + ".data"
            file_number.value += 1
            f = open(file_name, "wb")
            pickle.dump(walks, f)
            f.close()
            lock.release()
            break

        except:

            pass

def low_Mem_Random_Walks_Setup(adj_set, adj_weighted, all_nodes, num_walks, walk_length, cpu_cores, p, q):

    """
    This method parallelizes the sampling of random walks, when the low memory mode is enabled.
    """

    starttime = datetime.datetime.now()        
    print("\nPhase 2: Generating Random Walks\n") 

    file_number = Value('i', 0)
    walks_computed_global = Value("i", 0)
    walks_to_compute = all_nodes.size * num_walks
    walks_per_round = int(np.ceil(walks_to_compute / cpu_cores))
    nodes = []
    sub_nodes = []
    processes = []

    for _ in range(num_walks):

        np.random.shuffle(all_nodes)
        nodes.append(all_nodes.copy())

    nodes = np.concatenate(tuple(nodes))

    for proc_number in range(cpu_cores):

        if (proc_number + 1) * walks_per_round < nodes.size:

            sub_nodes = nodes[proc_number * walks_per_round : (proc_number + 1) * walks_per_round]
         
        else:

            sub_nodes = nodes[proc_number * walks_per_round:]

        processes.append(Process(target = low_Mem_Random_Walks, args = (adj_set, adj_weighted, sub_nodes, walk_length, file_number, p, q, walks_computed_global, walks_to_compute)))

    for p in processes:

        p.start()

    return (processes, starttime)
    
def learn_Embedding(walks, dimensions, window_size, cpu_cores, iter, output, low_mem_emb):

    """
    In this method, the embedding is learned. Additionally, it saves the results
    in the corresponding output folder.
    When the low memory mode for phase 3 is enabled (which it is by default),
    it creates a file and writes the random walks to it.
    """

    if low_mem_emb:

        path = os.getcwd() + "/input/intermediate_results/walks.txt"

        f = open(path, "w")

        for _ in range(len(walks)):

            walk = walks.pop()
            walk_str = str(walk[0])
        
            for i in range(walk.size - 1):

                walk_str += (" " + str(walk[i + 1]))

            f.write((walk_str + "\n"))

        f.close()
        gen_walks = LineSentence(datapath(path))
        model = Word2Vec(sentences = gen_walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=cpu_cores, iter=iter)
        model.wv.save_word2vec_format(output)
        os.remove(path)
        
    else:

        walks = [[str(walk_elements) for walk_elements in walk] for walk in walks]
        model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=cpu_cores, iter=iter)
        model.wv.save_word2vec_format(output)

def precompute_Adjacency_Structures_No_Mem_Setup(input, weighted, directed):

    """
    This method starts phase 1, when the low memory flag was set and monitors its runtime.
    """

    starttime = datetime.datetime.now()        
    print("\nPhase 1: Precumputing Adjacency Structures\n") 

    (adj_set, adj_weighted, first_step_probs, all_nodes, count_edges) = precompute_Adjacency_Structures(input, weighted, directed)

    endtime = datetime.datetime.now()        
    print("\nPhase 1 took " + str((endtime - starttime).total_seconds()) + " seconds.\n")

    return (adj_set, adj_weighted, all_nodes)
  
def precompute_Datastructures_Setup(input, weighted, directed, p, q, walk_length, cpu_cores):

    """
    This method parallelizes phase 1. It spawns the processes, waits until all of them finished and gathers the results at the end.
    """

    starttime = datetime.datetime.now()
    print("\nPhase 1: Precomputing Probability Vectors\n")

    (adj_set, adj_weighted, first_step_probs, all_nodes, count_edges) = precompute_Adjacency_Structures(input, weighted, directed)
    nodes_per_round = int(np.ceil(all_nodes.size / cpu_cores))
    np.random.shuffle(all_nodes)
    file_number = Value('i', 0)
    edges_computed_global = Value('i', 0)
    probs = {}
    processes = []
    
    for proc_number in range(cpu_cores):

        if (proc_number + 1) * nodes_per_round < all_nodes.size:

            nodes = all_nodes[proc_number* nodes_per_round : (proc_number + 1) * nodes_per_round]
         
        else:

            nodes = all_nodes[proc_number* nodes_per_round:]

        processes.append(Process(target = precompute_Probs, args = (adj_set, adj_weighted, nodes, p, q, file_number, edges_computed_global, count_edges)))

    for p in processes:

        p.start()

    for p in processes:

        p.join()

    for iter in range(cpu_cores):

        file_name = "input/intermediate_results/intermediate_probs" + str(iter) + ".data"
        f = open(file_name, "rb+")
        probs.update(pickle.load(f))
        f.close()
        os.remove(file_name)
   
    endtime = datetime.datetime.now()        
    print("\nPhase 1 took " + str((endtime - starttime).total_seconds()) + " seconds.\n")
        
    return (probs, first_step_probs, all_nodes)
        
def random_Walks_Setup(probs, first_step_probs, all_nodes, num_walks, walk_length, cpu_cores):

    """
    This method parallelizes phase 2. It spawns the processes.
    """

    starttime = datetime.datetime.now()        
    print("\nPhase 2: Generating Random Walks\n") 

    file_number = Value('i', 0)
    walks_computed_global = Value("i", 0)
    walks_to_compute = all_nodes.size * num_walks
    walks_per_round = int(np.ceil(walks_to_compute / cpu_cores))
    nodes = []
    sub_nodes = []
    processes = []

    for _ in range(num_walks):

        np.random.shuffle(all_nodes)
        nodes.append(all_nodes.copy())

    nodes = np.concatenate(tuple(nodes))

    for proc_number in range(cpu_cores):

        if (proc_number + 1) * walks_per_round < nodes.size:

            sub_nodes = nodes[proc_number * walks_per_round : (proc_number + 1) * walks_per_round]
         
        else:

            sub_nodes = nodes[proc_number * walks_per_round:]

        processes.append(Process(target = random_Walks, args = (probs, first_step_probs, sub_nodes, walk_length, file_number, walks_computed_global, walks_to_compute)))

    for p in processes:

        p.start()

    return (processes, starttime)

def read_Random_Walk_Data(processes, starttime, cpu_cores):

    """
    Aggregates all sampled random walks and returns them.
    """

    for p in processes:

        p.join()

    walks = []

    for iter in range(cpu_cores):

        file_name = "input/intermediate_results/intermediate_data" +str(iter) + ".data"
        f = open(file_name, "rb+")
        walks.extend(pickle.load(f))
        f.close()
        os.remove(file_name)
    
    endtime = datetime.datetime.now()       
    print("\nPhase 2 took " + str((endtime - starttime).total_seconds()) + " seconds.\n")
    
    return walks
    
def learn_Embedding_Setup(walks, dimensions, window_size, cpu_cores, iter, output, low_mem_emb):

    """
    This method monitors the runtime of phase 3.
    """

    starttime = datetime.datetime.now()
    print("\nPhase 3: Computing the Embedding\n") 
             
    learn_Embedding(walks, dimensions, window_size, cpu_cores, iter, output, low_mem_emb)   
     
    endtime = datetime.datetime.now()        
    print("\nPhase 3 took " + str((endtime - starttime).total_seconds()) + " seconds.\n")    
  
def node2vec(args): 

    """
    The three phases of the Node2vec algorithm are executed sequentially.
    In phase 2, some data structures are deleted, after they are not needed anymore.
    """
    
    if __name__ == '__main__':

        if args.low_mem_walks:
            
            (adj_set, adj_weighted, all_nodes) = precompute_Adjacency_Structures_No_Mem_Setup(args.input, args.weighted, args.directed)

            (processes, starttime) = low_Mem_Random_Walks_Setup(adj_set, adj_weighted, all_nodes, args.num_walks, args.walk_length, args.cpu_cores, args.p, args.q)                                                                     
            del adj_set
            del adj_weighted
            del all_nodes
            walks = read_Random_Walk_Data(processes, starttime, args.cpu_cores)  
 
            learn_Embedding_Setup(walks, args.dimensions, args.window_size, args.workers, args.iter, args.output, args.low_mem_emb)

        else:
            
            (probs, first_step_probs, all_nodes) = precompute_Datastructures_Setup(args.input, args.weighted, args.directed, args.p, args.q, args.walk_length, args.cpu_cores)  
            
            (processes, starttime) = random_Walks_Setup(probs, first_step_probs, all_nodes, args.num_walks, args.walk_length, args.cpu_cores)
            del probs
            del first_step_probs
            del all_nodes
            walks = read_Random_Walk_Data(processes, starttime, args.cpu_cores) 
      
            learn_Embedding_Setup(walks, args.dimensions, args.window_size, args.workers, args.iter, args.output, args.low_mem_emb)

"""
After parsing the input, the main algorithm is called.
"""

args = parse_Args()
args.output = further_parse_output_string(args.input, str(args.output))
node2vec(args)
