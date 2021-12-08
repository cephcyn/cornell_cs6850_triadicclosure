import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def graph(num_nodes, graph_type, seed=None, params=None):
    """
    Generates a random graph, based on given parameters
    num_nodes  : integer number of nodes in the graph
    graph_type : graph generation process
      'Gnp'        : G_np random distribution model
      'stochastic' : stochastic block model
    seed       : random seed to feed to graph generator
    params     : dict of parameters unique to graph_type
      params if graph_type is 'Gnp'
        p: edge probability
      params if graph_type is 'stochastic'
        within_p  : edge probability within group
        between_p : edge probability between groups
    """
    
    if graph_type == "Gnp":
        if params is None:
            params = {
                'p': 0.1,
            }
        G = nx.generators.random_graphs.gnp_random_graph(
            num_nodes, params['p'], seed=seed)
    elif graph_type == "stochastic":
        if params is None: 
            params = {
                'within_p': 0.5, #0.1,
                'between_p': 0.01 #0.005,
            }
        G = nx.generators.community.stochastic_block_model(
            [num_nodes//2, num_nodes//2], 
            [[params['within_p'], params['between_p']],
             [params['between_p'], params['within_p']]], 
            seed=seed)
    else:
        raise NotImplemented()
    return G


def get_pair_mutuals(G):
    """
    Get a list of tuple of tuple of two nodes that are not connected
    and the number of mutual neighbors between them.
    Does NOT include pairs with 0 mutual neighbors.
    Sorted in descending order of mutual neighbor count.
    Each element of the list is ((node1, node2), count)
    """

    num_nodes = len(G)
    pair_mutuals = []
    # check each pair of nodes
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            # ignore if they are already connected
            if j in list(G[i]): continue
            if i in list(G[j]): continue 
            # count number of mutual neighbors
            mutual_i_j = len(set(list(G[i])).intersection(set(list(G[j]))))
            if mutual_i_j>0:
                pair_mutuals.append(((i,j), mutual_i_j))
    # put them in descending order of mutual connection count
    pair_mutuals = sorted(pair_mutuals, key=lambda x: x[1], reverse=True)
    return pair_mutuals


def closure_round(G, num_edges, method='weighted_random'):
    """
    Does a single round of closure - adds edges to the graph
    G         : graph to modify
    num_edges : Upper limit on number of edges to add
    method    : edge selection method
      'weighted_random'         : 
      'squared_weighted_random' : 
      'max'                     : Adds top N edges sorted by mutual count
      'random'                  : Adds N random edges
      'static'                  : Don't do anything
    """

    pair_mutuals = get_pair_mutuals(G)
    # If requested to add more than is possible, cut it down
    if num_edges>len(pair_mutuals):
        num_edges = len(pair_mutuals)
    added_connections = []
    
    if method == 'weighted_random':
        # Probabilistically selects from pairs of nodes with shared neighbors
        # Linear weighting by mutual neighbor count
        def get_random_pair_to_close(pair_mutuals):
            num_mutuals = np.asarray([y for _, y in pair_mutuals])
            perc_mutuals = num_mutuals / np.sum(num_mutuals)
            indices = np.asarray(np.random.choice(
                np.arange(len(pair_mutuals)), num_edges,  p=perc_mutuals, replace=False))
            return [pair_mutuals[x] for x in indices]
        
        for (a, b), _ in get_random_pair_to_close(pair_mutuals):
            G.add_edge(a, b)
            added_connections.append((a,b))
            
    elif method == 'squared_weighted_random':
        # Probabilistically selects from pairs of nodes with shared neighbors
        # Exponential weighting by mutual neighbor count ^^2
        def get_random_pair_to_close(pair_mutuals):
            num_mutuals = np.asarray([y for _, y in pair_mutuals])
            perc_mutuals = num_mutuals**2
            perc_mutuals = perc_mutuals / np.sum(perc_mutuals)
            indices = np.asarray(np.random.choice(
                np.arange(len(pair_mutuals)), num_edges,  p=perc_mutuals, replace=False))
            return [pair_mutuals[x] for x in indices]
        
        for (a, b), _ in get_random_pair_to_close(pair_mutuals):
            G.add_edge(a, b)
            added_connections.append((a,b))
            
    elif method == 'max':
        # Deterministically selects from pairs of nodes with shared neighbors
        # Selects pairs with highest number of shared neighbors
        # If there is more than one with equal count, pick earliest one (WLOG)
        for (a, b), _ in pair_mutuals[:num_edges]:
            G.add_edge(a, b)
            added_connections.append((a, b))

    elif method == 'random':
        # Uniformly randomly selects from unconnected pairs of nodes
        while len(added_connections)<num_edges:
            a, b = np.random.choice(np.arange(len(G)), 2, replace=False)
            if (not G.has_edge(a, b)) and a!=b:
                G.add_edge(a, b)
                added_connections.append((a, b))
        
    elif method == 'static':
        # Don't add anything
        pass
        
    else:
        raise NotImplementedError()
    return added_connections

def get_seedset(node_ids, num_seeds):
    """
    Given nodes to select from and # of cascade start seeds, return IDs of nodes
    that are seeds
    """
    return np.random.choice(node_ids, num_seeds, replace=False)

def get_thresholds(num_nodes, mean=0.4, var=0.25):
    """
    Given total number of nodes, generate node percentage-thresholds using a
    normal distribution
    """
    thresholds = np.random.normal(mean, var, num_nodes)
    # Scale to between 0 and 1
    thresholds = np.clip(thresholds, 0, 1.1)
    return thresholds

def cascade_population(G, thresholds, 
                       seed_set=None, num_seeds=10, max_rounds=1000):
    """
    Do a population cascade simulation
    """
    num_nodes = len(G)
    # Initalize active nodes
    if seed_set is None:
        seed_set = get_seedset(num_nodes, num_seeds)
    active_nodes = np.zeros((num_nodes))
    active_nodes[seed_set] = 1

    # Iterate over rounds
    prev_total_active = np.sum(active_nodes)
    for r in range(max_rounds):
        total_active = np.count_nonzero(active_nodes)
        active_visible = total_active / num_nodes
        active_nodes[np.where(thresholds <= active_visible)[0]] += 1 

        total_active = np.count_nonzero(active_nodes)
        if prev_total_active == total_active:
            break
        prev_total_active = total_active
    return total_active, r, active_nodes