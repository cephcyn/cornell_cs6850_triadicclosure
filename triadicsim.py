import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def graph(num_nodes, graph_type='Gnp', seed=0, params=None):
    if graph_type == "Gnp":
        if params is None:
            params = {
                'p': 0.1,
            }
        G = nx.generators.random_graphs.gnp_random_graph(
            num_nodes, params['p'])
    elif graph_type == "stochastic":
        if params is None: 
            params = {
                'within_p': 0.5, #0.1,
                'between_p': 0.01 #0.005,
            }
        G = nx.generators.community.stochastic_block_model(
            [num_nodes//2, num_nodes//2], 
            [[params['within_p'], params['between_p']],
             [params['between_p'], params['within_p']]], seed=seed)
    else:
        raise NotImplemented()
    return G


def get_num_pair_mutuals(G):
    """Get a list of tuple of tuple of two nodes that are not connected
    and the number of mutual connections between them. """

    num_nodes = len(G)
    num_pair_mutuals = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if j in list(G[i]): continue
            if i in list(G[j]): continue 
            mutual_i_j = len(set(list(G[i])).intersection(set(list(G[j]))))
            if mutual_i_j:
                num_pair_mutuals.append(((i,j), mutual_i_j))
    num_pair_mutuals = sorted(num_pair_mutuals, key=lambda x: x[1], reverse=True)
    return num_pair_mutuals


def closure_round(G, num_edges, method='weighted_random'):
    """Add graphs to the edges."""

    num_pair_mutuals = get_num_pair_mutuals(G)
    
    added_connections = []
    if method == 'weighted_random':
        def get_random_pair_to_close(num_pair_mutuals):
            num_mutuals = np.asarray([y for _, y in num_pair_mutuals])
            perc_mutuals = num_mutuals / np.sum(num_mutuals)

            indices = np.asarray(np.random.choice(
                np.arange(len(num_pair_mutuals)), num_edges,  p=perc_mutuals))

            return [num_pair_mutuals[x] for x in indices]
        
        for (a, b), _ in get_random_pair_to_close(num_pair_mutuals):
            G.add_edge(a, b)
            added_connections.append((a,b))
    elif method == 'squared_weighted_random':
        def get_random_pair_to_close(num_pair_mutuals):
            num_mutuals = np.asarray([y for _, y in num_pair_mutuals])
            perc_mutuals = num_mutuals / np.sum(num_mutuals)
            perc_mutuals = perc_mutuals**2
            perc_mutuals = perc_mutuals / np.sum(perc_mutuals)

            indices = np.asarray(np.random.choice(
                np.arange(len(num_pair_mutuals)), num_edges,  p=perc_mutuals))

            return [num_pair_mutuals[x] for x in indices]
        
        for (a, b), _ in get_random_pair_to_close(num_pair_mutuals):
            G.add_edge(a, b)
            added_connections.append((a,b))
            
    elif method == 'max':
        """Add all edges that have the max # of mutuals."""
        num_mutuals = get_num_mutuals(G)
        if len(num_mutuals) == 0:
            return G, []

        added_connections = []
        max_mutuals = num_mutuals[0][-1]
        for (a, b), connections in num_mutuals:
            if connections == max_mutuals:
                G.add_edge(a, b)
                added_connections.append((a, b))

    elif method == 'random':
        a, b = None, None
        while True:
            a, b = np.random.choice(np.arange(num_nodes), 2)
            if not G.has_edge(a, b):
                break
        G.add_edge(a, b)
        added_connections.append((a, b))
    elif method == 'static':
        1 == 1
        
    else:
        raise NotImpelementedError()
    return G, added_connections
