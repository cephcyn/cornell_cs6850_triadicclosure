import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def graph(num_nodes, graph_type, params=None, rng_seed=None):
    """
    Generates a random graph, based on given parameters
    num_nodes  : integer number of nodes in the graph
    graph_type : graph generation process
      'Gnp'        : G_np random distribution model
      'stochastic' : stochastic block model
    params     : dict of parameters unique to graph_type
      params if graph_type is 'Gnp'
        p: edge probability
      params if graph_type is 'stochastic'
        within_p  : edge probability within group
        between_p : edge probability between groups
    rng_seed   : random seed to use
    """
    
    if graph_type == "Gnp":
        if params is None:
            params = {
                'p': 0.1,
            }
        G = nx.generators.random_graphs.gnp_random_graph(
            num_nodes, params['p'], seed=rng_seed)
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
            seed=rng_seed)
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


def closure_round(G, num_edges, method='weighted_random', rng_seed=None):
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
    rng_seed  : random seed to use
    """
    
    # Initialize round RNG
    rng = np.random.default_rng(rng_seed)

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
            indices = np.asarray(rng.choice(
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
            indices = np.asarray(rng.choice(
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
            a, b = rng.choice(np.arange(len(G)), 2, replace=False)
            if (not G.has_edge(a, b)) and a!=b:
                G.add_edge(a, b)
                added_connections.append((a, b))
        
    elif method == 'static':
        # Don't add anything
        pass
        
    else:
        raise NotImplementedError()
    return added_connections

def create_seedset(node_ids, num_seeds, rng_seed=None):
    """
    Given nodes to select from and # of cascade start seeds, return IDs of nodes
    that are seeds
    node_ids  : set of node IDs to select from
    num_seeds : number of seeds to create
    rng_seed  : random seed to use
    """
    
    rng = np.random.default_rng(rng_seed)
    return rng.choice(node_ids, num_seeds, replace=False)

def create_thresholds(num_nodes, mean=0.4, var=0.25, rng_seed=None):
    """
    Given total number of nodes, generate node percentage-thresholds using a
    normal distribution
    num_nodes : number of nodes to create thresholds for
    mean      : normal distribution mean
    var       : normal distribution variance
    rng_seed  : random seed to use
    """
    
    rng = np.random.default_rng(rng_seed)
    thresholds = rng.normal(mean, var, num_nodes)
    # Scale to between 0 and 1
    thresholds = np.clip(thresholds, 0, 1.1)
    return thresholds

def simulate_cascade(G, thresholds, seed_set,
                     use_network=True, sticky_cascade=True,
                     closure_type='random', closure_rate=1,
                     max_rounds=20, rng_seed=None):
    """
    Do a cascade simulation and return the node infection log over time
    G              : the graph of nodes
    thresholds     : thresholds[i] equals the percentage threshold of node with ID i
    seedset        : the set of nodes within graph initially infected
    use_network    : True for simulating network model, False for population model
    sticky_cascade : True if nodes are never able to quit the cascade, False otherwise
    closure_type   : type of edge creation to perform.
      Only considered if use_network==True
      'max'                     : instantly maximize triad creation (T)
      'squared_weighted_random' : probabilistically draw, weighted by T^2
      'weighted_random'         : probabilistically draw, weighted by T
      'random'                  : uniformly randomly draw
      'static'                  : do NOT create any edges (naive static network)
    closure_rate   : rate of edge creation per cascade round.
      If there are X<closure_rate edges available to consider in a single move, caps at X
    max_rounds     : maximum number of cascade rounds to simulate
    rng_seed       : random seed to use
    """
    
    rng = np.random.default_rng(rng_seed)
    
    # Fork the graph to prevent making changes to the original input graph structure
    G = G.copy()
    num_nodes = len(G)
    # Initialize active nodes
    curr_active_nodes = np.zeros((num_nodes))
    curr_active_nodes[seed_set] = 1
    # Initialize graph change history logs
    G_log = [G]
    node_log = [set(np.nonzero(curr_active_nodes)[0])]
    edge_log = [set()]
    
    # Define helper functions
    def get_frac_active_neighbors(G, n, active_nodes):
        """
        Return fraction of active neighbors out of all neighbors for a specific graph node
        G            : graph to work with
        n            : ID of node to investigate
        active_nodes : active_nodes[i]==1 iff node with ID i is activated
        """
        neighbors = list(G[n])
        if len(neighbors) == 0:
            return 0
        num_active_neighbors = np.count_nonzero([active_nodes[neighbor] for neighbor in neighbors])
        return num_active_neighbors / len(neighbors)
    
    # Run rounds
    curr_round = 0
    keep_simulating = True
    while (curr_round < max_rounds) and keep_simulating:
        G = G.copy()
        # Initialize description of changes to make
        next_active_nodes = np.zeros((num_nodes))
        newly_added_edges = []
        # Run cascade spread on nodes
        if (not use_network):
            # Do population model calculation
            current_active_num = np.sum(curr_active_nodes)
            next_active_nodes[np.where(thresholds <= (current_active_num / num_nodes))[0]] = 1
        else:
            # Do network model calculation
            next_active_nodes[[
                n for n in G.nodes() 
                if (thresholds[n]<=get_frac_active_neighbors(G, n, curr_active_nodes))
            ]] = 1
        # Add layer of cascade stickiness iff it's activated
        if sticky_cascade:
            next_active_nodes[np.nonzero(curr_active_nodes)[0]] = 1
        # Run graph closure / edge creation
        if (use_network) and (closure_type!='static'):
            newly_added_edges = closure_round(
                G, closure_rate, method=closure_type, rng_seed=rng_seed
            )
        # Update the logs and current state
        G_log.append(G)
        node_log.append(set(np.nonzero(next_active_nodes)[0]))
        edge_log.append(set(newly_added_edges))
        # Calculate whether we should continue or not
        unchanged_edges = len(newly_added_edges)==0
        fully_infected = sticky_cascade and np.sum(curr_active_nodes)==len(G)
        fully_frozen = np.sum(np.abs(next_active_nodes-curr_active_nodes))==0 and unchanged_edges
        keep_simulating = not (fully_infected or fully_frozen)
        # Update meta-fields
        curr_active_nodes = next_active_nodes
        curr_round += 1
    
    # We ran out of rounds, or there haven't been ANY changes in the graph state
    return G_log, node_log, edge_log, (fully_infected, fully_frozen)