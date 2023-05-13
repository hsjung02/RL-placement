import pickle
def load_netlist(path="./netlist"):
    with open(path+"/adjacency_matrix", "rb") as f:
        adjacency_matrix = pickle.load(f)

    with open(path+"/cells", "rb") as f:
        cells = pickle.load(f)

    with open(path+"/macro_indices", "rb") as f:
        macro_indices = pickle.load(f)

    with open(path+"/std_indices", "rb") as f:
        std_indices = pickle.load(f)
    
    return adjacency_matrix, cells, macro_indices, std_indices