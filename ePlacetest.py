
import torch as th
import numpy as np
from utils.parsing import load_netlist
from environment.environment import CircuitEnv


adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist")
env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,0])

env.place_macro(1)

#env.render()

env.place_std()

env.render()
