# %%
from utils.parsing import load_netlist
adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist")

from environment.environment import CircuitEnv
env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,0,10])

p = [4,15,26,100,111,122,196,207,218,292,303,314,388,399,410,484]
n = 16
for i in range(n):
    env.place_macro(p[i])
#env.render()
# %%
env.place_std()
# %%
env.render()