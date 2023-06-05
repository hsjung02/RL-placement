# %%
from utils.parsing import load_netlist
from environment.environment import CircuitEnv
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, help='model name')
args = parser.parse_args()

if(args.benchmark=='ispd18test8'):
    adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist/ispd18test8")
    p = [4,15,26,100,111,122,196,207,218,292,303,314,388,399,410,484]
    n = 16
elif(args.benchmark=='ispd18test3'):
    adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist/ispd18test3")
    p = [4,111,292,410]
    n=4    
else:
    sys.exit("invalid benchmark")


env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,0])

for i in range(n):
    env.step(p[i])
#env.render()
# %%
env.place_std()
# %%
env.render()