import torch as th
import numpy as np
from utils.parsing import load_netlist
from environment.environment import CircuitEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from model.agent import CircuitExtractor, CircuitActorCriticPolicy
from utils.routing import routing
import argparse



if __name__ == "__main__":
    dtype = "torch.cuda.FloatTensor" if th.cuda.is_available() else "torch.FloatTensor"
    th.set_default_tensor_type(dtype)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--lamb', type=str, help='model name')
    args = parser.parse_args()
    adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist")
    env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,int(args.lamb)])
    model = MaskablePPO.load(args.model)
    obs, _ = env.reset()
    done = False
    while not done:
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks)
        obs, reward, done, truncated, info = env.step(action)
    env.render(mode="save", path="before.png")
    cell_positions = np.asarray(env.cell_position, dtype=int)
    adj_i = env.static_features["adj_i"]
    adj_j = env.static_features["adj_j"]
    wirelength = routing(cell_positions, adj_i, adj_j)
    print("HPWL%f, CONGESTION %f, REAL WIRELENGTH %f for model %s"%(env.get_wirelength(), env.get_congestion(), wirelength, args.model))

