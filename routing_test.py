import torch as th
import numpy as np
from utils.parsing import load_netlist
from environment.environment import CircuitEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from model.agent import CircuitExtractor, CircuitActorCriticPolicy
from utils.callback import ProgressCallback, VideoRecorderCallback
from stable_baselines3.common.monitor import Monitor
from utils.routing import routing

adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist")
env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,0.001])

n_steps = 64
batch_size = 32
total_timesteps = 6000

policy_kwargs = dict(features_extractor_class=CircuitExtractor)
model = MaskablePPO(policy=CircuitActorCriticPolicy,
                    env=env,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    verbose=0,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log="./placement_tensorboard/")

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
print("HPWL%f, CONGESTION %f, REAL WIRELENGTH %f"%(env.get_wirelength(), env.get_congestion(), wirelength))

# model = MaskablePPO.load("placement_nsteps40_batch20")
model.learn(total_timesteps=total_timesteps, callback=ProgressCallback(check_freq=500))
obs, _ = env.reset()
done = False
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)
env.render(mode="save", path="after.png")
cell_positions = np.asarray(env.cell_position, dtype=int)
adj_i = env.static_features["adj_i"]
adj_j = env.static_features["adj_j"]
wirelength = routing(cell_positions, adj_i, adj_j)
print("HPWL%f, CONGESTION %f, REAL WIRELENGTH %f"%(env.get_wirelength(), env.get_congestion(), wirelength))