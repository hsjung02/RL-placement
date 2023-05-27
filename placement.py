# %%

import torch as th
if th.cuda.is_available():
    th.set_default_tensor_type('torch.cuda.FloatTensor')

from utils.parsing import load_netlist
adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist")

from environment.environment import CircuitEnv
env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,0])

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from model.agent import CircuitExtractor, CircuitActorCriticPolicy
policy_kwargs = dict(features_extractor_class=CircuitExtractor)
model = MaskablePPO(CircuitActorCriticPolicy, env, n_steps=100, policy_kwargs=policy_kwargs,verbose=1)
model.policy

done = False

# %%
# Before training
from time import time
start_time = time()
obs, _ = env.reset()
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)
    print("step done in", time()-start_time)
    start_time = time()
env.render()


from stable_baselines3.common.callbacks import BaseCallback

class ProgressCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"Total timesteps: {self.num_timesteps}")
        return True

callback = ProgressCallback(check_freq=100)

# %%
# After training

model.learn(total_timesteps=10, callback=callback)
obs, _ = env.reset()
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)
env.render()

# %%
obs, _ = env.reset()
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)
env.render()
# %%
model.save("placement.pt")

# %%
