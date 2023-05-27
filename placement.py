# %%

import torch as th
from utils.parsing import load_netlist
from environment.environment import CircuitEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from model.agent import CircuitExtractor, CircuitActorCriticPolicy
from utils.callback import ProgressCallback, VideoRecorderCallback
from stable_baselines3.common.monitor import Monitor

if th.cuda.is_available():
    th.set_default_tensor_type("th.cuda.FloatTensor")
# dtype = "th.cuda.FloatTensor" if th.cuda.is_available() else "th.FloatTensor"
# th.set_default_tensor_type(dtype)


adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist")
env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,0])
log_env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,0])
log_env = Monitor(log_env)

policy_kwargs = dict(features_extractor_class=CircuitExtractor)
model = MaskablePPO(CircuitActorCriticPolicy, env, n_steps=1, batch_size=1, policy_kwargs=policy_kwargs,verbose=1, tensorboard_log="./placement_tensorboard/")
model.policy

# %%
# Before training

obs, _ = env.reset()
done = False
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)
env.render()


# %%
# After training

callback = ProgressCallback(check_freq=100)
# callback = VideoRecorderCallback(log_env, render_freq=1)

model.learn(total_timesteps=1000, callback=callback)
# model.learn(total_timesteps=1000)

obs, _ = env.reset()
done = False
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)
env.render()
# %%
model.save("placement.pt")
