# %%

import torch as th
from utils.parsing import load_netlist
from environment.environment import CircuitEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from model.agent import CircuitExtractor, CircuitActorCriticPolicy
from utils.callback import ProgressCallback, VideoRecorderCallback
from stable_baselines3.common.monitor import Monitor
from time import time

dtype = "torch.cuda.FloatTensor" if th.cuda.is_available() else "torch.FloatTensor"
th.set_default_tensor_type(dtype)

lamb = 0

adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist")
env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,lamb])
print("Made environments")

n_steps = 128
batch_size = 32
total_timesteps = 15000

policy_kwargs = dict(features_extractor_class=CircuitExtractor)
model = MaskablePPO(policy=CircuitActorCriticPolicy,
                    env=env,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    verbose=0,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log="./placement_tensorboard/")
print("Model setting done")
# %%
# Before training
start_time = time()
obs, _ = env.reset()
done = False
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)
print("done in %fs"%(time()-start_time))
env.render()

print("Start training")
# %%
# After training

callback = ProgressCallback(check_freq=1500)

model.learn(total_timesteps=total_timesteps, callback=callback)

obs, _ = env.reset()
done = False
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)
env.render()
# %%
model.save("lambda_%d"%(lamb))