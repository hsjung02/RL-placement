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


adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist")
env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,0])
log_env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,0])
log_env = Monitor(log_env)
print("made environments")

n_steps = 40
batch_size = 20
total_timesteps = 3000

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

obs, _ = env.reset()
done = False
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)
env.render()

print("Start training")
start_time = time()
# %%
# After training

callback = ProgressCallback(check_freq=100)
# callback = VideoRecorderCallback(log_env, render_freq=1)

model.learn(total_timesteps=total_timesteps, callback=callback)
# model.learn(total_timesteps=1000)

print("Finish training in %ds"%(time()-start_time))
obs, _ = env.reset()
done = False
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)
env.render()
# %%
model.save("placement_nsteps%d_batch%d"%(n_stseps, batch_size))