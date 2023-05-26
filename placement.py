# %%

from utils.parsing import load_netlist
adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist")

from environment.environment import CircuitEnv
env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,0])

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from model.agent import CircuitExtractor, CircuitActorCriticPolicy
policy_kwargs = dict(features_extractor_class=CircuitExtractor)
model = MaskablePPO(CircuitActorCriticPolicy, env, policy_kwargs=policy_kwargs,verbose=1)
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


# %%
# After training

model.learn(1)

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
