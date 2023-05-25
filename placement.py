# %%

from utils.parsing import load_netlist
adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist")

from environment.environment import CircuitEnv
env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,0])

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from model.agent import CircuitExtractor, CircuitActorCriticPolicy
policy_kwargs = dict(features_extractor_class=CircuitExtractor)
model = MaskablePPO(CircuitActorCriticPolicy, env, policy_kwargs=policy_kwargs,verbose=0)
model.policy

done = False

# %%
# Before training

obs, _ = env.reset()
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)
env.render()


# %%
# After training

model.learn(100)

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
