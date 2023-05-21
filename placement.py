# %%

adjacency_matrix = [[0,1,1,0],[1,0,0,1],[1,0,0,0],[0,1,0,0]]
cells = {0:{'width':8,'height':8}, 1:{'width':6,'height':6},2:{'width':4,'height':4},3:{'width':2,'height':2}}
macro_indices = [0,1,2,3]
std_indices = []
pin_indices = []

from environment.environment import CircuitEnv
env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,0,10])

# from stable_baselines3.common.env_checker import check_env
# check_env(env)

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from model.agent import CircuitExtractor, CircuitActorCriticPolicy
policy_kwargs = dict(features_extractor_class=CircuitExtractor)
model = MaskablePPO(CircuitActorCriticPolicy, env, policy_kwargs=policy_kwargs,verbose=0)
model.policy

done = False

obs = env.reset()[0]
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    print(action)
    obs, reward, done, truncated, info = env.step(action)

env.render()

# %%
env.place_std()
env.render()
# %%

