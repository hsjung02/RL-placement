import torch as th
from utils.parsing import load_netlist
from environment.environment import CircuitEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from model.agent import CircuitExtractor, CircuitActorCriticPolicy
from utils.callback import ProgressCallback, VideoRecorderCallback
from stable_baselines3.common.monitor import Monitor
from time import time
import argparse
import sys

dtype = "torch.cuda.FloatTensor" if th.cuda.is_available() else "torch.FloatTensor"
th.set_default_tensor_type(dtype)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model name')
parser.add_argument('--benchmark', type=str, help='model name')
args = parser.parse_args()


if(args.benchmark=='ispd18test8'):
    adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist/ispd18test8")
elif(args.benchmark=='ispd18test3'):
    adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist/ispd18test3")
else:
    sys.exit("invalid benchmark")


#before training====================================================================================================    
env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,0])

n_steps = 128
batch_size = 32
total_timesteps = 3000

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
env.render(mode="save", path="./src/%s_before.png" % args.model)

del model
#===================================================================================================================

#after training=====================================================================================================   
model = MaskablePPO.load(args.model)

obs, _ = env.reset()
done = False
while not done:
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, done, truncated, info = env.step(action)

#===================================================================================================================   
env.render(mode="save", path="./src/%s_after.png" % args.model)