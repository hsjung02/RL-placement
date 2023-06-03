import torch as th
from utils.parsing import load_netlist
from environment.environment import CircuitEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from model.agent import CircuitExtractor, CircuitActorCriticPolicy
from utils.callback import ProgressCallback, VideoRecorderCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor


dtype = "torch.cuda.FloatTensor" if th.cuda.is_available() else "torch.FloatTensor"
th.set_default_tensor_type(dtype)

lamb = 30

adjacency_matrix, cells, macro_indices, std_indices, pin_indices = load_netlist("./netlist/ispd18test8/")
env = CircuitEnv(adjacency_matrix, cells, macro_indices, std_indices, pin_indices, reward_weights=[1,lamb])
print("Made environments")

eval_env = Monitor(env)
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-6000000, verbose=1)
eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)
#eval_callback = ProgressCallback(check_freq=1500)


n_steps = 128
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
print("Model setting done")

print("Start training")

total_cnt = 0
while True:
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    total_cnt += total_timesteps
    print("Total timesteps: %d"%(total_cnt))
    model.save("timesteps_%d_test8_lambda_%d" % (total_cnt, lamb))