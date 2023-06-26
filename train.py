from stable_baselines3 import SAC, HerReplayBuffer, PPO, DDPG
from sb3_contrib import TQC
import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_gym.envs.panda_tasks import panda_reach, panda_pick_and_place\
        , panda_push, panda_flip, panda_slide,panda_stack


reward_type = "dense"
env = panda_reach.RePandaReachEnv(render=False, reward_type= reward_type, control_type= "ee")
env_id = env.__class__.__name__

#print(env.reset())
#print(env.time)

log_dir = './tensorboard/' + env_id
total_timesteps = 100000

checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path='model_checkpoints/'+env_id,
                                         name_prefix=env_id)


model = PPO(policy="MultiInputPolicy", env=env, learning_rate=1e-3, batch_size=2048,
        policy_kwargs=dict(net_arch=[256,256]), gamma=0.95,
            verbose=1,
            tensorboard_log=log_dir)

# model = TQC(policy="MultiInputPolicy", env=env, learning_rate=1e-3, buffer_size=25_000, batch_size=512,
#              policy_kwargs=dict(net_arch=[256, 256], n_critics=2), gamma=0.95, tau=0.05,
#             verbose=1, tensorboard_log=log_dir)

# model = DDPG(policy="MultiInputPolicy", env=env, learning_rate=1e-3, buffer_size=25_000, batch_size=512,
#                 policy_kwargs=dict(net_arch=[256, 256], n_critics=2), gamma=0.95, tau=0.05,
#              verbose=1, tensorboard_log=log_dir)

# model = SAC(policy="MultiInputPolicy", env=env, learning_rate=1e-3, buffer_size=25_000, batch_size=512,
#                 policy_kwargs=dict(net_arch=[256, 256], n_critics=2), gamma=0.95, tau=0.05,
#              verbose=1, tensorboard_log=log_dir)



model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

model.save('./trained/'+env_id+'/'+str(model.__class__.__name__)+reward_type+'_'+str(total_timesteps))
