from stable_baselines3 import PPO, SAC
from sb3_contrib import TQC

from custom_gym.envs.panda_tasks import panda_reach, panda_pick_and_place\
        , panda_push, panda_flip, panda_slide,panda_stack

env_id = 'RePandaReachEnv'
algorithm_name = 'TQC'
reward_type = 'dense'
env = panda_reach.RePandaReachEnv(render=True, reward_type=reward_type,control_type="ee")

command = algorithm_name + ".load('./trained/' + env_id + '/' + str(algorithm_name)+'_'+str(reward_type)+'100000'.zip, env=env)"

model = eval(command)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(info)
    env.render()
    if done:
        print('Done')
        obs = env.reset()
