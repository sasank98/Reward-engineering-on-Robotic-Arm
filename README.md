# Reward engineering on Robotic Arm

Implemented reinforcement learning algorithms on a Robotic arm in panda-gym environment
Changed the reward functions from the standard panda-gym environment to improve the stability, changes were made in the custom_gym directory
`custom_gym` is cloned from panda-gym and changes in the reward function were made in the `panda_tasks` directory so the model can be directly trained without the `panda-gym library`

The models were trained for 100,000 iterations to show the effect of Reward engineering clearly.
The negative distance was multiplied by a metric of time to improve the stability in later stages.

_Reward = -distance*Number of Steps_

The above reward function would work only for dense rewards as sparse rewards give +1 at the end of a successful episode.

## requirements

stable-baselines3 == 1.5.1a5

sb3-contrib == 1.5.1a5

## Training and testing agents
Files can be found under the "trained" folder.

To test the trained agents, run `tensorboard.py`.

To test the trained agents, run `show.py`.

To re-train an agent, run `train.py`.

Code to run SAC, TQC, and DDPG algorithms was written as well and consistency was maintained across them

Only the agents trained with the new reward function are available in the `trained` directory

`

    def reset(self) -> None:
        with self.sim.no_rendering():
            self.time = 0
            self.robot.reset()
            self.task.reset()
        return self._get_obs()

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        self.time += 1
        d = self.task.compute_reward(achieved_goal, desired_goal, info)

        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return d*self.time
            
`

make changes to these methods in the, specific tasks to make a custom reward function
the tasks are present in `custom_gym\envs\panda_tasks` directory

## Graphs
Before testing the effectiveness of reward engineering, we iterated through different algorithms by varying their hyper-parameters on the Panda-reach environment

Mean rewards across different algorithms are plotted in the graph below

![image](/support/panda_reach_algorithms.png)

Mean Rewards for both cases on the DDPG algorithm

![image](/support/DDPG_rengg.png)
it can be seen that the mean reward is about the same for both cases, but we actually changed the reward function and gave an additional multiplication factor

Mean Rewards for both cases using TQC algorithm

![image](/support/TQC_rengg.png)

## Video for viewing the difference in stability

The video for the case without changing the reward function is given below

![image](/support/no_change_1.gif)

The video for the case with reward engineering is given below

![image](/support/reward_changed_2.gif)
