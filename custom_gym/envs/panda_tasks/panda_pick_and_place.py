import numpy as np

from typing import Any, Dict, Union
import sys
sys.path.append("..")

from custom_gym.envs.core import RobotTaskEnv
from custom_gym.envs.robots.panda import Panda
from custom_gym.envs.tasks.pick_and_place import PickAndPlace
from custom_gym.pybullet import PyBullet


class RePandaPickAndPlaceEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlace(sim, reward_type=reward_type)
        self.time = 0
        super().__init__(robot, task)

    def reset(self) -> None:
        with self.sim.no_rendering():
            self.time = 0
            self.robot.reset()
            self.task.reset()
        return self._get_obs()

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> Union[np.ndarray, float]:
        self.time += 1
        d = self.task.compute_reward(achieved_goal, desired_goal, info)
        '''
        d = distance(achieved_goal, desired_goal)
        '''
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float64)
        else:
            return d*self.time