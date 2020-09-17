import enum

from spinup.utils.test_policy import load_policy_and_env
from rrc_simulation.gym_wrapper.envs import cube_env, custom_env
from rrc_simulation.gym_wrapper.envs.cube_env import ActionType
from rrc_simulation.traj_opt import 
from rrc_simulation.control.custom_pinocchio_utils import CustomPinocchioUtils
from rrc_simulation.control.controller_utils import *


class PolicyMode(enum.Enum):
    TRAJ_OPT = enum.auto()
    IMPEDENCE = enum.auto()
    RL_ONLY = enum.auto()
    RL_RESIDUAL = enum.auto()


class ResidualPolicy:
    impedence_keys = []
    def __init__(self, 
                 observation_space,
                 action_space,
                 policy_mode=PolicyMode.RL_RESIDUAL):
        self.observation_space = observation_space
        self.action_space = action_space
        self.mode = policy_mode
        if self.mode == PolicyMode.RL_RESIDUAL:
            
    def get_action(self, observation):
        if self.mode == PolicyMode.RL_RESIDUAL:


class ResidualPolicyEnv(cube_env.CubeEnv):
    def __init__(self,
        initializer,
        action_type=ActionType.TORQUE,
        frameskip=1,
        visualization=False,
        policy_mode=PolicyMode.IMPEDANCE,
        policy=None,
        load_dir='',
        load_itr='last'
    ):
        super(ResidualPolicyEnv, self).__init__(initializer, action_type,
                frameskip, visualization)
        self.mode = policy_mode
        self.rl_policy = policy
        if self.rl_policy is None:
            self.load_policy(load_dir, load_itr)

    def load_policy(self, load_dir, load_itr='last', deterministic=False):
        self.rl_policy, _ = load_policy_and_env(load_dir, load_itr, deterministic)
        print(f'loaded policy from {load_dir}')
        return

    def get_action(self, observation):
        if self.mode == PolicyMode.TRAJ_OPT:
            return self.rl_policy(observation)
        elif self.mode == PolicyMode.RL_ONLY:

    def process_observation_residual(self, observation):
        return observation

    def process_observation_rl(self, observation):
        return observation
    
    def process_observation_impedence(self, observation):
        return observation

    def reset(self):
        self.last_obs = super(ResidualPolicyEnv, self).reset()
        return self.last_obs

    def step(self, action):
        if self.mode == PolicyMode.RL_RESIDUAL:
            action = self.rl_policy(np.concatenate(
                [self.process_observation_rl(self.last_obs), action])) 
        elif self.mode == PolicyMode.RL_ONLY:
            action = self.rl_policy(self.last_obs)
        # CubeEnv handles gym_action_to_robot_action
        return super(ResidualPolicyEnv, self).step(action)

