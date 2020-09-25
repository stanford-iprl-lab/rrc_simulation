import numpy as np
from gym.spaces import Dict
from gym import ObservationWrapper
from rrc_simulation.tasks import move_cube
from rrc_simulation.trifinger_platform import TriFingerPlatform
from rrc_simulation.gym_wrapper.envs import cube_env, custom_env
from rrc_simulation.gym_wrapper.envs.cube_env import ActionType
from rrc_simulation.control.custom_pinocchio_utils import CustomPinocchioUtils
from rrc_simulation.control.controller_utils import PolicyMode
from rrc_simulation.control import controller_utils as c_utils
from rrc_simulation.control.control_policy import HierarchicalControllerPolicy


class ResidualPolicyWrapper(ObservationWrapper):
    def __init__(self, env, policy):
        assert isinstance(env.unwrapped, cube_env.CubeEnv), 'env expects type CubeEnv'
        self.env = env
        self.reward_range = self.env.reward_range
        # set observation_space and action_space below
        spaces = TriFingerPlatform.spaces
        self._action_space = Dict({
            'torque': spaces.robot_torque.gym, 'position': spaces.robot_position.gym})
        self.set_policy(policy)

    @property
    def impedance_control_mode(self):
        return (self.mode == PolicyMode.IMPEDANCE or
                (self.mode == PolicyMode.RL_PUSH and
                 self.rl_observation_space is None))

    @property
    def action_space(self):
        if self.impedance_control_mode:
            return self._action_space['torque']
        else:
            return self._action_space['position']

    @property
    def action_type(self):
        if self.impedance_control_mode:
            return ActionType.TORQUE
        else:
            return ActionType.POSITION

    @property
    def mode(self):
        assert self.policy, 'Need to first call self.set_policy() to access mode'
        return self.policy.mode

    @property
    def frameskip(self):
        if self.mode == PolicyMode.RL_PUSH:
            return self.policy.rl_frameskip
        return 1

    def set_policy(self, policy):
        self.policy = policy
        if policy:
            self.rl_observation_names = policy.observation_names
            self.rl_observation_space = policy.rl_observation_space
            obs_dict = {'impedance': self.env.observation_space}
            if self.rl_observation_space:
                obs_dict['rl'] = self.rl_observation_space
            self.observation_space = Dict(obs_dict)

    def observation(self, observation):
        observation_imp = self.process_observation_impedance(observation)
        obs_dict = {'impedance': observation_imp}
        if 'rl' in self.observation_space.spaces:
            observation_rl = self.process_observation_rl(observation)
            obs_dict['rl'] = observation_rl
        return obs_dict

    def process_observation_residual(self, observation):
        return observation

    def process_observation_rl(self, observation):
        if len(self.platform._action_log['actions']):
            t = self.platform._action_log['actions'][-1]['t']
        else:
            t = 0
        robot_observation = self.platform.get_robot_observation(t)
        object_observation = self.platform.get_object_pose(t)
        robot_tip_positions = self.platform.forward_kinematics(
            robot_observation.position
        )
        robot_tip_positions = np.array(robot_tip_positions)

        observation = {
            "robot_position": robot_observation.position,
            "robot_velocity": robot_observation.velocity,
            "robot_tip_positions": robot_tip_positions,
            "object_position": object_observation.position,
            "object_orientation": object_observation.orientation,
            "goal_object_position": self.goal["position"],
            "goal_object_orientation": self.goal["orientation"],
        }
        observation = np.concatenate([observation[k].flatten() for k in self.rl_observation_names])
        return observation

    def process_observation_impedance(self, observation):
        return observation

    def reset(self):
        obs = super(ResidualPolicyWrapper, self).reset()
        self.policy.platform = self.env.unwrapped.platform
        if isinstance(self.policy, HierarchicalControllerPolicy):
            self.policy.mode = self.policy.start_mode
            self.policy.traj_initialized = False
        self.step_count = 0
        return obs

    def _step(self, action):
        if self.unwrapped.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.frameskip

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > move_cube.episode_length:
            excess = step_count_after - move_cube.episode_length
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > move_cube.episode_length:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.unwrapped.platform.append_desired_action(robot_action)

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            observation = self.unwrapped._create_observation(t + 1)

            reward += self.unwrapped.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.unwrapped.info,
            )

        is_done = self.step_count == move_cube.episode_length

        return observation, reward, is_done, self.env.info

    def _gym_action_to_robot_action(self, gym_action):
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def step(self, action):
        # CubeEnv handles gym_action_to_robot_action
        #print(self.mode)
        if self.mode == PolicyMode.RL_PUSH:
            self.unwrapped.frameskip = self.policy.rl_frameskip
        else:
            self.unwrapped.frameskip = 1

        obs, r, d, i = self._step(action)
        obs = self.observation(obs)
        return obs, r, d, i

