"""Custom Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import numpy as np
import gym
import pybullet

from gym import wrappers
from gym.spaces import Box
from gym.spaces import Discrete
from gym.spaces import MultiDiscrete
from gym.spaces import MultiBinary
from gym.spaces import Tuple
from gym.spaces import Dict
from gym.spaces import utils

from rrc_simulation.control import controller_utils as c_utils
from rrc_simulation.control.custom_pinocchio_utils import CustomPinocchioUtils

from rrc_simulation import TriFingerPlatform
from rrc_simulation import visual_objects
from rrc_simulation.tasks import move_cube
from rrc_simulation.gym_wrapper.envs.cube_env import CubeEnv, ActionType
from rrc_simulation.gym_wrapper.utils import configurable
from scipy.spatial.transform import Rotation


MAX_DIST = move_cube._max_cube_com_distance_to_center
DIST_THRESH = move_cube._CUBE_WIDTH / 5
ORI_THRESH = np.pi / 8
REW_BONUS = 1
POS_SCALE = np.array([0.128, 0.134, 0.203, 0.128, 0.134, 0.203, 0.128, 0.134,
                      0.203])


def reset_camera():
    camera_pos = (0.,0.2,-0.2)
    camera_dist = 1.0
    pitch = -45.
    yaw = 0.
    if pybullet.isConnected() != 0:
        pybullet.resetDebugVisualizerCamera(cameraDistance=camera_dist,
                                    cameraYaw=yaw,
                                    cameraPitch=pitch,
                                    cameraTargetPosition=camera_pos)


def random_xy(sample_radius_min=0., sample_radius_max=None):
    # sample uniform position in circle (https://stackoverflow.com/a/50746409)
    radius = np.random.uniform(sample_radius_min, sample_radius_max)
    theta = np.random.uniform(0, 2 * np.pi)
    # x,y-position of the cube
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


@configurable(pickleable=True)
class CurriculumInitializer:
    """Initializer that samples random initial states and goals."""

    def __init__(self, difficulty=1, initial_dist=move_cube._CUBE_WIDTH,
                 num_levels=4, num_episodes=5, fixed_goal=None):
        """Initialize.

        Args:
            initial_dist (float): Distance from center of arena
            num_levels (int): Number of steps to maximum radius
            num_episodes (int): Number of episodes to compute mean over
        """
        self.difficulty = difficulty
        self.num_levels = num_levels
        self._current_level = 0
        self.levels = np.linspace(initial_dist, MAX_DIST, num_levels)
        self.final_dist = np.array([np.inf for _ in range(num_episodes)])
        if difficulty == 4:
            self.final_ori = np.array([np.inf for _ in range(num_episodes)])
        self.fixed_goal = fixed_goal

    @property
    def current_level(self):
        return min(self.num_levels - 1, self._current_level)

    def random_xy(self, sample_radius_min=0., sample_radius=None):
        # sample uniform position in circle (https://stackoverflow.com/a/50746409)
        sample_radius_max = sample_radius or self.levels[self.current_level]
        return random_xy(sample_radius_min, sample_radius_max)

    def update_initializer(self, final_pose, goal_pose):
        assert np.all(goal_pose.position == self.goal_pose.position)
        self.final_dist = np.roll(self.final_dist, 1)
        final_dist = np.linalg.norm(goal_pose.position - final_pose.position)
        self.final_dist[0] = final_dist
        if self.difficulty == 4:
            self.final_ori = np.roll(self.final_ori, 1)
            self.final_ori[0] = compute_orientation_error(goal_pose, final_pose,
                                                          scale=False)

        update_level = np.mean(self.final_dist) < DIST_THRESH
        if self.difficulty == 4:
            update_level = update_level and np.mean(self.final_ori) < ORI_THRESH

        if update_level and self._current_level < self.num_levels - 1:
            pre_sample_dist = self.goal_sample_radius
            self._current_level += 1
            post_sample_dist = self.goal_sample_radius
            print("Old sampling distances: {}/New sampling distances: {}".format(
                pre_sample_dist, post_sample_dist))

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        x, y = self.random_xy()
        self.initial_pose = move_cube.sample_goal(difficulty=-1)
        z = self.initial_pose.position[-1]
        self.initial_pose.position = np.array((x, y, z))
        return self.initial_pose

    @property
    def goal_sample_radius(self):
        if self.fixed_goal:
            goal_dist = np.linalg.norm(self.fixed_goal.position)
            return (goal_dist, goal_dist)
        if self._current_level == self.num_levels - 1:
            sample_radius_min = 0.
        else:
            sample_radius_min = self.levels[self.current_level]
        sample_radius_max = self.levels[min(self.num_levels - 1, self._current_level + 1)]
        return (sample_radius_min, sample_radius_max)

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        if self.fixed_goal:
            self.goal_pose = self.fixed_goal
            return self.fixed_goal
        # goal_sample_radius is further than past distances
        sample_radius_min, sample_radius_max = self.goal_sample_radius
        x, y = self.random_xy(sample_radius_min, sample_radius_max)
        self.goal_pose = move_cube.sample_goal(difficulty=self.difficulty)
        self.goal_pose.position = np.array((x, y, self.goal_pose.position[-1]))
        return self.goal_pose


@configurable(pickleable=True)
class ReorientInitializer:
    """Initializer that samples random initial states and goals."""
    goal_pose = move_cube.Pose(np.array([0,0,move_cube._CUBE_WIDTH/2]), np.array([0,0,0,1]))

    def __init__(self, difficulty=1, initial_dist=move_cube._CUBE_WIDTH):
        self.difficulty = difficulty
        self.initial_dist = initial_dist
        self.random = np.random.RandomState()

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        x, y = random_xy(self.initial_dist, MAX_DIST)
        self.initial_pose = move_cube.sample_goal(difficulty=-1)
        z = self.initial_pose.position[-1]
        self.initial_pose.position = np.array((x, y, z))
        if self.difficulty == 4:
            self.initial_pose.orientation = Rotation.random(random_state=self.random).as_quat()
        return self.initial_pose

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        return self.goal_pose


class RandomOrientationInitializer:
    goal = move_cube.Pose(np.array([0,0,move_cube._CUBE_WIDTH/2]), np.array([0,0,0,1]))

    def __init__(self, difficulty=4):
        self.difficulty = difficulty

    def get_initial_state(self):
        return move_cube.sample_goal(-1)

    def get_goal(self):
        return self.goal


@configurable(pickleable=True)
class PushCubeEnv(gym.Env):
    observation_names = [
            "robot_position",
            "robot_velocity",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position",
        ]

    def __init__(
        self,
        initializer=None,
        action_type=ActionType.POSITION,
        frameskip=1,
        visualization=False,
        ):
        """Initialize.

        Args:
            initializer: Initializer class for providing initial cube pose and
                goal pose. If no initializer is provided, we will initialize in a way
                which is be helpful for learning.
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            frameskip (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
        """
        # Basic initialization
        # ====================
        self.initializer = initializer
        self.action_type = action_type
        self.visualization = visualization

        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip

        # will be initialized in reset()
        self.platform = None

        # Create the action and observation spaces
        # ========================================

        spaces = TriFingerPlatform.spaces

        if self.action_type == ActionType.TORQUE:
            self.action_space = spaces.robot_torque.gym
        elif self.action_type == ActionType.POSITION:
            self.action_space = spaces.robot_position.gym
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": spaces.robot_torque.gym,
                    "position": spaces.robot_position.gym,
                }
            )
        else:
            raise ValueError("Invalid action_type")

        self.observation_space = gym.spaces.Dict(
            {
                "robot_position": spaces.robot_position.gym,
                "robot_velocity": spaces.robot_velocity.gym,
                "robot_tip_positions": gym.spaces.Box(
                    low=np.array([spaces.object_position.low] * 3),
                    high=np.array([spaces.object_position.high] * 3),
                ),
                "object_position": spaces.object_position.gym,
                "object_orientation": spaces.object_orientation.gym,
                "goal_object_position": spaces.object_position.gym,
            }
        )

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        move_cube.random = self.np_random
        return [seed]

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type
        if self.action_type == ActionType.TORQUE:
            robot_action = self.platform.Action(torque=gym_action)
        elif self.action_type == ActionType.POSITION:
            robot_action = self.platform.Action(position=gym_action)
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            robot_action = self.platform.Action(
                torque=gym_action["torque"], position=gym_action["position"]
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def reset(self):
        # reset simulation
        del self.platform

        # initialize simulation
        if self.initializer is None:
            # if no initializer is given (which will be the case during training),
            # we can initialize in any way desired. here, we initialize the cube always
            # in the center of the arena, instead of randomly, as this appears to help
            # training
            initial_robot_position = TriFingerPlatform.spaces.robot_position.default
            default_object_position = (
                TriFingerPlatform.spaces.object_position.default
            )
            default_object_orientation = (
                TriFingerPlatform.spaces.object_orientation.default
            )
            initial_object_pose = move_cube.Pose(
                position=default_object_position,
                orientation=default_object_orientation,
            )
            goal_object_pose = move_cube.sample_goal(difficulty=1)
        else:
            # if an initializer is given, i.e. during evaluation, we need to initialize
            # according to it, to make sure we remain coherent with the standard CubeEnv.
            # otherwise the trajectories produced during evaluation will be invalid.
            initial_robot_position = TriFingerPlatform.spaces.robot_position.default
            initial_object_pose=self.initializer.get_initial_state()
            goal_object_pose = self.initializer.get_goal()

        self.platform = TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            initial_object_pose=initial_object_pose,
        )

        self.goal = {
            "position": goal_object_pose.position,
            "orientation": goal_object_pose.orientation,
        }
        # visualize the goal
        if self.visualization:
            self.goal_marker = visual_objects.CubeMarker(
                width=0.065,
                position=goal_object_pose.position,
                orientation=goal_object_pose.orientation,
                physicsClientId=self.platform.simfinger._pybullet_client_id,
            )
            reset_camera()

        self.info = {"difficulty": self.initializer.difficulty}

        self.step_count = 0

        return self._create_observation(0)

    def _create_observation(self, t):
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
        }
        return observation

    @staticmethod
    def _compute_reward(previous_observation, observation):

        # calculate first reward term
        current_distance_from_block = np.linalg.norm(
            observation["robot_tip_positions"] - observation["object_position"]
        )
        previous_distance_from_block = np.linalg.norm(
            previous_observation["robot_tip_positions"]
            - previous_observation["object_position"]
        )

        reward_term_1 = (
            previous_distance_from_block - current_distance_from_block
        )

        # calculate second reward term
        current_dist_to_goal = np.linalg.norm(
            observation["goal_object_position"]
            - observation["object_position"]
        )
        previous_dist_to_goal = np.linalg.norm(
            previous_observation["goal_object_position"]
            - previous_observation["object_position"]
        )
        reward_term_2 = previous_dist_to_goal - current_dist_to_goal

        reward = 500 * reward_term_1 + 250 * reward_term_2
        return reward

    def step(self, action):
        if self.platform is None:
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
            t = self.platform.append_desired_action(robot_action)

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            previous_observation = self._create_observation(t)
            observation = self._create_observation(t + 1)

            reward += self._compute_reward(
                previous_observation=previous_observation,
                observation=observation,
            )

        is_done = self.step_count == move_cube.episode_length
        if is_done and isinstance(self.initializer, CurriculumInitializer):
            goal_pose = self.goal
            if not isinstance(goal_pose, move_cube.Pose):
                goal_pose = move_cube.Pose.from_dict(goal_pose)
            object_pose = move_cube.Pose.from_dict(dict(
                position=observation['object_position'].flatten(),
                orientation=observation['object_orientation'].flatten()))
            self.initializer.update_initializer(object_pose, goal_pose)
        return observation, reward, is_done, self.info


@configurable(pickleable=True)
class PushReorientCubeEnv(PushCubeEnv):
    observation_names = [
            "robot_position",
            "robot_velocity",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position",
            "goal_object_orientation",
        ]

    def __init__(self, *args, **kwargs):
        super(PushReorientCubeEnv, self).__init__(*args, **kwargs)

        spaces = TriFingerPlatform.spaces

        self.observation_space = gym.spaces.Dict(
            {
                "robot_position": spaces.robot_position.gym,
                "robot_velocity": spaces.robot_velocity.gym,
                "robot_tip_positions": gym.spaces.Box(
                    low=np.array([spaces.object_position.low] * 3),
                    high=np.array([spaces.object_position.high] * 3),
                ),
                "object_position": spaces.object_position.gym,
                "object_orientation": spaces.object_orientation.gym,
                "goal_object_position": spaces.object_position.gym,
                "goal_object_orientation": spaces.object_orientation.gym,
            }
        )

    def _create_observation(self, t):
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
        return observation


@configurable(pickleable=True)
class SparseCubeEnv(CubeEnv):
    def __init__(
            self,
            initializer,
            action_type=ActionType.POSITION,
            frameskip=1,
            visualization=False,
            pos_thresh=DIST_THRESH,
            ori_thresh=ORI_THRESH
            ):
        super(SparseCubeEnv, self).__init__(initializer, action_type,
                frameskip, visualization)
        self.pos_thresh = pos_thresh
        self.ori_thresh = ori_thresh

    def compute_reward(self, achieved_goal, desired_goal, info):
        goal_pose = move_cube.Pose.from_dict(desired_goal)
        obj_pose = move_cube.Pose.from_dict(achieved_goal)
        pos_error = np.linalg.norm(goal_pose.position - obj_pose.position)
        ori_error = compute_orientation_error(goal_pose, obj_pose, scale=False)
        return float(pos_error < self.pos_thresh and ori_error < self.ori_thresh)


@configurable(pickleable=True)
class TaskSpaceWrapper(gym.ActionWrapper):
    def __init__(self, env, goal_env=False, relative=True, scale=.008, ac_pen=0.001):
        super(TaskSpaceWrapper, self).__init__(env)
        assert self.unwrapped.action_type == ActionType.TORQUE, ''
        spaces = TriFingerPlatform.spaces
        self.goal_env = goal_env
        self.relative = relative
        low = np.array([spaces.object_position.low]* 3).flatten()
        high = np.array([spaces.object_position.high]* 3).flatten()
        if relative:
            low = -np.ones_like(low)
            high = np.ones_like(high)
        self.action_space = gym.spaces.Box(low=low, high=high)
        self.scale = scale
        self.pinocchio_utils = None
        self.ac_pen = ac_pen

    def reset(self):
        obs = super(TaskSpaceWrapper, self).reset()
        platform = self.unwrapped.platform
        if self.pinocchio_utils is None:
            self.pinocchio_utils = CustomPinocchioUtils(
                    platform.simfinger.finger_urdf_path,
                    platform.simfinger.tip_link_names)
        self._prev_obs = obs
        self._last_action = np.zeros_like(self.action_space.sample())
        return obs

    def step(self, action):
        o, r, d, i = super(TaskSpaceWrapper, self).step(action)
        self._prev_obs = o
        if self.relative:
            r -= self.ac_pen * np.linalg.norm(action)
        else:
            r -= self.ac_pen * np.linalg.norm(self._last_action - action)
        self._last_action =  action
        return o, r, d, i

    def action(self, action):
        obs = self._prev_obs
        poskey, velkey = 'robot_position', 'robot_velocity'
        if self.goal_env:
            obs, poskey, velkey = obs['observation'], 'position', 'velocity'
        current_position, current_velocity = obs[poskey], obs[velkey]

        if self.relative:
            fingertip_goals = self.pinocchio_utils.forward_kinematics(current_position.flatten())
            fingertip_goals = np.asarray(fingertip_goals)
            fingertip_goals = fingertip_goals + self.scale * action.reshape((3,3))
        else:
            fingertip_goals = action
        torque, goal_reached = c_utils.impedance_controller(
                fingertip_goals, current_position, current_velocity,
                self.pinocchio_utils, None, 0.007)
        torque = np.clip(torque, self.unwrapped.action_space.low,
                         self.unwrapped.action_space.high)
        return torque


@configurable(pickleable=True)
class ScaledActionWrapper(gym.ActionWrapper):
    def __init__(self, env, goal_env=False, relative=True, scale=POS_SCALE):
        super(TaskSpaceWrapper, self).__init__(env)
        assert self.unwrapped.action_type == ActionType.POSITION, 'position control only'
        self.spaces = TriFingerPlatform.spaces
        self.goal_env = goal_env
        self.relative = relative
        low = self.action_space.low
        high = self.action_space.high
        if relative:
            low = -np.ones_like(low)
            high = np.ones_like(high)
        self.action_space = gym.spaces.Box(low=low, high=high)
        self.scale = scale

    @property
    def pinocchio_utils(self):
        assert self.platform, 'platform must be reset to use pinocchio'
        return self.platform.pinocchio_utils

    def reset(self):
        obs = super(TaskSpaceWrapper, self).reset()
        platform = self.unwrapped.platform
        self._prev_obs = obs
        self._last_action = np.zeros_like(self.action_space.sample())
        return obs

    def step(self, action):
        o, r, d, i = super(TaskSpaceWrapper, self).step(action)
        self._prev_obs = o
        if self.relative:
            r -= self.ac_pen * np.linalg.norm(action)
        else:
            r -= self.ac_pen * np.linalg.norm(self._last_action - action)
        self._last_action =  action
        return o, r, d, i

    def action(self, action):
        obs = self._prev_obs
        poskey, velkey = 'robot_position', 'robot_velocity'
        if self.goal_env:
            obs, poskey, velkey = obs['observation'], 'position', 'velocity'
        current_position, current_velocity = obs[poskey], obs[velkey]
        if self.relative:
            goal_position = current_position + self.scale * action
        else:
            pos_low, pos_high = self.spaces.robot_position.low, self.spaces.robot_position.high
            pos_low = np.clip(current_position - self.scale, pos_low)
            pos_high = np.clip(current_position + self.scale, pos_high)
            goal_position = np.clip(action, pos_low, pos_high)
        return goal_position


@configurable(pickleable=True)
class ReorientWrapper(gym.Wrapper):
    def __init__(self, env, goal_env=True, rew_bonus=REW_BONUS,
                 dist_thresh=0.09, ori_thresh=np.pi/6):
        super(ReorientWrapper, self).__init__(env)
        if not isinstance(self.unwrapped.initializer, ReorientInitializer):
            initializer = ReorientInitializer(initial_dist=0.1)
            self.unwrapped.initializer = initializer
        self.goal_env = goal_env
        self.rew_bonus = rew_bonus
        self.dist_thresh = dist_thresh
        self.ori_thresh = ori_thresh

    def step(self, action):
        o, r, d, i = super(ReorientWrapper, self).step(action)
        i['is_success'] = self.is_success(o)
        if i['is_success']:
            r += self.rew_bonus
        return o, r, d, i

    def is_success(self, observation):
        if self.goal_env:
            goal_pose = move_cube.Pose.from_dict(observation['desired_goal'])
            obj_pose = move_cube.Pose.from_dict(observation['achieved_goal'])
        else:
            goal_pose = move_cube.Pose.from_dict(self.unwrapped.goal)
            obj_pose = move_cube.Pose.from_dict(
                    dict(position=observation['object_position'],
                         orientation=observation['object_orientation']))

        obj_dist = np.linalg.norm(obj_pose.position - goal_pose.position)
        ori_dist = compute_orientation_error(goal_pose, obj_pose, quad=True)
        return obj_dist < self.dist_thresh and ori_dist < self.ori_thresh


class FlattenGoalWrapper(gym.ObservationWrapper):
    """Wrapper to make rrc env baselines and VDS compatible"""
    def __init__(self, env):
        super(FlattenGoalWrapper, self).__init__(env)
        self._sample_goal_fun = None
        self._max_episode_steps = env._max_episode_steps
        self.observation_space = gym.spaces.Dict({
            k: flatten_space(v)
            for k, v in env.observation_space.spaces.items()
            })

    def update_goal_sampler(self, goal_sampler):
        self._sample_goal_fun = goal_sampler

    def sample_goal_fun(self, **kwargs):
        return self._sample_goal_fun(**kwargs)

    @property
    def goal(self):
        return np.concatenate([self.unwrapped.goal['position'],
                               self.unwrapped.goal['orientation']])

    @goal.setter
    def goal(self, g):
        if isinstance(g, dict):
            self.unwrapped.goal = g
            return
        pos, ori = g[...,:3], g[...,3:]
        self.unwrapped.goal = {'position': pos, 'orientation': ori}

    def compute_reward(self, achieved_goal, desired_goal, info):
        if len(achieved_goal.shape) > 1:
            r = []
            info = {"difficulty": self.initializer.difficulty}
            for i in range(achieved_goal.shape[0]):
                pos, ori = achieved_goal[i,:3], achieved_goal[i,3:]
                ag = dict(position=pos, orientation=ori)
                pos, ori = desired_goal[i,:3], desired_goal[i,3:]
                dg = dict(position=pos, orientation=ori)
                r.append(self.env.compute_reward(ag, dg, info))
            return np.array(r)
        achieved_goal = dict(position=achieved_goal[...,:3], orientation=achieved_goal[...,3:])
        desired_goal = dict(position=desired_goal[...,:3], orientation=desired_goal[...,3:])
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def _sample_goal(self):
        return np.concatenate(list(self.initializer.get_goal().to_dict().values()))

    def reset(self, *args, reset_goal=True):
        self.env._elapsed_steps = 0
        obs = super(FlattenGoalWrapper, self).reset(*args)
        if reset_goal and self._sample_goal_fun is not None:
            self.goal = self.sample_goal_fun(obs_dict=obs)
        goal_object_pose = move_cube.Pose.from_dict(self.unwrapped.goal)
        self.unwrapped.goal_marker = visual_objects.CubeMarker(
            width=0.065,
            position=goal_object_pose.position,
            orientation=goal_object_pose.orientation,
            physicsClientId=self.platform.simfinger._pybullet_client_id,
        )
        obs = self.unwrapped._create_observation(0)
        return self.observation(obs)

    def observation(self, observation):
        observation = {k: gym.spaces.flatten(self.env.observation_space[k], v)
                for k, v in observation.items()}
        return observation


# DEPRECATED, USE CubeRewardWrapper INSTEAD
class DistRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, target_dist=0.2, dist_coef=1., ori_coef=1.,
                 ac_norm_pen=0.2, final_step_only=True, augment_reward=True,
                 rew_fn='lin'):
        super(DistRewardWrapper, self).__init__(env)
        self._target_dist = target_dist  # 0.156
        self._dist_coef = dist_coef
        self._ori_coef = ori_coef
        self._ac_norm_pen = ac_norm_pen
        self._last_action = None
        self.final_step_only = final_step_only
        self.augment_reward = augment_reward
        self.rew_fn = rew_fn
        print('DistRewardWrapper is deprecated, use CubeRewardWrapper instead')

    @property
    def target_dist(self):
        target_dist = self._target_dist
        if target_dist is None:
            if isinstance(self.initializer, CurriculumInitializer):
                _, target_dist = self.initializer.goal_sample_radius
                target_dist = 2 * target_dist  # use sample diameter
            else:
                target_dist = move_cube._ARENA_RADIUS
        return target_dist

    @property
    def difficulty(self):
        return self.unwrapped.initializer.difficulty

    def reset(self, **reset_kwargs):
        self._last_action = None
        return super(DistRewardWrapper, self).reset(**reset_kwargs)

    def step(self, action):
        self._last_action = action
        observation, reward, done, info = self.env.step(action)
        if self.final_step_only and done:
            return observation, reward, done, info
        else:
            return observation, self.reward(reward, info), done, info

    def reward(self, reward, info):
        final_dist = self.compute_goal_dist(info)
        if self.rew_fn == 'lin':
            rew = self._dist_coef * (1 - final_dist/self.target_dist)
            if self.info['difficulty'] == 4:
                rew += self._ori_coef * (1 - self.compute_orientation_error())
        elif self.rew_fn == 'exp':
            rew = self._dist_coef * np.exp(-final_dist/self.target_dist)
            if self.info['difficulty'] == 4:
                rew += self._ori_coef * np.exp(-self.compute_orientation_error())
        if self.augment_reward:
            rew += reward
        if self._ac_norm_pen:
            rew -= np.linalg.norm(self._last_action) * self._ac_norm_pen
        return rew

    def get_goal_object_pose(self):
        goal_pose = self.unwrapped.goal
        if not isinstance(goal_pose, move_cube.Pose):
            goal_pose = move_cube.Pose.from_dict(goal_pose)
        cube_state = self.unwrapped.platform.cube.get_state()
        object_pose = move_cube.Pose(
                np.asarray(cube_state[0]).flatten(),
                np.asarray(cube_state[1]).flatten())
        return goal_pose, object_pose

    def compute_orientation_error(self, scale=True):
        goal_pose, object_pose = self.get_goal_object_pose()
        orientation_error = compute_orientation_error(goal_pose, object_pose,
                                                      scale=scale, difficulty=self.difficulty)
        return orientation_error

    def compute_goal_dist(self, info):
        goal_pose, object_pose = self.get_goal_object_pose()
        pos_idx = 3 if info['difficulty'] > 3 else 2
        goal_dist = np.linalg.norm(object_pose.position[:pos_idx] -
                                   goal_pose.position[:pos_idx])
        return goal_dist


class CubeRewardWrapper(gym.Wrapper):
    def __init__(self, env, target_dist=0.195, pos_coef=1., ori_coef=0.,
                 fingertip_coef=0., goal_env=False, ac_norm_pen=0.2, rew_fn='exp',
                 augment_reward=False):
        super(CubeRewardWrapper, self).__init__(env)
        self._target_dist = target_dist  # 0.156
        self._pos_coef = pos_coef
        self._ori_coef = ori_coef
        self._fingertip_coef = fingertip_coef
        self._goal_env = goal_env
        self._ac_norm_pen = ac_norm_pen
        self._prev_action = None
        self._prev_obs = None
        self._augment_reward = augment_reward
        self.rew_fn = rew_fn

    @property
    def target_dist(self):
        target_dist = self._target_dist
        if target_dist is None:
            if isinstance(self.initializer, CurriculumInitializer):
                _, target_dist = self.initializer.goal_sample_radius
                target_dist = 2 * target_dist  # use sample diameter
            else:
                target_dist = move_cube._ARENA_RADIUS
        return target_dist

    @property
    def difficulty(self):
        return self.unwrapped.initializer.difficulty

    def reset(self, **reset_kwargs):
        self._prev_action = None
        self._prev_obs = super(CubeRewardWrapper, self).reset(**reset_kwargs)
        return self._prev_obs

    def step(self, action):
        self._prev_action = action
        observation, r, done, info = super(CubeRewardWrapper, self).step(action)
        if self._goal_env:
            reward = self.compute_reward(observation['achieved_goal'],
                                         observation['desired_goal'], info)
            if self._fingertip_coef:
                reward += self.compute_fingertip_reward(observation['observation'],
                                                  self._prev_obs['observation'])
        else:
            goal_pose, prev_object_pose = self.get_goal_object_pose(self._prev_obs)
            goal_pose, object_pose = self.get_goal_object_pose(observation)
            reward = self._compute_reward(goal_pose, object_pose, prev_object_pose)
            if self._fingertip_coef:
                reward += self.compute_fingertip_reward(observation, self._prev_obs)
        if self._augment_reward:
            reward += r

        self._prev_obs = observation
        return observation, reward, done, info

    def compute_fingertip_reward(self, observation, previous_observation):
        if not isinstance(observation, dict):
            obs_space = self.unwrapped.observation_space
            if self._goal_env: obs_space = obs_space.spaces['observation']
            observation = self.unflatten_observation(observation, obs_space)
            previous_observation = self.unflatten_observation(previous_observation, obs_space)

        if 'robot_tip_position' in observation:
            prev_ftip_pos = previous_observation['robot_tip_position']
            curr_ftip_pos = observation['robot_tip_position']
        else:
            prev_ftip_pos = self.platform.forward_kinematics(previous_observation['robot_position'])
            curr_ftip_pos = self.platform.forward_kinematics(observation['robot_position'])

        current_distance_from_block = np.linalg.norm(
           curr_ftip_pos - observation["object_position"]
        )
        previous_distance_from_block = np.linalg.norm(
            prev_ftip_pos
            - previous_observation["object_position"]
        )

        step_ftip_rew = (
            previous_distance_from_block - current_distance_from_block
        )
        return self._fingertip_coef * step_ftip_rew

    def compute_reward(self, achieved_goal, desired_goal, info):
        if isinstance(achieved_goal, dict):
            obj_pos, obj_ori = achieved_goal['position'], achieved_goal['orientation']
            goal_pos, goal_ori = desired_goal['position'], desired_goal['orientation']
        else:
            obj_pos, obj_ori = achieved_goal[:3], achieved_goal[3:]
            goal_pos, goal_ori = desired_goal[:3], desired_goal[3:]
        goal_pose = move_cube.Pose(position=goal_pos, orientation=goal_ori)
        object_pose = move_cube.Pose(position=obj_pos, orientation=obj_ori)
        return self._compute_reward(goal_pose, object_pose, info=info)

    def _compute_reward(self, goal_pose, object_pose, prev_object_pose=None, info=None):
        info = info or self.unwrapped.info
        pos_error = self.compute_position_error(goal_pose, object_pose)
        if prev_object_pose is not None:
            prev_pos_error = self.compute_position_error(goal_pose, prev_object_pose)
            step_rew = step_pos_rew = prev_pos_error - pos_error
        else:
            step_rew = 0
        if self.difficulty == 4 or self._ori_coef:
            ori_error = compute_orientation_error(goal_pose, object_pose, scale=True)
            if prev_object_pose is not None:
                prev_ori_error = compute_orientation_error(goal_pose, prev_object_pose, scale=True)
                step_ori_rew = prev_ori_error - ori_error
                step_rew = (step_pos_rew * self._pos_coef +
                            step_ori_rew * self._ori_coef)
            else:
                step_rew = 0
        if self.rew_fn == 'lin':
            rew = self._pos_coef * (1 - pos_error/self.target_dist)
            if self.difficulty == 4 or self._ori_coef:
                rew += self._ori_coef * (1 - ori_error)
        elif self.rew_fn == 'exp':
            rew = self._pos_coef * np.exp(-pos_error/self.target_dist)
            if self.difficulty == 4 or self._ori_coef:
                rew += self._ori_coef * np.exp(-ori_error)

        ac_penalty = -np.linalg.norm(self._prev_action) * self._ac_norm_pen
        info['ac_penalty'] = ac_penalty
        if step_rew:
            info['step_rew'] = step_rew
        info['rew'] = rew
        info['pos_error'] = pos_error
        if self.difficulty == 4 or self._ori_coef:
            info['ori_error'] = ori_error
        total_rew = step_rew * 3 + rew + ac_penalty
        if pos_error < DIST_THRESH or ori_error < ORI_THRESH:
            return 2.5 * ((pos_error < DIST_THRESH) + (ori_error < ORI_THRESH))
        return total_rew

    def unflatten_observation(self, observation, obs_space=None):
        filter_keys = []
        env = self.env
        while env != self.unwrapped:
            if isinstance(env, wrappers.FilterObservation):
                filter_keys = env._filter_keys
            env = env.env

        obs_space = obs_space or self.unwrapped.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            if filter_keys:
                obs_space = gym.spaces.Dict({obs_space[k] for k in filter_keys})
            observation = utils.unflatten(obs_space, observation)
        return observation

    def get_goal_object_pose(self, observation):
        goal_pose = self.unwrapped.goal
        goal_pose = move_cube.Pose.from_dict(goal_pose)
        if not self._goal_env:
            if not isinstance(observation, dict):
                observation = self.unflatten_observation(observation)
            pos, ori = observation['object_position'], observation['object_orientation'],
        elif 'achieved_goal' in observation:
            pos, ori = observation['achieved_goal'][:3], observation['achieved_goal'][3:]
        object_pose = move_cube.Pose(position=pos,
                                     orientation=ori)
        return goal_pose, object_pose

    def compute_position_error(self, goal_pose, object_pose):
        pos_error = np.linalg.norm(object_pose.position - goal_pose.position)
        return pos_error


class LogInfoWrapper(gym.Wrapper):
    valid_keys = ['dist', 'score', 'ori_dist', 'ori_scaled',
                  'is_success', 'is_success_ori', 'is_success_ori_dist']

    def __init__(self, env, info_keys=[]):
        super(LogInfoWrapper, self).__init__(env)
        if isinstance(env.initializer, CurriculumInitializer):
            new_keys = ['init_sample_radius','goal_sample_radius']
            [self.valid_keys.append(k) for k in new_keys if k not in self.valid_keys]
        for k in info_keys:
            assert k.split('final_')[-1] in self.valid_keys, f'{k} is not a valid key'
        self.info_keys = info_keys

    def get_goal_object_pose(self):
        goal_pose = self.unwrapped.goal
        if not isinstance(goal_pose, move_cube.Pose):
            goal_pose = move_cube.Pose.from_dict(goal_pose)
        cube_state = self.unwrapped.platform.cube.get_state()
        object_pose = move_cube.Pose(
                np.asarray(cube_state[0]).flatten(),
                np.asarray(cube_state[1]).flatten())
        return goal_pose, object_pose

    def compute_position_error(self, info, score=False):
        goal_pose, object_pose = self.get_goal_object_pose()
        if score:
            return move_cube.evaluate_state(goal_pose, object_pose,
                                            info['difficulty'])
        pos_idx = 3 if info['difficulty'] > 3 else 2
        return np.linalg.norm(object_pose.position[:pos_idx] -
                              goal_pose.position[:pos_idx])

    def compute_orientation_error(self, info, scale=False):
        goal_pose, object_pose = self.get_goal_object_pose()
        return compute_orientation_error(goal_pose, object_pose, scale=scale)

    def step(self, action):
        o, r, d, i = super(LogInfoWrapper, self).step(action)
        for k in self.info_keys:
            if k not in i:
                shortened_k = k.split('final_')[-1]
                final = shortened_k != k
                if shortened_k == 'score' and final == d:
                    i[k] = self.compute_position_error(i, score=True)
                elif shortened_k == 'dist' and final == d:
                    i[k] = self.compute_position_error(i, score=False)
                elif shortened_k == 'ori_dist' and final == d:
                    i[k] = self.compute_orientation_error(i, scale=False)
                elif shortened_k == 'ori_scaled' and final ==  d:
                    i[k] = self.compute_orientation_error(i, scale=True)
                elif k == 'is_success' and d:
                    i[k] = self.compute_position_error(i) < DIST_THRESH
                elif k == 'is_success_ori' and d:
                    if 'is_success' not in self.info_keys and 'is_success' not in i:
                        k = 'is_success'
                    i[k] = self.compute_orientation_error(i) < ORI_THRESH
                elif k == 'is_success_ori_dist' and d:
                    if 'is_success' not in self.info_keys and 'is_success' not in i:
                        k = 'is_success'
                    i[k] = (self.compute_orientation_error(i) < ORI_THRESH and
                            self.compute_position_error(i) < DIST_THRESH)
                elif k == 'init_sample_radius' and d:
                    initializer = self.unwrapped.initializer
                    sample_radius = np.linalg.norm(initializer.initial_pose.position[:2])
                    i[k] = sample_radius
                elif k == 'goal_sample_radius' and d:
                    initializer = self.unwrapped.initializer
                    sample_radius = np.linalg.norm(initializer.goal_pose.position[:2])
                    i[k] = sample_radius
        self.info = i
        return o, r, d, self.info


class StepRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(StepRewardWrapper, self).__init__(env)
        self._last_rew = 0.

    def reset(self):
        self._last_rew = 0.
        return super(StepRewardWrapper, self).reset()

    def reward(self, reward):
        step_reward = reward - self._last_rew
        self._last_rew = reward
        return step_reward


def compute_orientation_error(goal_pose, actual_pose, scale=False,
                              yaw_only=False, quad=False):
    if yaw_only:
        goal_ori = Rotation.from_quat(goal_pose.orientation).as_euler('xyz')
        goal_ori[:2] = 0
        goal_rot = Rotation.from_euler('xyz', goal_ori)
        actual_ori = Rotation.from_quat(actual_pose.orientation).as_euler('xyz')
        actual_ori[:2] = 0
        actual_rot = Rotation.from_euler('xyz', actual_ori)
    else:
        goal_rot = Rotation.from_quat(goal_pose.orientation)
        actual_rot = Rotation.from_quat(actual_pose.orientation)
    error_rot = goal_rot.inv() * actual_rot
    orientation_error = error_rot.magnitude()
    # computes orientation error symmetric to 4 quadrants of the cube
    if quad:
        orientation_error = orientation_error % (np.pi/2)
        if orientation_error > np.pi/4:
            orientation_error = np.pi/2 - orientation_error
    if scale:
        orientation_error = orientation_error / np.pi
    return orientation_error


def flatten_space(space):
    """Flatten a space into a single ``Box``.
    This is equivalent to ``flatten()``, but operates on the space itself. The
    result always is a `Box` with flat boundaries. The box has exactly
    ``flatdim(space)`` dimensions. Flattening a sample of the original space
    has the same effect as taking a sample of the flattenend space.
    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.
    Example::
        >>> box = Box(0.0, 1.0, shape=(3, 4, 5))
        >>> box
        Box(3, 4, 5)
        >>> flatten_space(box)
        Box(60,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True
    Example that flattens a discrete space::
        >>> discrete = Discrete(5)
        >>> flatten_space(discrete)
        Box(5,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True
    Example that recursively flattens a dict::
        >>> space = Dict({"position": Discrete(2),
        ...               "velocity": Box(0, 1, shape=(2, 2))})
        >>> flatten_space(space)
        Box(6,)
        >>> flatten(space, space.sample()) in flatten_space(space)
        True
    """
    if isinstance(space, Box):
        return Box(space.low.flatten(), space.high.flatten())
    if isinstance(space, Discrete):
        return Box(low=0, high=1, shape=(space.n, ))
    if isinstance(space, Tuple):
        space = [flatten_space(s) for s in space.spaces]
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
        )
    if isinstance(space, Dict):
        space = [flatten_space(s) for s in space.spaces.values()]
        return Box(
            low=np.concatenate([s.low for s in space]),
            high=np.concatenate([s.high for s in space]),
        )
    if isinstance(space, MultiBinary):
        return Box(low=0, high=1, shape=(space.n, ))
    if isinstance(space, MultiDiscrete):
        return Box(
            low=np.zeros_like(space.nvec),
            high=space.nvec,
        )
    raise NotImplementedError

