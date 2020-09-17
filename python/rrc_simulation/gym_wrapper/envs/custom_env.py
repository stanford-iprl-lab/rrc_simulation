"""Custom Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import numpy as np
import gym
import pybullet

from gym.spaces import Box
from gym.spaces import Discrete
from gym.spaces import MultiDiscrete
from gym.spaces import MultiBinary
from gym.spaces import Tuple
from gym.spaces import Dict

from rrc_simulation import TriFingerPlatform
from rrc_simulation import visual_objects
from rrc_simulation.tasks import move_cube
from rrc_simulation.gym_wrapper.envs.cube_env import CubeEnv, ActionType
from rrc_simulation.gym_wrapper.utils import configurable
from scipy.spatial.transform import Rotation


MAX_DIST = move_cube._max_cube_com_distance_to_center
DIST_THRESH = move_cube._CUBE_WIDTH / 5
ORI_THRESH = np.pi / 8
REW_BONUS = 0.5

camera_pos = (0.,0.2,-0.2)
camera_dist = 1.0
pitch = -45.
yaw = 0.

reset_camera = lambda: pybullet.resetDebugVisualizerCamera(cameraDistance=camera_dist, 
                                    cameraYaw=yaw,
                                    cameraPitch=pitch,
                                    cameraTargetPosition=camera_pos)

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
        sample_radius = sample_radius or self.levels[self.current_level]
        radius = np.random.uniform(sample_radius_min, sample_radius)
        theta = np.random.uniform(0, 2 * np.pi)

        # x,y-position of the cube
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        return x, y

    def update_initializer(self, final_pose, goal_pose):
        assert np.all(goal_pose.position == self.goal_pose.position)
        self.final_dist = np.roll(self.final_dist, 1)
        final_dist = np.linalg.norm(goal_pose.position - final_pose.position)
        self.final_dist[0] = final_dist
        if self.difficulty == 4:
            self.final_ori = np.roll(self.final_ori, 1)
            self.final_ori[0] = compute_orientation_error(goal_pose, final_pose,
                                                          difficulty=self.difficulty)

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

        self.observation_names = [
            "robot_position",
            "robot_velocity",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position",
        ]

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

        if current_dist_to_goal < DIST_THRESH:
            reward += REW_BONUS

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
            self.info['is_success'] = (
                    np.linalg.norm(observation['object_position'] -
                        observation['goal_object_position']) < DIST_THRESH)
            goal_pose = self.goal
            if not isinstance(goal_pose, move_cube.Pose):
                goal_pose = move_cube.Pose.from_dict(goal_pose)
            object_pose = move_cube.Pose.from_dict(dict(
                position=observation['object_position'].flatten(),
                orientation=observation['object_orientation'].flatten()))
            self.initializer.update_initializer(object_pose, goal_pose)
            self.info['final_score'] = move_cube.evaluate_state(
                goal_pose, object_pose, self.info['difficulty'])
            pos_idx = 3 if self.info['difficulty'] > 3 else 2
            c_pose, g_pose = object_pose.position[:pos_idx], goal_pose.position[:pos_idx]
            self.info['final_dist'] = np.linalg.norm(c_pose - g_pose)
        return observation, reward, is_done, self.info


@configurable(pickleable=True)
class PushReorientCubeEnv(PushCubeEnv):
    def __init__(self, *args, **kwargs):
        super(PushReorientCubeEnv, self).__init__(*args, **kwargs)

        spaces = TriFingerPlatform.spaces
        self.observation_names = [
            "robot_position",
            "robot_velocity",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position",
            "goal_object_orientation",
        ]

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


class FlattenGoalWrapper(gym.ObservationWrapper):
    """Wrapper to make rrc env baselines and VDS compatible"""
    def __init__(self, env, step_rew_thresh=0.01):
        super(FlattenGoalWrapper, self).__init__(env)
        self._sample_goal_fun = None
        self._max_episode_steps = env._max_episode_steps
        self.observation_space = gym.spaces.Dict({
            k: flatten_space(v) 
            for k, v in env.observation_space.spaces.items()
            })
        self._step_rew_thresh = step_rew_thresh

    def update_goal_sampler(self, goal_sampler):
        self._sample_goal_fun = goal_sampler

    def sample_goal_fun(self, **kwargs):
        return self._sample_goal_fun(**kwargs)

    @property
    def goal(self):
        return np.concatenate(list(self.unwrapped.goal.values()))

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
                r.append(self.unwrapped.compute_reward(ag, dg, info))
            return np.array(r)
        achieved_goal = dict(position=achieved_goal[...,:3], orientation=achieved_goal[...,3:])
        desired_goal = dict(position=desired_goal[...,:3], orientation=desired_goal[...,3:])
        return self.unwrapped.compute_reward(achieved_goal, desired_goal, info)
    
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

    def step(self, action):
        o, r, d, i = super(FlattenGoalWrapper, self).step(action)
        step_rew = r / self.frameskip
        i = i.copy()
        i['is_success'] = np.abs(step_rew) < self._step_rew_thresh
        return o, r, d, i

    def observation(self, observation):
        observation = {k: gym.spaces.flatten(self.env.observation_space[k], v)
                for k, v in observation.items()}
        return observation


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


class LogInfoWrapper(gym.Wrapper):
    valid_keys = ['dist', 'score', 'ori_dist', 'ori_scaled',
                  'is_success', 'is_success_ori']

    def __init__(self, env, info_keys=[]):
        super(LogInfoWrapper, self).__init__(env)
        if isinstance(env.initializer, CurriculumInitializer):
            new_keys = ['init_sample_radius','goal_sample_radius']
            [self.valid_keys.append(k) for k in new_keys if k not in self.valid_keys]
        for k in info_keys:
            assert k.split('final_')[-1] in self.valid_keys, f'{k} is not a valid key'
        assert not ('is_success' in info_keys and 'is_success_ori' in info_keys)
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

    def compute_goal_dist(self, info, score=False):
        goal_pose, object_pose = self.get_goal_object_pose()
        if score:
            return move_cube.evaluate_state(goal_pose, object_pose,
                                            info['difficulty'])
        pos_idx = 3 if info['difficulty'] > 3 else 2
        return np.linalg.norm(object_pose.position[:pos_idx] -
                              goal_pose.position[:pos_idx])

    def compute_goal_orientation_dist(self, info, scale=False):
        goal_pose, object_pose = self.get_goal_object_pose()
        return compute_orientation_error(goal_pose, object_pose, scale=scale, quad=True)

    def step(self, action):
        o, r, d, i = super(LogInfoWrapper, self).step(action)
        for k in self.info_keys:
            if k not in i:
                shortened_k = k.split('final_')[-1]
                final = shortened_k != k
                if shortened_k == 'score' and final == d:
                    i[k] = self.compute_goal_dist(i, score=True)
                elif shortened_k == 'dist' and final == d:
                    i[k] = self.compute_goal_dist(i, score=False)
                elif shortened_k == 'ori_dist' and final == d:
                    i[k] = self.compute_goal_orientation_dist(i, scale=False)
                elif shortened_k == 'ori_scaled' and final ==  d:
                    i[k] = self.compute_goal_orientation_dist(i, scale=True)
                elif k == 'is_success' and d:
                    i[k] = self.compute_goal_dist(i) < DIST_THRESH 
                elif k == 'is_success_ori' and d:
                    i['is_success'] = (self.compute_goal_dist(i) < DIST_THRESH
                            and self.compute_goal_orientation_dist(i) < ORI_THRESH)
                elif k == 'init_sample_radius' and d:
                    initializer = self.unwrapped.initializer
                    sample_radius = np.linalg.norm(initializer.initial_pose.position[:2])
                    i[k] = sample_radius
                elif k == 'goal_sample_radius' and d:
                    initializer = self.unwrapped.initializer
                    sample_radius = np.linalg.norm(initializer.goal_pose.position[:2])
                    i[k] = sample_radius

        return o, r, d, i


def compute_orientation_error(goal_pose, actual_pose, scale=False, difficulty=2):
    yaw_only = quad = difficulty != 4
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

