"""Custom Gym environment for the Real Robot Challenge Phase 1 (Simulation)."""
import numpy as np
import gym

from gym.spaces import Box
from gym.spaces import Discrete
from gym.spaces import MultiDiscrete
from gym.spaces import MultiBinary
from gym.spaces import Tuple
from gym.spaces import Dict
from rrc_simulation import TriFingerPlatform
from rrc_simulation import visual_objects
from rrc_simulation.tasks import move_cube
from rrc_simulation.gym_wrapper.envs.cube_env import ActionType
from rrc_simulation.gym_wrapper.utils import configurable


MAX_DIST = move_cube._max_cube_com_distance_to_center
DIST_THRESH = move_cube._CUBE_WIDTH / 4
REW_BONUS = 1


@configurable(pickleable=True)
class CurriculumInitializer:
    """Initializer that samples random initial states and goals."""

    def __init__(self, difficulty=1, initial_dist=move_cube._CUBE_WIDTH,
                 num_levels=4, num_episodes=5):
        """Initialize.

        Args:
            initial_dist (float): Distance from center of arena
            num_levels (int): Number of steps to maximum radius
            num_episodes (int): Number of episodes to compute mean over
        """
        self.difficulty = difficulty
        self.initial_dist = initial_dist
        self.num_levels = num_levels
        self.current_level = 0
        self.levels = np.linspace(self.initial_dist, MAX_DIST, num_levels)
        self.episode_dist = np.array([np.inf for _ in range(num_episodes)])

    def random_xy(self):
        # sample uniform position in circle (https://stackoverflow.com/a/50746409)
        radius = self.initial_dist * np.sqrt(np.random.sample())
        theta = np.random.uniform(0, 2 * np.pi)

        # x,y-position of the cube
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        return x, y

    def update_initializer(self, final_dist):
        self.episode_dist = np.roll(self.episode_dist, 1)
        self.episode_dist[0] = final_dist
        if self.current_level >= self.num_levels:
            return
        self.current_level += 1
        if np.mean(self.episode_dist) < DIST_THRESH:
            print("UPDATING INITIALIZER TO SAMPLE TO DISTANCE")
            print("Old sampling distance: {}/New sampling distance: {}".format(
                self.initial_dist, self.levels[self.current_level]))
            self.initial_dist = self.levels[self.current_level]

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        x, y = self.random_xy()
        initial_pose = move_cube.sample_goal(difficulty=-1)
        z = initial_pose.position[-1]
        initial_pose.position = np.array((x, y, z))
        return initial_pose

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        return move_cube.sample_goal(difficulty=self.difficulty)


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

        self.info = dict()

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

        reward = 750 * reward_term_1 + 250 * reward_term_2

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
            final_dist = np.linalg.norm(
                observation["goal_object_position"]
                - observation["object_position"]
            )
            self.initializer.update_initializer(final_dist)

        return observation, reward, is_done, self.info


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

