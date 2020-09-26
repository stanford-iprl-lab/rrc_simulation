"""
Implements ImpedanceControllerPolicy class which returns actions to be compatible
with Gym environment
"""

import os
import os.path as osp
import numpy as np
import joblib

from datetime import date
from rrc_simulation import trifinger_platform
from rrc_simulation import run_rrc_sb as sb_utils
from rrc_simulation.tasks import move_cube
from rrc_simulation.control import control_trifinger_platform
from rrc_simulation.control.custom_pinocchio_utils import CustomPinocchioUtils
from rrc_simulation.control import controller_utils as c_utils
from rrc_simulation import visual_objects
from rrc_simulation.gym_wrapper.envs.custom_env import reset_camera
from rrc_simulation.control.controller_utils import PolicyMode
from rrc_simulation.gym_wrapper.envs import rrc_utils
import torch


RESET_TIME_LIMIT = 150
RL_RETRY_STEPS = 70
MAX_RETRIES = 3


class ImpedanceControllerPolicy:
    def __init__(self, action_space=None, initial_pose=None, goal_pose=None,
                 npz_file=None, debug_waypoints=False):
        self.action_space = action_space
        if npz_file is not None:
            self.load_npz(npz_file)
        else:
            self.nGrid = 50
            self.dt = 0.01
        self.flipping = False
        self.debug_waypoints = debug_waypoints
        self.set_init_goal(initial_pose, goal_pose)
        self.setup_logging()
        self.finger_waypoints = None
        self.done_with_primitive = True
        self.init_face = None
        self.goal_face = None
        self.platform = None
        self.step_count = 0 # To keep track of time spent reaching 1 waypoint
        self.max_step_count = 200

    def reset_policy(self, platform=None):
        self.step_count = 0
        if platform:
            self.platform = platform

    def set_init_goal(self, initial_pose, goal_pose, flip=False):
        self.done_with_primitive = False
        self.goal_pose = goal_pose
        self.x0 = np.concatenate([initial_pose.position, initial_pose.orientation])[None]
        if not flip:
            self.x_goal = self.x0.copy()
            self.x_goal[0, :3] = goal_pose.position
            self.flipping = False
        else:
            self.x_goal = np.concatenate([goal_pose.position, goal_pose.orientation])[None]
            self.flipping = True
        self.x0_pos = self.x0[0,0:3]
        self.x0_quat = self.x0[0,3:]
        init_goal_dist = np.linalg.norm(goal_pose.position - initial_pose.position)
        #print(f'init position: {initial_pose.position}, goal position: {goal_pose.position}, '
        #      f'dist: {init_goal_dist}')
        #print(f'init orientation: {initial_pose.orientation}, goal orientation: {goal_pose.orientation}')

    def setup_logging(self):
        x_goal_str = "-".join(map(str, self.x_goal[0,:].tolist()))
        x0_str = "-".join(map(str, self.x0[0,:].tolist()))
        today_date = date.today().strftime("%m-%d-%y")
        self.save_dir = "./logs/{}/x0_{}_xgoal_{}_nGrid_{}_dt_{}".format(
                today_date ,x0_str, x_goal_str, self.nGrid, self.dt)
        # Create directory if it does not exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def load_npz(self, npz_file):
        # Open .npz file and parse
        npzfile   = np.load(npz_file)
        self.nGrid     = npzfile["t"].shape[0]
        self.x_goal    = npzfile["x_goal"]
        self.x0        = npzfile["x0"]
        self.x_soln    = npzfile["x"]
        self.l_wf_soln = npzfile["l_wf"]
        self.dt        = npzfile["dt"]
        self.cp_params = npzfile["cp_params"]

    def set_waypoints(self, observation):
        self.step_count = 0
        self.custom_pinocchio_utils = CustomPinocchioUtils(
                self.platform.simfinger.finger_urdf_path,
                self.platform.simfinger.tip_link_names)
        reset_camera()

      # Get object pose
        obj_pose = get_pose_from_observation(observation)

        # Get initial fingertip positions in world frame
        current_position, _ = get_robot_position_velocity(observation)

        if self.flipping:
            self.cp_params, self.init_face, self.goal_face = c_utils.get_flipping_cp_params(
                obj_pose, self.goal_pose)
            self.flipping_wp = None
        else:
            self.x_soln, self.l_wf_soln, self.cp_params = control_trifinger_platform.run_traj_opt(
                    obj_pose, current_position, self.custom_pinocchio_utils,
                    self.x0, self.x_goal, self.nGrid, self.dt, self.save_dir)

        #print(self.flipping)
        #print(self.cp_params)
        self.goal_reached = False

        # Get object pose
        obj_pose = get_pose_from_observation(observation)

        if self.debug_waypoints and self.finger_waypoints is None:
            # Visual markers
            init_cps = visual_objects.Marker(number_of_goals=3, goal_size=0.008)
            self.finger_waypoints = visual_objects.Marker(number_of_goals=3, goal_size=0.008)

            # Draw target contact points
            # target_cps_wf = control_trifinger_platform.get_cp_wf_list_from_cp_params(self.cp_params, self.x0_pos, self.x0_quat)
            # init_cps.set_state(target_cps_wf)

        # Get initial contact points and waypoints to them
        self.finger_waypoints_list = []
        self.fingertips_init = self.custom_pinocchio_utils.forward_kinematics(current_position)
        for f_i in range(3):
            tip_current = self.fingertips_init[f_i]
            waypoints = c_utils.get_waypoints_to_cp_param(obj_pose, tip_current, self.cp_params[f_i])
            self.finger_waypoints_list.append(waypoints)
        self.pre_traj_waypoint_i = 0
        self.traj_waypoint_i = 0
        self.goal_reached = False

    def predict(self, observation):
        self.step_count += 1
        observation = observation['observation']
        current_position, current_velocity = observation['position'], observation['velocity']
        if len(self.platform._action_log['actions']) > 0:
            object_pose = self.platform.get_object_pose(self.platform._action_log['actions'][-1]['t'])
        else:
            object_pose = self.platform.get_object_pose(0)
        if self.pre_traj_waypoint_i < len(self.finger_waypoints_list[0]):
            # Get fingertip goals from finger_waypoints_list
            self.fingertip_goal_list = []
            for f_i in range(3):
                self.fingertip_goal_list.append(self.finger_waypoints_list[f_i][self.pre_traj_waypoint_i])
            self.tol = 0.009
            self.tip_forces_wf = None
        elif self.flipping:
            self.fingertip_goal_list = self.flipping_wp
            self.tip_forces_wf = None
        # Follow trajectory to lift object
        elif self.traj_waypoint_i < self.nGrid:
            self.fingertip_goal_list = []
            next_cube_pos_wf = self.x_soln[self.traj_waypoint_i, 0:3]
            next_cube_quat_wf = self.x_soln[self.traj_waypoint_i, 3:]

            self.fingertip_goal_list = control_trifinger_platform.get_cp_wf_list_from_cp_params(
                    self.cp_params, next_cube_pos_wf, next_cube_quat_wf)
            # Get target contact forces in world frame 
            self.tip_forces_wf = self.l_wf_soln[self.traj_waypoint_i, :]
            self.tol = 0.007
        if self.debug_waypoints:
            self.finger_waypoints.set_state(self.fingertip_goal_list)
        # currently, torques are not limited to same range as what is used by simulator
        # torque commands are breaking limits for initial and final goal poses that require 
        # huge distances are covered in a few waypoints? Assign # waypoints wrt distance between
        # start and goal
        torque, self.goal_reached = c_utils.impedance_controller(
            self.fingertip_goal_list, current_position, current_velocity,
            self.custom_pinocchio_utils, tip_forces_wf=self.tip_forces_wf,
            tol=self.tol)
        torque = np.clip(torque, self.action_space.low, self.action_space.high)

        if self.goal_reached:
            self.step_count = 0 # Reset step count
            if self.pre_traj_waypoint_i < len(self.finger_waypoints_list[0]):
                self.pre_traj_waypoint_i += 1
                self.goal_reached = False
            if self.flipping:
                fingertips_current = self.custom_pinocchio_utils.forward_kinematics(
                        current_position)
                self.flipping_wp, self.done_with_primitive = c_utils.get_flipping_waypoint(
                        object_pose, self.init_face, self.goal_face,
                        fingertips_current, self.fingertips_init, self.cp_params)
                self.goal_reached = False
            elif self.traj_waypoint_i < self.nGrid:
                # print("trajectory waypoint: {}".format(self.traj_waypoint_i))
                self.traj_waypoint_i += 1
                self.goal_reached = False
        else:
            if self.flipping and self.step_count > self.max_step_count:
                self.done_with_primitive = True

        return torque


class HierarchicalControllerPolicy:
    DIST_THRESH = 0.1
    ORI_THRESH = np.pi / 6
    default_robot_position = trifinger_platform.TriFingerPlatform.spaces.robot_position.default

    def __init__(self, action_space=None, initial_pose=None, goal_pose=None,
                 npz_file=None, load_dir='', load_itr='last',
                 start_mode=PolicyMode.RL_PUSH, difficulty=1, deterministic=True,
                 debug_waypoints=False):
        self.full_action_space = action_space
        action_space = action_space['torque']
        self.impedance_controller = ImpedanceControllerPolicy(
                action_space, initial_pose, goal_pose, npz_file, debug_waypoints=debug_waypoints)
        self.load_policy(load_dir, load_itr, deterministic)
        self.start_mode = start_mode
        self._platform = None
        self.steps_from_reset = 0
        self.step_count = self.rl_start_step = 0
        self.traj_initialized = False
        self.rl_retries = int(self.start_mode == PolicyMode.RL_PUSH)
        self.difficulty = difficulty

    def reset_policy(self, platform=None):
        self.mode = self.start_mode
        self.traj_initialized = False
        self.steps_from_reset = self.step_count = self.rl_start_step = 0
        if platform:
            self._platform = platform
        self.impedance_controller.reset_policy(platform)

    @property
    def platform(self):
        assert self._platform is not None, 'HierarchicalControlPolicy.platform is not set'
        return self._platform

    @platform.setter
    def platform(self, platform):
        assert platform is not None, 'platform is not yet initialized'
        self._platform = platform
        self.impedance_controller.platform = platform

    def load_policy(self, load_dir, load_itr, deterministic=False):
        self.observation_names = []
        if not load_dir:
            self.rl_frameskip = 1
            self.rl_observation_space = None
            self.rl_policy = lambda obs: self.impedance_controller.predict(obs)
        elif osp.exists(load_dir) and 'pyt_save' in os.listdir(load_dir):
            self.load_spinup_policy(load_dir, load_itr, deterministic=deterministic)
        else:
            self.load_sb_policy(load_dir, load_itr)

    def load_sb_policy(self, load_dir, load_itr):
        # loads make_env, make_reorient_env, and make_model helpers
        assert 'HER-SAC' in load_dir, 'only configured HER-SAC policies so far'
        if '_push' in load_dir:
            self.rl_env = sb_utils.make_env()
        else:
            self.rl_env = sb_utils.make_reorient_env()
        self.rl_frameskip = self.rl_env.unwrapped.frameskip
        self.observation_names = list(self.rl_env.unwrapped.observation_space.spaces.keys())
        self.rl_observation_space = self.rl_env.observation_space
        self.sb_policy = sb_utils.make_her_sac_model(None, None)
        self.sb_policy.load(load_dir)
        self.rl_policy = lambda obs: self.sb_policy.predict(obs)[0]

    def load_spinup_policy(self, load_dir, load_itr='last', deterministic=False):
        self.rl_env, self.rl_policy = load_policy_and_env(load_dir, load_itr, deterministic)
        if self.rl_env:
            self.rl_frameskip = self.rl_env.frameskip
        else:
            self.rl_frameskip = 10
        self.observation_names = list(self.rl_env.unwrapped.observation_space.spaces.keys())
        self.rl_observation_space = self.rl_env.observation_space
        print('loaded policy from {}'.format(load_dir))

    def activate_rl(self, obj_pose):
        if self.start_mode != PolicyMode.RL_PUSH or self.rl_retries == MAX_RETRIES:
            return False
        return np.linalg.norm(obj_pose.position[:2] - np.zeros(2)) > self.DIST_THRESH

    def initialize_traj_opt(self, observation):
        obj_pose = get_pose_from_observation(observation)
        goal_pose = get_pose_from_observation(observation, goal_pose=True)

        # TODO: check orientation error
        if (self.activate_rl(obj_pose) and
            self.start_mode == PolicyMode.RL_PUSH and
            self.mode != PolicyMode.RESET):
            if self.mode != PolicyMode.RL_PUSH:
                self.mode = PolicyMode.RL_PUSH
                self.rl_start_step = self.step_count
            elif self.step_count - self.rl_start_step == RL_RETRY_STEPS:
                self.mode = PolicyMode.RESET
            return False
        elif self.mode == PolicyMode.RL_PUSH:
            if self.step_count > 0:
                self.mode = PolicyMode.RESET
                return False
            else: # skips reset if starting at RL_PUSH
                self.mode = PolicyMode.TRAJ_OPT
                return True
        elif (self.mode == PolicyMode.RESET and
              (self.steps_from_reset >= RESET_TIME_LIMIT and
               obj_pose.position[2] < 0.034)):
            self.steps_from_reset = 0
            if self.activate_rl(obj_pose):
                self.rl_retries += 1
                self.mode = PolicyMode.RL_PUSH
                self.rl_start_step = self.step_count
                return False
            else:
                self.mode = PolicyMode.TRAJ_OPT
                return True
        elif self.mode == PolicyMode.TRAJ_OPT:
            return True
        else:
            if self.impedance_controller.done_with_primitive:
                self.mode = PolicyMode.RESET
                return False
            return True

    def set_waypoints(self, observation):
        if self.mode == PolicyMode.TRAJ_OPT:
            init_pose = get_pose_from_observation(observation)
            goal_pose = get_pose_from_observation(observation, goal_pose=True)
            if self.difficulty == 4:
                self.impedance_controller.set_init_goal(
                        init_pose, goal_pose, flip=flip_needed(init_pose, goal_pose))
            else:
                self.impedance_controller.set_init_goal(init_pose, goal_pose)
            self.impedance_controller.set_waypoints(observation)
            self.traj_initialized = True  # pre_traj_wp are initialized
            self.mode = PolicyMode.IMPEDANCE

    def reset_action(self, observation):
        robot_position = observation['observation']['position']
        time_limit_step = RESET_TIME_LIMIT // 3
        if self.steps_from_reset < time_limit_step:  # middle
            robot_position[1::3] = self.default_robot_position[1::3]
        elif time_limit_step <= self.steps_from_reset < 2*time_limit_step:  # tip
            robot_position[2::3] = self.default_robot_position[2::3]
        else:  # base
            robot_position[::3] = self.default_robot_position[::3]
        self.steps_from_reset += 1
        return robot_position

    def predict(self, observation):
        if not self.traj_initialized and self.initialize_traj_opt(observation['impedance']):
            self.set_waypoints(observation['impedance'])

        if self.mode == PolicyMode.RL_PUSH and self.rl_observation_space is not None:
            ac = self.rl_policy(observation['rl'])
            ac = np.clip(ac, self.full_action_space['position'].low,
                         self.full_action_space['position'].high)
        elif self.mode == PolicyMode.RESET:
            ac = self.reset_action(observation['impedance'])
            ac = np.clip(ac, self.full_action_space['position'].low,
                         self.full_action_space['position'].high)
        elif self.mode == PolicyMode.IMPEDANCE:
            ac = self.impedance_controller.predict(observation['impedance'])
            if self.impedance_controller.done_with_primitive:
                self.traj_initialized = False
        else:
            assert False, 'use a different start mode, started with: {}'.format(self.start_mode)
        self.step_count += 1
        return ac


def get_pose_from_observation(observation, goal_pose=False):
    key = 'achieved_goal' if not goal_pose else 'desired_goal'
    return move_cube.Pose.from_dict(observation[key])


def flip_needed(init_pose, goal_pose):
    return (c_utils.get_closest_ground_face(init_pose) !=
            c_utils.get_closest_ground_face(goal_pose))

def get_robot_position_velocity(observation):
    observation = observation['observation']
    return observation['position'], observation['velocity']


def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.
    """

    backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
    env = state['env']

    return env, get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            if deterministic:
                action = model.pi(x)[0].mean.numpy()
            else:
                action = model.act(x)
        return action

    return get_action



