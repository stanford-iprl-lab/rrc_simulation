"""
Implements ImpedanceControllerPolicy class which returns actions to be compatible
with Gym environment
"""

import os
import os.path as osp
import numpy as np

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
from spinup.utils.test_policy import load_policy_and_env


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

    def set_init_goal(self, initial_pose, goal_pose, flip=False):
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
        print(f'init position: {initial_pose.position}, goal position: {goal_pose.position}, '
              f'dist: {init_goal_dist}')
        print(f'init orientation: {initial_pose.orientation}, goal orientation: {goal_pose.orientation}')

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

    def set_waypoints(self, platform, observation):
        self.step_count = 0
        self.platform = platform
        self.custom_pinocchio_utils = CustomPinocchioUtils(
                platform.simfinger.finger_urdf_path,
                platform.simfinger.tip_link_names)
        reset_camera()

      # Get object pose
        obj_pose = get_pose_from_observation(observation)

        # Get initial fingertip positions in world frame
        current_position, _ = get_robot_position_velocity(observation)

        if self.flipping:
            self.cp_params = c_utils.get_flipping_cp_params(
                obj_pose, self.goal_pose)
            self.flipping_wp = None
        else:
            self.x_soln, self.l_wf_soln, self.cp_params = control_trifinger_platform.run_traj_opt(
                    obj_pose, current_position, self.custom_pinocchio_utils,
                    self.x0, self.x_goal, self.nGrid, self.dt, self.save_dir)

        self.goal_reached = False

        # Get object pose
        obj_pose = get_pose_from_observation(observation)

        if self.debug_waypoints:
            # Visual markers
            init_cps = visual_objects.Marker(number_of_goals=3, goal_size=0.008)
            self.finger_waypoints = visual_objects.Marker(number_of_goals=3, goal_size=0.008)

            # Draw target contact points
            target_cps_wf = control_trifinger_platform.get_cp_wf_list_from_cp_params(self.cp_params, self.x0_pos, self.x0_quat)
            init_cps.set_state(target_cps_wf)

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
        object_pose = self.platform.get_object_pose(self.platform._action_log['actions'][-1]['t'])
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
            if self.pre_traj_waypoint_i < len(self.finger_waypoints_list[0]):
                self.pre_traj_waypoint_i += 1
                self.goal_reached = False
            if self.flipping:
                fingertips_current = self.custom_pinocchio_utils.forward_kinematics(
                        current_position)
                self.flipping_wp, _ = c_utils.get_flipping_waypoint(
                        object_pose, self.goal_pose,
                        fingertips_current, self.fingertips_init, self.cp_params)
                self.goal_reached = False
            elif self.traj_waypoint_i < self.nGrid:
                # print("trajectory waypoint: {}".format(self.traj_waypoint_i))
                self.traj_waypoint_i += 1
                self.goal_reached = False
        return torque


class HierarchicalControllerPolicy:
    DIST_THRESH = 0.09
    ORI_THRESH = np.pi / 6
    default_robot_position = trifinger_platform.TriFingerPlatform.spaces.robot_position.default

    def __init__(self, action_space=None, initial_pose=None, goal_pose=None,
                 npz_file=None, load_dir='', load_itr='last',
                 start_mode=PolicyMode.RL_PUSH, difficulty=1):
        self.full_action_space = action_space
        action_space = action_space['torque']
        self.impedance_controller = ImpedanceControllerPolicy(
                action_space, initial_pose, goal_pose, npz_file)
        self.load_policy(load_dir, load_itr)
        self.mode = start_mode
        self.steps_from_reset = 0
        self._platform = None
        self.traj_initialized = False
        self.difficulty = difficulty

    @property
    def platform(self):
        assert self._platform is not None, 'HierarchicalControlPolicy.platform is not set'
        return self._platform

    @platform.setter
    def platform(self, platform):
        assert platform is not None, 'platform is not yet initialized'
        self._platform = platform

    def load_policy(self, load_dir, load_itr):
        if osp.exists(load_dir) and 'pyt_save' in os.listdir(load_dir):
            self.load_spinup_policy(load_dir, load_itr)
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
        self.rl_observation_space = self.rl_env.observation_space
        self.sb_policy = sb_utils.make_her_sac_model(None, None)
        self.sb_policy.load(load_dir)
        self.policy = lambda obs: self.sb_policy.predict(obs)[0]

    def load_spinup_policy(self, load_dir, load_itr='last', deterministic=True):
        self.rl_env, self.rl_policy = load_policy_and_env(load_dir, load_itr, deterministic)
        if self.rl_env:
            self.rl_frameskip = self.rl_env.frameskip
        else:
            self.rl_frameskip = 10
        self.observation_names = list(self.rl_env.unwrapped.observation_space.spaces.keys())
        self.rl_observation_space = self.rl_env.observation_space
        print('loaded policy from {}'.format(load_dir))

    def initialize_traj_opt(self, observation):
        obj_pose = get_pose_from_observation(observation)
        goal_pose = get_pose_from_observation(observation, goal_pose=True)

        # TODO: check orientation error
        if np.linalg.norm(obj_pose.position - goal_pose.position) > self.DIST_THRESH:
            self.mode = PolicyMode.RL_PUSH
            return False
        elif self.mode == PolicyMode.RL_PUSH:
            self.mode = PolicyMode.RESET
            return False
        elif self.mode == PolicyMode.RESET and self.steps_from_reset >= 30:
            self.mode = PolicyMode.TRAJ_OPT
            self.steps_from_reset = 0
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
            self.impedance_controller.set_waypoints(self.platform, observation)
            self.traj_initialized = True  # pre_traj_wp are initialized
            self.mode = PolicyMode.IMPEDANCE

    def predict(self, observation):
        if not self.traj_initialized and self.initialize_traj_opt(observation['impedance']):
            self.set_waypoints(observation['impedance'])

        if self.mode == PolicyMode.RESET:
            ac = self.default_robot_position
            self.steps_from_reset += 1
        elif self.mode == PolicyMode.IMPEDANCE:
            ac = self.impedance_controller.predict(observation['impedance'])
        elif self.mode == PolicyMode.RL_PUSH:
            ac = self.rl_policy(observation['rl'])
            ac = np.clip(ac, self.full_action_space['position'].low,
                         self.full_action_space['position'].high)
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

