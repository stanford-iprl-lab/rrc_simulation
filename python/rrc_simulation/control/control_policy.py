"""
Implements ImpedanceControllerPolicy class which returns actions to be compatible
with Gym environment
"""

from datetime import date, datetime
import os
import numpy as np

from gym.spaces import Dict
from rrc_simulation import trifinger_platform, sample, visual_objects
from rrc_simulation.control import control_trifinger_platform
from rrc_simulation.control.custom_pinocchio_utils import CustomPinocchioUtils
from rrc_simulation.control.controller_utils import *
from rrc_simulation.gym_wrapper.envs.custom_env import reset_camera
from rrc_simulation.gym_wrapper.envs.control_env import PolicyMode
from rrc_simulation.traj_opt.fixed_contact_point_opt import FixedContactPointOpt
from spinup.utils.test_policy import load_policy_and_env

class ImpedanceControllerPolicy:
    def __init__(self, action_space=None, initial_pose=None, goal_pose=None,
                 npz_file=None):
        self.action_space = action_space
        if npz_file is not None:
            self.load_npz(npz_file)
        else:
            yaw = 0.
            self.x0 = np.concatenate([initial_pose.position, initial_pose.orientation])[None]
            self.x_goal =  np.concatenate([goal_pose.position, goal_pose.orientation])[None]
            self.nGrid = 50
            self.dt = 0.01
        self.x0_pos = self.x0[0,0:3]
        self.x0_quat = self.x0[0,3:]
        init_goal_dist = np.linalg.norm(goal_pose.position - initial_pose.position)
        print(f'init position: {initial_pose.position}, goal position: {goal_pose.position}, '
              f'dist: {init_goal_dist}')
        print(f'init orientation: {initial_pose.orientation}, goal orientation: {goal_pose.orientation}')
        self.setup_logging()

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

    def get_pose_from_observation(self, observation, goal_pose=False):
        key = 'achieved_goal' if not goal_pose else 'desired_goal'
        return move_cube.Pose.from_dict(observation[key])
    
    def get_robot_position_velocity(self, observation):
        observation = observation['observation']
        return observation['position'], observation['velocity']

    def set_waypoints(self, platform, observation):
        self.step_count = 0
        self.custom_pinocchio_utils = CustomPinocchioUtils(platform.simfinger.finger_urdf_path, platform.simfinger.tip_link_names)
        self.x_soln, self.l_wf_soln, self.cp_params = control_trifinger_platform.run_traj_opt(
                platform, self.custom_pinocchio_utils, self.x0, self.x_goal, self.nGrid, self.dt, self.save_dir)
        self.goal_reached = False

        custom_pinocchio_utils = self.custom_pinocchio_utils
        # Fudge the cube dimensions slightly for computing contact point positions in world frame to account for fingertip radius
        self.cube_half_size = move_cube._CUBE_WIDTH/2 + 0.008

        reset_camera()

        # Get object pose
        obj_pose = self.get_pose_from_observation(observation)

        # Visual markers
        init_cps = visual_objects.Marker(number_of_goals=3, goal_size=0.008)
        self.finger_waypoints = visual_objects.Marker(number_of_goals=3, goal_size=0.008)

        # Draw target contact points
        target_cps_wf = control_trifinger_platform.get_cp_wf_list_from_cp_params(self.cp_params, self.x0_pos, self.x0_quat, self.cube_half_size)
        init_cps.set_state(target_cps_wf)

        # Get initial fingertip positions in world frame
        current_position, _ = self.get_robot_position_velocity(observation)

        # Get initial contact points and waypoints to them
        self.finger_waypoints_list = []
        for f_i in range(3):
            tip_current = custom_pinocchio_utils.forward_kinematics(current_position)[f_i]
            waypoints = get_waypoints_to_cp_param(obj_pose, self.cube_half_size, tip_current, self.cp_params[f_i])
            self.finger_waypoints_list.append(waypoints)
        self.pre_traj_waypoint_i = 0
        self.traj_waypoint_i = 0
        self.goal_reached = False

    def predict(self, observation):
        self.step_count += 1
        if self.step_count == 1:
            return np.zeros_like(self.action_space.low)
        current_position, current_velocity = self.get_robot_position_velocity(observation)

        if self.pre_traj_waypoint_i < len(self.finger_waypoints_list[0]):
            # Get fingertip goals from finger_waypoints_list
            self.fingertip_goal_list = []
            for f_i in range(3):
                self.fingertip_goal_list.append(self.finger_waypoints_list[f_i][self.pre_traj_waypoint_i])
            #print(self.fingertip_goal_list)
            self.tol = 0.009
            self.tip_forces_wf = None
        # Follow trajectory to lift object
        elif self.traj_waypoint_i < self.nGrid:
            self.fingertip_goal_list = []
            next_cube_pos_wf = self.x_soln[self.traj_waypoint_i, 0:3]
            next_cube_quat_wf = self.x_soln[self.traj_waypoint_i, 3:]

            self.fingertip_goal_list = control_trifinger_platform.get_cp_wf_list_from_cp_params(self.cp_params,
                                                                next_cube_pos_wf,
                                                                next_cube_quat_wf,
                                                                self.cube_half_size)
            # Get target contact forces in world frame 
            self.tip_forces_wf = self.l_wf_soln[self.traj_waypoint_i, :]
            self.tol = 0.007

        self.finger_waypoints.set_state(self.fingertip_goal_list)
        # currently, torques are not limited to same range as what is used by simulator
        # torque commands are breaking limits for initial and final goal poses that require 
        # huge distances are covered in a few waypoints? Assign # waypoints wrt distance between
        # start and goal
        torque, self.goal_reached = impedance_controller(
                                          self.fingertip_goal_list,
                                          current_position,
                                          current_velocity,
                                          self.custom_pinocchio_utils,
                                          tip_forces_wf = self.tip_forces_wf,
                                          tol           = self.tol
                                          )
        torque = np.clip(torque, self.action_space.low, self.action_space.high)

        if self.goal_reached:
            if self.pre_traj_waypoint_i < len(self.finger_waypoints_list[0]):
                self.pre_traj_waypoint_i += 1
                self.goal_reached = False
            elif self.traj_waypoint_i < self.nGrid:
                # print("trajectory waypoint: {}".format(self.traj_waypoint_i))
                self.traj_waypoint_i += 1
                self.goal_reached = False
        return torque

    
class HierarchicalControllerPolicy(ImpedanceControllerPolicy):
    DIST_THRESH = 0.09
    ORI_THRESH = np.pi / 6
    default_robot_position = trifinger_platform.TriFingerPlatform.spaces.robot_position.default

    def __init__(self, action_space=None, initial_pose=None, goal_pose=None,
                 npz_file=None, load_dir='', load_itr='last',
                 start_mode=PolicyMode.IMPEDANCE):
        self.full_action_space = action_space
        action_space = action_space['torque']
        super(HierarchicalControllerPolicy, self).__init__(
                action_space, initial_pose, goal_pose, npz_file)
        self.load_policy(load_dir, load_itr)
        self.mode = start_mode
        self.platform = None
        self.init_traj = False

    def load_policy(self, load_dir, load_itr='last', deterministic=True):
        self.rl_env, self.rl_policy = load_policy_and_env(load_dir, load_itr, deterministic)
        if self.rl_env:
            self.rl_frameskip = self.rl_env.frameskip
        else:
            self.rl_frameskip = 10
        self.observation_names = list(self.rl_env.unwrapped.observation_space.spaces.keys())
        self.rl_observation_space = self.rl_env.observation_space
        print('loaded policy from {}'.format(load_dir))
    
    def activate_traj_opt(self, observation):
        obj_pose = self.get_pose_from_observation(observation)
        # TODO: check orientation error
        if np.linalg.norm(obj_pose.position) > self.DIST_THRESH:
            self.mode = PolicyMode.RL_ONLY
            return False
        robot_pos, robot_vel = self.get_robot_position_velocity(observation)
        # UNKNOWN IF THIS WILL WORK, hopefully will retract without bumping cube too much
        if (np.isclose(robot_vel, np.zeros_like(robot_vel)).all()
            and np.isclose(robot_pos, self.default_robot_position).all()):
            self.mode = PolicyMode.TRAJ_OPT
        else:
            self.mode = PolicyMode.RESET
        return True
    
    def set_waypoints(self, platform, observation):
        # stores platform if set_waypoints called later
        if 'impedance' in observation:
            observation = observation['impedance']
        self.platform = self.platform or platform
        self.activate_traj_opt(observation)
        if self.mode == PolicyMode.TRAJ_OPT:
            self.init_traj = True
            super(HierarchicalControllerPolicy, self).set_waypoints(self.platform, observation)
            self.mode = PolicyMode.IMPEDANCE

    def predict(self, observation):
        if not self.init_traj and self.activate_traj_opt(observation['impedance']):
            self.set_waypoints(None, observation['impedance'])
        if self.mode == PolicyMode.RESET:
            ac = self.default_robot_position
        elif self.mode == PolicyMode.IMPEDANCE:
            ac = super(HierarchicalControllerPolicy, self).predict(observation['impedance'])
        elif self.mode == PolicyMode.RL_ONLY:
            ac = self.rl_policy(observation['rl'])
            ac = np.clip(ac, self.full_action_space['position'].low, 
                    self.full_action_space['position'].high)
        return ac
