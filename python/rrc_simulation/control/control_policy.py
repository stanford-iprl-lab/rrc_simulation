"""
Implements ImpedenceControllerPolicy class which returns actions to be compatible
with Gym environment
"""

from datetime import date, datetime
import os
import numpy as np

from rrc_simulation import trifinger_platform, sample, visual_objects
from rrc_simulation.control import control_trifinger_platform
from rrc_simulation.control.custom_pinocchio_utils import CustomPinocchioUtils
from rrc_simulation.control.controller_utils import *
from rrc_simulation.traj_opt.fixed_contact_point_opt import FixedContactPointOpt


class ImpedenceControllerPolicy:
    def __init__(self, npz_file=None, initial_pose=None, goal_pose=None):
        if npz_file is not None:
            self.load_npz(npz_file)
        else:
            yaw = 0.
            self.x0 = np.concatenate([initial_pose.position, initial_pose.orientation])[None]
            self.x_goal =  np.concatenate([goal_pose.position, goal_pose.orientation])[None]
            self.nGrid = 20
            self.dt = 0.05
        self.x0_pos = self.x0[0,0:3]
        self.x0_quat = self.x0[0,3:]
        print(f'init position: {initial_pose.position}, goal position: {goal_pose.position}')
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

    def get_waypoints(self, platform, observation):
        self.platform = platform
        self.custom_pinocchio_utils = CustomPinocchioUtils(platform.simfinger.finger_urdf_path, platform.simfinger.tip_link_names)
        self.x_soln, self.l_wf_soln, self.cp_params = control_trifinger_platform.run_traj_opt(
                platform, self.custom_pinocchio_utils, self.x0, self.x_goal, self.nGrid, self.dt, self.save_dir)
        self.pre_traj_waypoint_i = 0
        self.traj_waypoint_i = 0
        self.goal_reached = False

        custom_pinocchio_utils = self.custom_pinocchio_utils
        cube_half_size = move_cube._CUBE_WIDTH/2 + 0.008 # Fudge the cube dimensions slightly for computing contact point positions in world frame to account for fingertip radius

        pybullet.resetDebugVisualizerCamera(cameraDistance=1.54, cameraYaw=4.749999523162842, cameraPitch=-42.44065475463867, cameraTargetPosition=(-0.11500892043113708, 0.6501579880714417, -0.6364855170249939))

# Take first action
        finger_action = self.platform.Action(position=platform.spaces.robot_position.default)
        t = platform.append_desired_action(finger_action)
# Get object pose
        obj_pose = platform.get_object_pose(t)

# Visual markers
        init_cps = visual_objects.Marker(number_of_goals=3, goal_size=0.008)
        self.finger_waypoints = visual_objects.Marker(number_of_goals=3, goal_size=0.008)

# Draw target contact points
        target_cps_wf = control_trifinger_platform.get_cp_wf_list_from_cp_params(self.cp_params, self.x0_pos, self.x0_quat, cube_half_size)
        init_cps.set_state(target_cps_wf)

# Get initial fingertip positions in world frame
        current_position = platform.get_robot_observation(t).position

# Get initial contact points and waypoints to them
        self.finger_waypoints_list = []
        for f_i in range(3):
            tip_current = custom_pinocchio_utils.forward_kinematics(current_position)[f_i]
            waypoints = get_waypoints_to_cp_param(obj_pose, cube_half_size, tip_current, self.cp_params[f_i])
            self.finger_waypoints_list.append(waypoints)
        self.goal_reached = False

    def predict(self, observation):
        observation = observation['observation']
        current_position, current_velocity = observation['position'], observation['velocity']
        finger_waypoints_list = self.finger_waypoints_list
        custom_pinocchio_utils = self.custom_pinocchio_utils

        if self.pre_traj_waypoint_i < len(self.finger_waypoints_list[0]):
            # Get fingertip goals from finger_waypoints_list
            fingertip_goal_list = []
            for f_i in range(3):
                fingertip_goal_list.append(finger_waypoints_list[f_i][self.pre_traj_waypoint_i])
            tol = 0.009
            tip_forces_wf = None
        # Follow trajectory to lift object
        elif self.traj_waypoint_i < self.nGrid:
            fingertip_goal_list = []
            next_cube_pos_wf = self.x_soln[self.traj_waypoint_i, 0:3]
            next_cube_quat_wf = self.x_soln[self.traj_waypoint_i, 3:]

            fingertip_goal_list = control_trifinger_platform.get_cp_wf_list_from_cp_params(self.cp_params,
                                                                next_cube_pos_wf,
                                                                next_cube_quat_wf,
                                                                self.cube_half_size)
            # Get target contact forces in world frame 
            tip_forces_wf = self.l_wf_soln[self.traj_waypoint_i, :]
            tol = 0.007

        self.finger_waypoints.set_state(fingertip_goal_list)
        # currently, torques are not limited to same range as what is used by simulator
        # torque commands are breaking limits for initial and final goal poses that require 
        # huge distances are covered in a few waypoints? Assign # waypoints wrt distance between
        # start and goal
        torque, self.goal_reached = impedance_controller(
                                          fingertip_goal_list,
                                          current_position,
                                          current_velocity,
                                          custom_pinocchio_utils,
                                          tip_forces_wf = tip_forces_wf,
                                          tol           = tol
                                          )

        if self.goal_reached:
            self.goal_reached = False
            if self.pre_traj_waypoint_i < len(self.finger_waypoints_list[0]):
                self.pre_traj_waypoint_i += 1
            elif self.traj_waypoint_i < self.nGrid:
                # print("trajectory waypoint: {}".format(self.traj_waypoint_i))
                self.traj_waypoint_i += 1
        return torque

