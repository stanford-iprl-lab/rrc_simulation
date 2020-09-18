#!/usr/bin/env python3
"""
Testing impedance controller for lifting object
Right now a bit of a mess, but will clean up soon.
"""
import argparse
import time
from datetime import date, datetime
import matplotlib.pyplot as plt
import pybullet
import os
from tqdm import tqdm

import cv2
import numpy as np

from rrc_simulation import trifinger_platform, sample, visual_objects
from rrc_simulation.tasks import move_cube
from rrc_simulation.control.custom_pinocchio_utils import CustomPinocchioUtils
from rrc_simulation.control.controller_utils import *
from rrc_simulation.traj_opt.fixed_contact_point_opt import FixedContactPointOpt

num_fingers = 3
episode_length = move_cube.episode_length
#episode_length = 500

DIFFICULTY = 4
def main(args):
  if args.npz_file is not None:
    # Open .npz file and parse
    npzfile   = np.load(args.npz_file)
    nGrid     = npzfile["t"].shape[0]
    x_goal    = npzfile["x_goal"]
    x0        = npzfile["x0"]
    x_soln    = npzfile["x"]
    l_wf_soln = npzfile["l_wf"]
    dt        = npzfile["dt"]
    cp_params = npzfile["cp_params"]

  else:
    x0 = np.array([[0.01559624,0.04523149,0.0325,0,0,0.9622262,0.27225125]])
    yaw = 0
    x_goal = np.array([[0,0,0.05 + 0.0325,0,0,np.sin(yaw/2),np.cos(yaw/2)]]) 
    nGrid = 50
    dt = 0.01

  # Save directory
  x_goal_str = "-".join(map(str,x_goal[0,:].tolist()))
  x0_str = "-".join(map(str,x0[0,:].tolist()))
  today_date = date.today().strftime("%m-%d-%y")
  save_dir = "./logs/{}/x0_{}_xgoal_{}_nGrid_{}_dt_{}".format(today_date ,x0_str, x_goal_str, nGrid, dt)
  # Create directory if it does not exist
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  
  # Set initial object pose to match npz file
  x0_pos = x0[0,0:3]
  x0_quat = x0[0,3:]
  #x0_pos = np.array([0, 0.01, 0.0325]) # test
  #x0_pos = np.array([1, 1, 0.0325]) # test
  #x0_quat = np.array([0, 0, -0.707, 0.707]) # test
  init_object_pose = move_cube.Pose(
                position=x0_pos,
                orientation=x0_quat,
            )
  #init_object_pose = None # Use default init pose

  platform = trifinger_platform.TriFingerPlatform(
      visualization=args.visualize, enable_cameras=args.enable_cameras, initial_object_pose=init_object_pose
  )

  # Instantiate custom pinocchio utils class for access to Jacobian
  custom_pinocchio_utils = CustomPinocchioUtils(platform.simfinger.finger_urdf_path, platform.simfinger.tip_link_names) 
  
  if args.npz_file is None:
    x_soln, l_wf_soln, cp_params = run_traj_opt(platform, custom_pinocchio_utils, x0, x_goal, nGrid, dt, save_dir)

  fingertip_pos_list, x_pos_list, x_quat_list, x_goal, fingertip_goal_list = run_episode(platform,
                                                                              custom_pinocchio_utils,
                                                                              nGrid,
                                                                              x0,
                                                                              x_goal,
                                                                              x_soln,
                                                                              l_wf_soln,
                                                                              cp_params,
                                                                              )

  plot_state(save_dir, fingertip_pos_list, x_pos_list, x_quat_list, x_goal,fingertip_goal_list)

"""
Given intial state, run trajectory optimization
"""
def run_traj_opt(obj_pose, current_position, custom_pinocchio_utils, x0, x_goal, nGrid, dt, save_dir):
  # obj_pose = platform.get_object_pose(0)
  # current_position = platform.get_robot_observation(0).position
  init_fingertip_pos_list = [[],[],[]] # Containts 3 lists, one for each finger
  for finger_id in range(3):
    tip_current = custom_pinocchio_utils.forward_kinematics(current_position)[finger_id]
    init_fingertip_pos_list[finger_id].append(tip_current)

  # Get initial contact points and waypoints to them
  cp_params = get_initial_cp_params(obj_pose, init_fingertip_pos_list)

  cube_shape = (move_cube._CUBE_WIDTH, move_cube._CUBE_WIDTH, move_cube._CUBE_WIDTH)
  cube_mass = 0.02

  # Formulate and solve optimization problem
  opt_problem = FixedContactPointOpt(
                                    nGrid     = nGrid, # Number of timesteps
                                    dt        = dt,   # Length of each timestep (seconds)
                                    cp_params = cp_params,
                                    x0        = x0,
                                    x_goal    = x_goal,
                                    obj_shape = cube_shape,
                                    obj_mass  = cube_mass,
                                    )
  x_soln     = np.array(opt_problem.x_soln)
  l_wf_soln  = np.array(opt_problem.l_wf_soln)
  cp_params  = np.array(cp_params)

  # Save solution in npz file
  save_string = "{}/trajectory".format(save_dir)
  np.savez(save_string,
           dt         = opt_problem.dt,
           x0         = x0,
           x_goal     = x_goal,
           t          = opt_problem.t_soln,
           x          = opt_problem.x_soln,
           dx         = opt_problem.dx_soln,
           l          = opt_problem.l_soln,
           l_wf       = opt_problem.l_wf_soln,
           cp_params  = np.array(cp_params),
           obj_shape  = cube_shape,
           obj_mass   = cube_mass,
           fnum       = opt_problem.system.fnum, 
           qnum       = opt_problem.system.qnum, 
           x_dim      = opt_problem.system.x_dim, 
           dx_dim     = opt_problem.system.dx_dim, 
          )
  return x_soln, l_wf_soln, cp_params
  
"""
Run episode
Inputs:
nGrid
"""
def run_episode(platform, custom_pinocchio_utils,
                nGrid,
                x0,
                x_goal,
                x_soln,
                l_wf_soln,
                cp_params,
                ):

  # Lists for storing values to plot
  fingertip_pos_list = [[],[],[]] # Containts 3 lists, one for each finger
  fingertip_goal_log = [[],[],[]] # Containts 3 lists, one for each finger
  x_pos_list = [] # Object positions
  x_quat_list = [] # Object positions

  x0_pos = x0[0,0:3]
  x0_quat = x0[0,3:]

  #cube_half_size = move_cube._CUBE_WIDTH/2
  cube_half_size = move_cube._CUBE_WIDTH/2 + 0.008 # Fudge the cube dimensions slightly for computing contact point positions in world frame to account for fingertip radius

  pybullet.resetDebugVisualizerCamera(cameraDistance=1.54, cameraYaw=4.749999523162842, cameraPitch=-42.44065475463867, cameraTargetPosition=(-0.11500892043113708, 0.6501579880714417, -0.6364855170249939))
  # MP4 logging
  mp4_save_string = "./test.mp4"
  if args.save_viz_mp4:
    pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, mp4_save_string)

  # Take first action
  finger_action = platform.Action(position=platform.spaces.robot_position.default)
  t = platform.append_desired_action(finger_action)
  # Get object pose
  obj_pose = platform.get_object_pose(t)

  # Visual markers
  init_cps = visual_objects.Marker(number_of_goals=num_fingers, goal_size=0.008)
  finger_waypoints = visual_objects.Marker(number_of_goals=num_fingers, goal_size=0.008)

  # Draw target contact points
  target_cps_wf = get_cp_wf_list_from_cp_params(cp_params, x0_pos, x0_quat, cube_half_size)
  init_cps.set_state(target_cps_wf)

  # Get initial fingertip positions in world frame
  current_position = platform.get_robot_observation(t).position

  # Get initial contact points and waypoints to them
  finger_waypoints_list = []
  for f_i in range(3):
    tip_current = custom_pinocchio_utils.forward_kinematics(current_position)[f_i]
    waypoints = get_waypoints_to_cp_param(obj_pose, cube_half_size, tip_current, cp_params[f_i])
    finger_waypoints_list.append(waypoints)
  
  pre_traj_waypoint_i = 0
  traj_waypoint_i = 0
  goal_reached = False
  reward = 0
  goal_pose = move_cube.Pose(position=x_goal[0,0:3], orientation=x_goal[0,3:])

  for timestep in tqdm(range(episode_length)):

    # Get joint positions        
    current_position = platform.get_robot_observation(t).position
    # Joint velocities
    current_velocity = platform.get_robot_observation(t).velocity

    # Follow trajectory to position fingertips before moving to object
    if pre_traj_waypoint_i < len(finger_waypoints_list[0]):
      # Get fingertip goals from finger_waypoints_list
      fingertip_goal_list = []
      for f_i in range(num_fingers):
        fingertip_goal_list.append(finger_waypoints_list[f_i][pre_traj_waypoint_i])
      tol = 0.009
      tip_forces_wf = None
    # Follow trajectory to lift object
    elif traj_waypoint_i < nGrid:
      fingertip_goal_list = []
      next_cube_pos_wf = x_soln[traj_waypoint_i, 0:3]
      next_cube_quat_wf = x_soln[traj_waypoint_i, 3:]

      fingertip_goal_list = get_cp_wf_list_from_cp_params(cp_params,
                                                          next_cube_pos_wf,
                                                          next_cube_quat_wf,
                                                          cube_half_size)
      # Get target contact forces in world frame 
      tip_forces_wf = l_wf_soln[traj_waypoint_i, :]
      tol = 0.008
    
    finger_waypoints.set_state(fingertip_goal_list)

    torque, goal_reached = impedance_controller(
                                  fingertip_goal_list,
                                  current_position,
                                  current_velocity,
                                  custom_pinocchio_utils,
                                  tip_forces_wf = tip_forces_wf,
                                  tol           = tol
                                  )

    if goal_reached:
      goal_reached = False
      if pre_traj_waypoint_i < len(finger_waypoints_list[0]):
        pre_traj_waypoint_i += 1
      elif traj_waypoint_i < nGrid:
        print("trajectory waypoint: {}".format(traj_waypoint_i))
        traj_waypoint_i += 1

    # Save current state for plotting
    # Add fingertip positions to list
    current_position = platform.get_robot_observation(t).position
    for finger_id in range(3):
      tip_current = custom_pinocchio_utils.forward_kinematics(current_position)[finger_id]
      fingertip_pos_list[finger_id].append(tip_current)
      fingertip_goal_log[finger_id].append(fingertip_goal_list[finger_id])
    # Add current object pose to list
    obj_pose = platform.get_object_pose(t)
    x_pos_list.append(obj_pose.position)
    x_quat_list.append(obj_pose.orientation)

    # Accumulate reward
    r = -move_cube.evaluate_state(
            goal_pose,
            obj_pose,
            DIFFICULTY,
        )
    reward += r
  
    clipped_torque = np.clip(
            np.asarray(torque),
            -platform._max_torque_Nm,
            +platform._max_torque_Nm,
        )
    # Check torque limits
    #print("Torque upper limits: {}".format(platform.spaces.robot_torque))
    if not platform.spaces.robot_torque.gym.contains(clipped_torque):
      print("Time {} Actual torque: {}".format(t, clipped_torque))
    #if not platform.spaces.robot_position.gym.contains(current_position):
    #  print("Actual position: {}".format(current_position))

    # Apply torque action
    finger_action = platform.Action(torque=clipped_torque)
    t = platform.append_desired_action(finger_action)

    #time.sleep(platform.get_time_step())

  # Compute score
  print("Reward: {}".format(reward))

  return fingertip_pos_list, x_pos_list, x_quat_list, x_goal, fingertip_goal_log
  
"""
PLOTTING
"""
def plot_state(save_dir, fingertip_pos_list, x_pos_list, x_quat_list, x_goal, fingertip_goal_list):
  total_timesteps = episode_length

  # Plot end effector trajectory
  fingertip_pos_array = np.array(fingertip_pos_list)
  fingertip_goal_array = np.array(fingertip_goal_list)
  x_pos_array = np.array(x_pos_list)
  x_quat_array = np.array(x_quat_list)

  ## Object position
  plt.figure(figsize=(12, 9))
  plt.subplots_adjust(hspace=0.3)
  for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.title("Fingertip {} position".format(i))
    plt.plot(list(range(total_timesteps)), fingertip_pos_array[i,:,0], c="C0", label="x")
    plt.plot(list(range(total_timesteps)), fingertip_pos_array[i,:,1], c="C1", label="y")
    plt.plot(list(range(total_timesteps)), fingertip_pos_array[i,:,2], c="C2", label="z")
    plt.plot(list(range(total_timesteps)), fingertip_goal_array[i,:,0], ":", c="C0", label="x_goal")
    plt.plot(list(range(total_timesteps)), fingertip_goal_array[i,:,1], ":", c="C1", label="y_goal")
    plt.plot(list(range(total_timesteps)), fingertip_goal_array[i,:,2], ":", c="C2", label="z_goal")
  plt.legend()
  if args.save_state_log:
    plt.savefig("{}/fingertip_positions.png".format(save_dir))

  plt.figure()
  plt.suptitle("Object pose")
  plt.figure(figsize=(6, 12))
  plt.subplots_adjust(hspace=0.3)
  plt.subplot(2, 1, 1)
  plt.title("Object position")
  for i in range(3):
    plt.plot(list(range(total_timesteps)), x_pos_array[:,i], c="C{}".format(i), label="actual - dimension {}".format(i))
    plt.plot(list(range(total_timesteps)), np.ones(total_timesteps)*x_goal[0,i], ":", c="C{}".format(i), label="goal")
  plt.legend()
  
  plt.subplot(2, 1, 2)
  plt.title("Object Orientation")
  for i in range(4):
    plt.plot(list(range(total_timesteps)), x_quat_array[:,i], c="C{}".format(i), label="actual - dimension {}".format(i))
    plt.plot(list(range(total_timesteps)), np.ones(total_timesteps)*x_goal[0,i+3], ":", c="C{}".format(i), label="goal")
  plt.legend()

  if args.save_state_log:
    plt.savefig("{}/object_position.png".format(save_dir))

""" Get contact point positions in world frame from cp_params
"""
def get_cp_wf_list_from_cp_params(cp_params, cube_pos, cube_quat, cube_half_size):
  # Get contact points in wf
  fingertip_goal_list = []
  for i in range(num_fingers):
    fingertip_goal_list.append(get_cp_wf_from_cp_param(cp_params[i], cube_pos, cube_quat, cube_half_size))
  return fingertip_goal_list

"""
For testing - hold joints at initial position
"""
def _test_hold_initial_state():
  # For testing - hold joints at initial positions
  while(1):
    finger_action = platform.Action(position=platform.spaces.robot_position.default)
    t = platform.append_desired_action(finger_action)
    time.sleep(platform.get_time_step())

    # Debug visualizer camera params
    camParams = pybullet.getDebugVisualizerCamera()
    print("cameraDistance={}, cameraYaw={}, cameraPitch={}, cameraTargetPosition={}".format(camParams[-2], camParams[-4], camParams[-3], camParams[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enable-cameras",
        "-c",
        action="store_true",
        help="Enable camera observations.",
    )
    parser.add_argument(
        "--save_state_log",
        "-s",
        action="store_true",
        help="Save plots of state over episode",
    )
    parser.add_argument(
        "--save_viz_mp4",
        "-sv",
        action="store_true",
        help="Save MP4 of visualization.",
    )
    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Visualize with GUI.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of motions that are performed.",
    )
    parser.add_argument(
        "--save-action-log",
        type=str,
        metavar="FILENAME",
        help="If set, save the action log to the specified file.",
    )

    parser.add_argument("--npz_file")

    args = parser.parse_args()

    main(args)
