#!/usr/bin/env python3
"""
Testing impedance controller for lifting object
Right now a bit of a mess, but will clean up soon.
"""
import argparse
import time
import matplotlib.pyplot as plt
import pybullet

import cv2
import numpy as np

from rrc_simulation import trifinger_platform, sample
from rrc_simulation.tasks import move_cube
from custom_pinocchio_utils import CustomPinocchioUtils
from controller_utils import *

# Lists for storing values to plot
fingertip_pos_list = [[],[],[]] # Containts 3 lists, one for each finger
x_pos_list = [] # Object positions
x_quat_list = [] # Object positions

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--enable-cameras",
      "-c",
      action="store_true",
      help="Enable camera observations.",
  )
  parser.add_argument(
      "--save_sim_mp4",
      "-s",
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

  # Open .npz file and parse
  npzfile = np.load(args.npz_file)
  nGrid   = npzfile["t"].shape[0]
  x_dim   = npzfile["x_dim"]
  dx_dim  = npzfile["dx_dim"]
  t_soln  = npzfile["t"]
  x_goal  = npzfile["x_goal"]
  x_soln  = npzfile["x"]
  dx_soln = npzfile["dx"]
  l_soln  = npzfile["l"]
  l_wf_soln  = npzfile["l_wf"]
  dt      = npzfile["dt"]
  cp_params = npzfile["cp_params"]

  # Set initial object pose, for testing
  init_object_pose = move_cube.Pose(
                position=np.array([1,1,0]),
                orientation=np.array([1,0,0,0]),
            )
  init_object_pose = None # Use default init pose

  platform = trifinger_platform.TriFingerPlatform(
      visualization=args.visualize, enable_cameras=args.enable_cameras, initial_object_pose=init_object_pose
  )

  # Instantiate custom pinocchio utils class for access to Jacobian
  custom_pinocchio_utils = CustomPinocchioUtils(platform.simfinger.finger_urdf_path, platform.simfinger.tip_link_names) 
  
  fingertip_goal_list = []
  #cube_pos_wf, cube_quat_wf = platform.cube.get_state()
  cube_half_size = move_cube._CUBE_WIDTH/2 + 0.005 # Fudge the cube dimensions slightly for computing contact point positions in world frame to account for fingertip radius

  pybullet.resetDebugVisualizerCamera(cameraDistance=1.54, cameraYaw=4.749999523162842, cameraPitch=-42.44065475463867, cameraTargetPosition=(-0.11500892043113708, 0.6501579880714417, -0.6364855170249939))
  # MP4 logging
  mp4_save_string = "./test.mp4"
  if args.save_sim_mp4:
    pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, mp4_save_string)

  finger_action = platform.Action(position=platform.spaces.robot_position.default)
  t = platform.append_desired_action(finger_action)

  for waypoint_i in range(nGrid):
  # For testing - hold joints at initial positions
    #while(1):
    #  finger_action = platform.Action(position=platform.spaces.robot_position.default)
    #  t = platform.append_desired_action(finger_action)
    #  time.sleep(platform.get_time_step())

    #  # Debug visualizer camera params
    #  camParams = pybullet.getDebugVisualizerCamera()
    #  print("cameraDistance={}, cameraYaw={}, cameraPitch={}, cameraTargetPosition={}".format(camParams[-2], camParams[-4], camParams[-3], camParams[-1]))

    # Get fingertip target positions from trajectory
    goal_reached = False
    fingertip_goal_list = []
    next_cube_pos_wf = x_soln[waypoint_i, 0:3]
    next_cube_quat_wf = x_soln[waypoint_i, 3:]
    for i in range(3):
      fingertip_goal_list.append(get_cp_wf_from_cp_param(cp_params[i], next_cube_pos_wf, next_cube_quat_wf, cube_half_size))

    while not goal_reached:
      # Get joint positions        
      current_position = platform.get_robot_observation(t).position

      # Joint velocities
      current_velocity = platform.get_robot_observation(t).velocity
      
      # Get target contact forces in world frame 
      tip_forces_wf = l_wf_soln[waypoint_i, :]
      #print("desired tip forces: {}".format(tip_forces_wf))

      torque, goal_reached = impedance_controller(
                                    fingertip_goal_list,
                                    current_position,
                                    current_velocity,
                                    custom_pinocchio_utils,
                                    #tip_forces_wf = None,
                                    tip_forces_wf = tip_forces_wf,
                                    )

      finger_action = platform.Action(torque=torque)
      t = platform.append_desired_action(finger_action)

      # Save current state for plotting
      # Add fingertip positions to list
      current_position = platform.get_robot_observation(t).position
      for finger_id in range(3):
        tip_current = custom_pinocchio_utils.forward_kinematics(current_position)[finger_id]
        fingertip_pos_list[finger_id].append(tip_current)
      # Add current object pose to list
      cube_pos_wf, cube_quat_wf = platform.cube.get_state()
      x_pos_list.append(cube_pos_wf)
      x_quat_list.append(cube_quat_wf)

      time.sleep(platform.get_time_step())

  """
  PLOTTING
  """
  total_timesteps = t

  # Plot end effector trajectory
  fingertip_pos_array = np.array(fingertip_pos_list)
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
    plt.plot(list(range(total_timesteps)), np.ones(total_timesteps)*fingertip_goal_list[i][0], ":", c="C0", label="x_goal")
    plt.plot(list(range(total_timesteps)), np.ones(total_timesteps)*fingertip_goal_list[i][1], ":", c="C1", label="y_goal")
    plt.plot(list(range(total_timesteps)), np.ones(total_timesteps)*fingertip_goal_list[i][2], ":", c="C2", label="z_goal")
  plt.legend()

  plt.figure()
  plt.title("Object position".format(i))
  plt.plot(list(range(total_timesteps)), x_pos_array[:,0], c="C0", label="x")
  plt.plot(list(range(total_timesteps)), x_pos_array[:,1], c="C1", label="y")
  plt.plot(list(range(total_timesteps)), x_pos_array[:,2], c="C2", label="z")
  plt.legend()
  
  plt.show()
  #plt.savefig("{}/object_position.png".format(save_dir))

  if args.save_action_log:
      platform.store_action_log(args.save_action_log)
    
if __name__ == "__main__":
    main()
