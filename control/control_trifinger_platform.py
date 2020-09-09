#!/usr/bin/env python3
"""Simple demo on how to use the TriFingerPlatform interface."""
import argparse
import time
import matplotlib.pyplot as plt

import cv2
import numpy as np

from rrc_simulation import trifinger_platform, sample
from rrc_simulation.tasks import move_cube
from custom_pinocchio_utils import CustomPinocchioUtils

fingertip_pos_list = [[],[],[]] # Containts 3 lists, one for each finger

"""
Compute joint torques to move fingertips to desired locations
Inputs:
tip_pos_desired_list: List of desired fingertip positions for each finger
q_current: Current joint angles
dq_current: Current joint velocities
"""
def impedance_controller(tip_pos_desired_list, q_current, dq_current, custom_pinocchio_utils):
  torque = 0
  for finger_id in range(3):
    torque += impedance_controller_single_finger(finger_id, tip_pos_desired_list[finger_id], q_current, dq_current, custom_pinocchio_utils)
  
  return torque

"""
Compute joint torques to move fingertip to desired location
Inputs:
finger_id: Finger 0, 1, or 2
x_desired: Desired fingertip pose **ORIENTATION??**
  for orientation: transform fingertip reference frame to world frame (take into account object orientation)
  for now, just track position
q_current: Current joint angles
dq_current: Current joint velocities
"""
def impedance_controller_single_finger(finger_id, x_desired, q_current, dq_current, custom_pinocchio_utils):
  Kp_x = 130
  Kp_y = 130
  Kp_z = 300
  Kp = np.diag([Kp_x, Kp_y, Kp_z])
  Kv_x = Kv_y = 6
  Kv_z = 7
  Kv = np.diag([Kv_x, Kv_y, Kv_z])

  # Compute current fingertip position
  x_current = custom_pinocchio_utils.forward_kinematics(q_current)[finger_id]
  fingertip_pos_list[finger_id].append(x_current)

  delta_x = np.expand_dims(np.array(x_desired) - np.array(x_current), 1)
  print("Current x: {}".format(x_current))
  print("Desired x: {}".format(x_desired))
  
  # Get full Jacobian for finger
  Ji = custom_pinocchio_utils.get_tip_link_jacobian(finger_id, q_current)
  # Just take first 3 rows, which correspond to linear velocities of fingertip
  Ji = Ji[:3, :]

  # Get current fingertip velocity
  dx_current = Ji @ np.expand_dims(np.array(dq_current), 1)

  torque = np.squeeze(Ji.T @ (Kp @ delta_x - Kv @ dx_current))
  print(torque)    
  return torque

def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--enable-cameras",
      "-c",
      action="store_true",
      help="Enable camera observations.",
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
      default=1,
      help="Number of motions that are performed.",
  )
  parser.add_argument(
      "--save-action-log",
      type=str,
      metavar="FILENAME",
      help="If set, save the action log to the specified file.",
  )
  args = parser.parse_args()


  init_object_pose_out_of_stage = move_cube.Pose(
                position=np.array([1,1,0]),
                orientation=np.array([1,0,0,0]),
            )

  platform = trifinger_platform.TriFingerPlatform(
      visualization=args.visualize, enable_cameras=args.enable_cameras, initial_object_pose=init_object_pose_out_of_stage
  )

  # Instantiate custom pinocchio utils class for access to Jacobian
  custom_pinocchio_utils = CustomPinocchioUtils(platform.simfinger.finger_urdf_path, platform.simfinger.tip_link_names) 
  
  # Move the fingers to random positions so that the cube is kicked around
  # (and thus it's position changes).
  for _ in range(args.iterations):
      goal = np.array(
          sample.random_joint_positions(
              number_of_fingers=3,
              lower_bounds=[-1, -1, -2],
              upper_bounds=[1, 1, 2],
          )
      )

      finger_action = platform.Action(position=goal)
      t = platform.append_desired_action(finger_action)
    
      # To test impedance controller for a single finger, set object position to be out of the way

      num_steps = 200
      finger0_goal = [0, 0.1, 0.03]
      finger1_goal = [0.1, 0, 0.04]
      finger2_goal = [-0.1, 0, 0.04]
      fingertip_goal_list = [finger0_goal, finger1_goal, finger2_goal]

      # apply action for a few steps, so the fingers can move to the target
      # position and stay there for a while
      for _ in range(num_steps):
          # Get joint positions        
          current_position = platform.get_robot_observation(t).position

          # Joint velocities
          current_velocity = platform.get_robot_observation(t).velocity

          #torque = impedance_controller_single_finger(0, fingertip_goal, current_position, current_velocity, custom_pinocchio_utils)
          torque = impedance_controller(fingertip_goal_list, current_position, current_velocity, custom_pinocchio_utils)

          finger_action = platform.Action(torque=torque)
          t = platform.append_desired_action(finger_action)
           
          time.sleep(platform.get_time_step())

  # Plot end effector trajectory
  fingertip_pos_array = np.array(fingertip_pos_list)
  #print(x_array)

  ## Object position
  plt.figure(figsize=(12, 9))
  plt.subplots_adjust(hspace=0.3)
  for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.title("Fingertip {} position".format(i))
    plt.plot(list(range(num_steps)), fingertip_pos_array[i,:,0], c="C0", label="x")
    plt.plot(list(range(num_steps)), fingertip_pos_array[i,:,1], c="C1", label="y")
    plt.plot(list(range(num_steps)), fingertip_pos_array[i,:,2], c="C2", label="z")
    plt.plot(list(range(num_steps)), np.ones(num_steps)*fingertip_goal_list[i][0], ":", c="C0", label="x_goal")
    plt.plot(list(range(num_steps)), np.ones(num_steps)*fingertip_goal_list[i][1], ":", c="C1", label="y_goal")
    plt.plot(list(range(num_steps)), np.ones(num_steps)*fingertip_goal_list[i][2], ":", c="C2", label="z_goal")
  plt.legend()
  plt.show()
  #plt.savefig("{}/object_position.png".format(save_dir))

  if args.save_action_log:
      platform.store_action_log(args.save_action_log)
    
if __name__ == "__main__":
    main()
