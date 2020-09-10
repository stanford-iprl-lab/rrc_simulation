#!/usr/bin/env python3
"""Simple demo on how to use the TriFingerPlatform interface."""
import argparse
import time
from datetime import date, datetime
import os
import pybullet

import cv2
import numpy as np

from rrc_simulation import trifinger_platform, sample
from rrc_simulation.tasks import move_cube
from fixed_contact_point_opt import FixedContactPointOpt

#x_goal = np.array([[0,0,0,0.707,-0.707,0,0]])
x_goal = np.array([[0,0,0.1+0.0325,1,0,0,0]])
#x_goal = np.array([[0.15,0,0.0325,1,0,0,0]])
#x_goal = np.array([[0,0,0.2,1,0,0,0]])
x_goal_str = "-".join(map(str,x_goal[0,:].tolist()))

nGrid = 10
dt = 0.1

# Contact point position parameters
# 1 finger on face 5, 1 finger on face 3
cp_params = [
             [0, 1, 0],
             [0, -1, 0],
             [-1, 0, 0],
            ]
#cp_params = [
#             [0, 0, 1],
#             [0, 0, 1],
#            ]

# Files to save solutions
today_date = date.today().strftime("%m-%d-%y")
save_dir = "./logs/{}".format(today_date)
# Create directory if it does not exist
if not os.path.exists(save_dir):
  os.makedirs(save_dir)
save_string = "{}/fixed_cp_object_xgoal_{}_nGrid_{}_dt_{}".format(save_dir, x_goal_str, nGrid, dt) 

def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()

    #Tester Initial ObjectPose
    initial_object_pose = move_cube.Pose(
                position=np.array([0, 0, 0.0325]),
                orientation=np.array([0.707, 0, 0, 0.707]),
            )
    initial_object_pose = None

    platform = trifinger_platform.TriFingerPlatform(
        visualization=False,
        enable_cameras=False,
        initial_robot_position=np.zeros(9),
        initial_object_pose = initial_object_pose
    )

    t = 0

    # show the latest observations
    robot_observation = platform.get_robot_observation(t)
    print("Finger0 Position: %s" % robot_observation.position[:3])

    cube_pose = platform.get_object_pose(t)
    print("Cube Position: %s" % cube_pose.position)
    print("Cube Orientation: %s" % cube_pose.orientation)

    cube_shape = (move_cube._CUBE_WIDTH, move_cube._CUBE_WIDTH, move_cube._CUBE_WIDTH)
    cube_mass = 0.02

    # Formulate and solve optimization problem
    opt_problem = FixedContactPointOpt(
                                      nGrid     = nGrid, # Number of timesteps
                                      dt        = dt,   # Length of each timestep (seconds)
                                      cp_params = cp_params,
                                      x_goal    = x_goal,
                                      platform  = platform,
                                      obj_pose  = cube_pose,
                                      obj_shape = cube_shape,
                                      obj_mass  = cube_mass,
                                      )
    
    # Save solution in npz file
    np.savez(save_string,
             dt         = opt_problem.dt,
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

    if args.save_action_log:
        platform.store_action_log(args.save_action_log)

if __name__ == "__main__":
    main()
