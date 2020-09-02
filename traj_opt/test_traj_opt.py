#!/usr/bin/env python3
"""Simple demo on how to use the TriFingerPlatform interface."""
import argparse
import time
import pybullet

import cv2
import numpy as np

from rrc_simulation import trifinger_platform, sample
from rrc_simulation.tasks import move_cube
from fixed_contact_point_system import FixedContactPointSystem

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
    
    system = FixedContactPointSystem(
                                     platform = platform,
                                     obj_pose = cube_pose,
                                     obj_shape = cube_shape,
                                     obj_mass = cube_mass,
                                    )
    
    x = np.zeros(7)
    x[0:3] = cube_pose.position
    x[3] = cube_pose.orientation[3]
    x[4:7] = cube_pose.orientation[0:3]
    print(x)
    system.get_H_w_2_o(x)
    
    if args.save_action_log:
        platform.store_action_log(args.save_action_log)

if __name__ == "__main__":
    main()
