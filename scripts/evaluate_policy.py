#!/usr/bin/env python3
"""Example evaluation script to evaluate a policy.

This is an example evaluation script for evaluating a "RandomPolicy".  Use this
as a base for your own script to evaluate your policy.  All you need to do is
to replace the `RandomPolicy` and potentially the Gym environment with your own
ones (see the TODOs in the code below).

This script will be executed in an automated procedure.  For this to work, make
sure you do not change the overall structure of the script!

This script expects the following arguments in the given order:
 - Difficulty level (needed for reward computation)
 - initial pose of the cube (as JSON string)
 - goal pose of the cube (as JSON string)
 - file to which the action log is written

It is then expected to initialize the environment with the given initial pose
and execute exactly one episode with the policy that is to be evaluated.

When finished, the action log, which is created by the TriFingerPlatform class,
is written to the specified file.  This log file is crucial as it is used to
evaluate the actual performance of the policy.
"""
import sys

import gym
import numpy as np

from rrc_simulation.gym_wrapper.envs import cube_env, control_env
from rrc_simulation.gym_wrapper.envs.control_env import ResidualPolicyWrapper
from rrc_simulation.tasks import move_cube
from rrc_simulation.control.control_policy import ImpedanceControllerPolicy
from rrc_simulation.control.control_policy import HierarchicalControllerPolicy


class RandomPolicy:
    """Dummy policy which uses random actions."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation):
        return self.action_space.sample()


def main():
    try:
        difficulty = int(sys.argv[1])
        initial_pose_json = sys.argv[2]
        goal_pose_json = sys.argv[3]
        output_file = sys.argv[4]
    except IndexError:
        print("Incorrect number of arguments.")
        print(
            "Usage:\n"
            "\tevaluate_policy.py <difficulty_level> <initial_pose>"
            " <goal_pose> <output_file>"
        )
        sys.exit(1)

    # the poses are passes as JSON strings, so they need to be converted first
    initial_pose = move_cube.Pose.from_json(initial_pose_json)
    goal_pose = move_cube.Pose.from_json(goal_pose_json)
    # initial_pose = move_cube.Pose(position=np.array([0,0,0.0325]), orientation=np.array([0,0,0,1]))
    # goal_pose =  move_cube.Pose(position=np.array([0,0,0.0825]), orientation=np.array([0,0,0,1]))

    # create a FixedInitializer with the given values
    initializer = cube_env.FixedInitializer(
        difficulty, initial_pose, goal_pose
    )

    # TODO: Replace with your environment if you used a custom one.
    if difficulty in [2, 3]:
        action_type = cube_env.ActionType.TORQUE  # _AND_POSITION
    else:
        action_type = cube_env.ActionType.POSITION

    env = gym.make(
        "rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1",
        initializer=initializer,
        action_type=action_type,
        visualization=False,
    )

    # TODO: Replace this with your model
    # Note: You may also use a different policy for each difficulty level (difficulty)
    if difficulty in [2, 3]:
        policy = ImpedanceControllerPolicy(action_space=env.action_space,
                    initial_pose=initial_pose, goal_pose=goal_pose)
        # policy = HierarchicalControllerPolicy(action_space=env.action_space,
        #             initial_pose=initial_pose, goal_pose=goal_pose,
        #             load_dir='./push_reorient/push_reorient_s0')
        # env = ResidualPolicyWrapper(env, policy)
    else:
        policy = RandomPolicy(env.action_space)

    # Execute one episode.  Make sure that the number of simulation steps
    # matches with the episode length of the task.  When using the default Gym
    # environment, this is the case when looping until is_done == True.  Make
    # sure to adjust this in case your custom environment behaves differently!

    is_done = False
    observation = env.reset()

    if isinstance(policy, (HierarchicalControllerPolicy, ImpedanceControllerPolicy)):
        policy.set_waypoints(env.platform, observation)
    accumulated_reward = 0
    while not is_done:
        action = policy.predict(observation)
        action = np.round(action.astype('float64'), 8)
        observation, reward, is_done, info = env.step(action)
        accumulated_reward += reward

    print("Accumulated reward: {}".format(accumulated_reward))
    dist_to_goal = np.linalg.norm(observation['desired_goal']['position'] -
                                  observation['achieved_goal']['position'])
    print(f"Final score: {reward}, Final dist to goal: {dist_to_goal}")

    # store the log for evaluation
    env.platform.store_action_log(output_file)


if __name__ == "__main__":
    main()
