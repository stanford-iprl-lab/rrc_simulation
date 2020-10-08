"""
Script to run policy on one sample (from an output file)
To run:
python run_eval.py --dir </path/to/output/directory/> --l <difficulty level of sample> --i <sample number>
flags:
-v: visualize
"""
import os
import os.path as osp
import gym
import numpy as np
import pickle
import argparse

from scripts.run_evaluate_policy_all_levels import TestSample
from rrc_simulation.gym_wrapper.envs import cube_env, custom_env
print("before control_env")
from rrc_simulation.gym_wrapper.envs import control_env
print("before residual plicy")
from rrc_simulation.gym_wrapper.envs.control_env import ResidualPolicyWrapper
from rrc_simulation.tasks import move_cube
from rrc_simulation.control import control_policy
from rrc_simulation.control.controller_utils import PolicyMode

# Choose with rl policy to load:
#rl_load_dir = './scripts/models/push_reorient/push_reorient_s0/'
#rl_load_dir = './scripts/models/scaled_actions/scaled_actions_s0'
# Uncomment below if not using RL
rl_load_dir = ''

# Set start mode based on if there is a policy directory specified
if not rl_load_dir:
    start_mode = PolicyMode.TRAJ_OPT
else:
    start_mode = PolicyMode.RL_PUSH
    
parser = argparse.ArgumentParser()

parser.add_argument(
    "--dir",
    type=str,
    help="Directory containing the generated log files.",
)

parser.add_argument(
    "--i",
    type=int,
    help="Sample number",
)

parser.add_argument(
    "--l",
    type=int,
    help="Difficulty level",
)

parser.add_argument(
    "--visualize",
    "-v",
    action="store_true",
    help="Visualize with GUI.",
)

args = parser.parse_args()

# load samples
sample_file = os.path.join(args.dir, "test_data.p")
with open(sample_file, "rb") as fh:
    test_data = pickle.load(fh)

for sample in test_data:
    if sample.iteration != args.i or sample.difficulty != args.l:
        continue

    # Set init and goal poses
    init_pose_json = sample.init_pose_json
    goal_pose_json = sample.goal_pose_json

initial_pose = move_cube.Pose.from_json(init_pose_json)
goal_pose = move_cube.Pose.from_json(goal_pose_json)
difficulty = args.l
#goal_pose.position = np.array([0, 0, 0.06])

def get_ac_log(rpath):
    with open(rpath, 'rb') as fh:
        action_log = pickle.load(fh)
    return action_log

def run_eval(initial_pose, goal_pose, difficulty=2, sample=None, rl_load_dir=None, n_eps=3, start_mode=None):
    if sample:
        intial_pose = move_cube.Pose.from_json(sample.init_pose_json)
        goal_pose = move_cube.Pose.from_json(sample.goal_pose_json)
        difficulty = sample.difficulty
    initializer = cube_env.FixedInitializer(
        difficulty, initial_pose, goal_pose
    )
    action_type = cube_env.ActionType.TORQUE_AND_POSITION

    env = gym.make(
        "rrc_simulation.gym_wrapper:real_robot_challenge_phase_1-v1",
        initializer=initializer,
        action_type=action_type,
        visualization=args.visualize,
    )
    policy = control_policy.HierarchicalControllerPolicy(action_space=env.action_space,
                initial_pose=initial_pose, goal_pose=goal_pose,
                load_dir=rl_load_dir, difficulty=difficulty, start_mode=start_mode, debug_waypoints=True)
    env = ResidualPolicyWrapper(env, policy)
    rews, infos = [], []
    for _ in range(n_eps):
        obs, done = env.reset(), False
        old_mode = policy.mode
        custom_env.reset_camera()
        while not done:
            obs, r, done, i = env.step(policy.predict(obs))
            if policy.mode != old_mode:
                print('Mode changed: {} to {}'.format(old_mode, policy.mode))
                old_mode = policy.mode
        print('final_info:', i)

run_eval(initial_pose, goal_pose, rl_load_dir=rl_load_dir, difficulty=difficulty, start_mode=start_mode)

