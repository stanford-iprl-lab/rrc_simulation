import IPython

import gym
import os
import os.path as osp
import time
import numpy as np

from stable_baselines import HER, SAC
from stable_baselines.common.atari_wrappers import FrameStack
from rrc_simulation.gym_wrapper.envs import custom_env
from spinup.utils import rrc_utils
from stable_baselines.common.vec_env import DummyVecEnv


def make_reorient_env():
    info_keys = ['is_success', 'is_success_ori_dist', 'dist', 'final_dist', 'final_score',
                 'final_ori_dist']

    wrappers = [gym.wrappers.ClipAction,
                {'cls': custom_env.LogInfoWrapper,
                 'kwargs': dict(info_keys=info_keys)},
                {'cls': custom_env.CubeRewardWrapper,
                 'kwargs': dict(pos_coef=1., ori_coef=1.,
                                ac_norm_pen=0.2, rew_fn='exp',
                                goal_env=True)},
                {'cls': custom_env.ReorientWrapper,
                 'kwargs': dict(goal_env=True, dist_thresh=0.06)},
                {'cls': gym.wrappers.TimeLimit,
                 'kwargs': dict(max_episode_steps=rrc_utils.EPLEN)},
                custom_env.FlattenGoalWrapper]
    initializer = custom_env.ReorientInitializer(1, 0.1)
    env_fn = rrc_utils.make_env_fn('real_robot_challenge_phase_1-v1', wrapper_params=wrappers,
                                   action_type=rrc_utils.action_type,
                                   initializer=initializer,
                                   frameskip=rrc_utils.FRAMESKIP,
                                   visualization=False)
    env = env_fn()
    return env


def make_env():
    info_keys = ['is_success', 'is_success_ori_dist', 'dist', 'final_dist', 'final_score',
                 'final_ori_dist', 'init_sample_radius']

    wrappers = [gym.wrappers.ClipAction,
                {'cls': custom_env.LogInfoWrapper,
                 'kwargs': dict(info_keys=info_keys)},
                {'cls': gym.wrappers.TimeLimit,
                 'kwargs': dict(max_episode_steps=rrc_utils.EPLEN)},
                custom_env.FlattenGoalWrapper]
    cube_wrapper = {'cls': custom_env.CubeRewardWrapper,
                    'kwargs': dict(pos_coef=1., ori_coef=1.,
                                ac_norm_pen=0.2, rew_fn='exp',
                                goal_env=True)}
    initializer = custom_env.ReorientInitializer(1, 0.1)
    env_fn = rrc_utils.make_env_fn('real_robot_challenge_phase_1-v4', wrapper_params=wrappers,
                                   action_type=rrc_utils.action_type,
                                   initializer=initializer,
                                   frameskip=rrc_utils.FRAMESKIP,
                                   visualization=False)
    env = env_fn()
    return env


def make_exp_dir():
    exp_root = './data'
    hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = 'HER-SAC_sparse_push'
    exp_dir = osp.join(exp_root, exp_name, hms_time)
    os.makedirs(exp_dir)
    return exp_dir


def make_model(env, exp_dir):
    model = HER('MlpPolicy', env, SAC, n_sampled_goal=4,
                tensorboard_log=exp_dir,
                goal_selection_strategy='future',
                verbose=1, buffer_size=int(1e4),
                learning_rate=3e-5,
                gamma=0.95, batch_size=256,
                policy_kwargs=dict(layers=[256, 256]))
    return model


def train_save_model(model, exp_dir, steps=1e6, reset_num_timesteps=False):
# Train for 1e6 steps
    model.learn(int(steps), reset_num_timesteps=reset_num_timesteps)
# Save the trained agent
    model.save(osp.join(exp_dir, '{}-steps'.format(model.num_timesteps)))
    return model


def main():
    env = make_reorient_env()
    exp_dir = make_exp_dir()
    model = make_model(env, exp_dir)
    train_save_model(model, exp_dir, 1e6)
    IPython.embed()


if __name__ == '__main__':
    main()
