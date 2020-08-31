from gym.envs.registration import register
from rrc_simulation.gym_wrapper.envs import cube_env


initializer = cube_env.RandomInitializer(difficulty=1)

register(
    id="real_robot_challenge_phase_1-v1",
    kwargs={'initializer': initializer,
            'action_type': cube_env.ActionType.POSITION,
            'frameskip': 100},
    entry_point="rrc_simulation.gym_wrapper.envs.cube_env:CubeEnv",
    max_episode_steps=38
)

