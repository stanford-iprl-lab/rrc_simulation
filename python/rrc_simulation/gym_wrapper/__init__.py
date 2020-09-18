from gym.envs.registration import register
from rrc_simulation.gym_wrapper.envs import cube_env


register(
    id="real_robot_challenge_phase_1-v1",
    entry_point="rrc_simulation.gym_wrapper.envs.cube_env:CubeEnv",
)

register(
    id="real_robot_challenge_phase_1-v2",
    entry_point="rrc_simulation.gym_wrapper.envs.custom_env:PushCubeEnv",
)

register(
    id="real_robot_challenge_phase_1-v3",
    entry_point="rrc_simulation.gym_wrapper.envs.custom_env:PushReorientCubeEnv",
)

register(
    id="real_robot_challenge_phase_1-v4",
    entry_point="rrc_simulation.gym_wrapper.envs.custom_env:SparseCubeEnv",
)
