from gym.envs.registration import register

register(
    id='nas-v0',
    entry_point='gym_ytopt.envs:NasEnv',
)
