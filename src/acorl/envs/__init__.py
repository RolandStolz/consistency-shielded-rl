from gymnasium.envs.registration import register

register(
    id='acorl-envs/SeekerUnicycleEnv',
    entry_point='acorl.envs.seeker.unicycle:SeekerUnicycleEnv',
    max_episode_steps=200,
)
