from gym.envs.registration import register

register(
    id='snake-v0',
    entry_point='gym_snake.envs:SnakeEnv',
)

register(
    id='snake-v1',
    entry_point='gym_snake.envs:SnakeEnv',
    screen_width = 25, 
    screen_height=25, 
    n_actors = 2,
)
