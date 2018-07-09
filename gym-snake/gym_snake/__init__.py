from gym.envs.registration import register

register(
    id='snake-v0',
    entry_point='gym_snake.envs:SnakeEnv',
)

register(
    id='snake-v1',
    entry_point='gym_snake.envs:SnakeEnv',
    kwargs={'screen_width' : 15,
    		'screen_height': 15,
    		'n_actors': 2},
)
