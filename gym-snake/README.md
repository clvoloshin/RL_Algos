# Classic Snake Game

This is an implementation of the classic snake game into an OpenAi gym environment. It has been written to generalize to N snakes.

## Human Play

~~~
pip install gym
pip install -e .

cd ./gym_snake/envs/
python test.py
~~~
Use keys (w,a,s,d) for (up, left, down, right) respectively.

## Computer Self Play

See ../research/slitherin

## Implementation

There are two Snake games available:
  1. snake-v0: 
      *  1 Snake 
      * Shape: 6x6 (8x8 including bounadary)
  2. snake-v1: 
      *  2 Snake 
      * Shape: 15x15 (17x17 including bounadary)
 
 Snake colors are generated randomly unless it is the snake-v0, in which case the snake color is always the same. Food is always the same color.

STATE:
The state of the world is represented by an array of 2*N+1 grids where N is the number of snakes which are alive.
Consider, for simplicity, the snake-v0 game. Then the state of the world is given by:
  1. Body: An 8x8 matrix which has -1 on the border, 1 for the snake's body (including head), 0 elsewhere
  2. Head: An 8x8 matrix which has -1 on the border, 1 for just the snake's head, 0 elsewhere
  3. Food: An 8x8 matrix which has -1 on the border, 1 for each piece of food.

In general the array will look like (Snake 1 body, Snake 1 head,..., Snake N body, Snake N head, food)

REWARD:
A snake recieves:
  1. 1 for finding food
  2. -1 for dying
  3. 0 for staying alive

END:
The game is over when a snake's head:
  1. Hits a boundary
  2. Hits itself
  3. Hits another snake
  *Edge case: If two snakes run into eachother they both die.*

 Visualization:
 While the state is split, the game is rendered as one image like:
 
 ![Example of SnakeWorld](../research/slitherin/media/example.jpg)
 
