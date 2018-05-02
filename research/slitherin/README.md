# Slitherin'
OpenAi requests for research.

<!---
## Visualization
The snake is blue: (0,0,255), but its head is slighly darker (0,0,127).
The barrier is red: (255,0,0)
The food is green: (0,255,0)

![Example of SnakeWorld](./media/example.jpg)
--->

## Replication

To replicate: Clone the directory and follow the following:
~~~
pip install gym
cd /dir/RL_Algos/gym-snake
pip install -e .
cd /dir/RL_Algos/research/slitherin
pip install -e .
python train_simple.py --iterations 10000 -seq 100 --animate -gpe 1 -bpe 1 -b 64 -uf 500 -bs 100000 --seed 3 -hless
python train_single.py --iterations 10000 -seq 100 --animate -gpe 1 -bpe 1 -b 64 -uf 500 -bs 100000 --seed 5 -hless
~~~
*Make sure you you specify the correct directory for /dir/*

## 1 Snake

I tried two different representations (preprocessing) for the 1 snake scenario: one that is natural and one less so but arguably more generalizable.

### Natural
*Note: all of the matrices shown are rotated by 90 degrees around the center of the matrix so that they match the images. In reality (in system memory), they are in unrotated form.*

Taking the SnakeWorld exactly as is, then making it grayscale. In other words, consider the rendering of the SnakeWorld is:

![Example of SnakeWorld](./media/example.jpg) 

This state is given by and RGB array of size (3, 8, 8) representing (channes, height, width).

<!---
|255|255|255|255|255|255|255|255|
|--|--|--|--|--|--|--|--|
|255|0 |0 |0 |0 |0 |0 |255|
|255|0 |0 |0 |0 |0 |0 |255|
|255|0 |0 |0 |0 |0 |0 |255|
|255|0 |0 |0 |0 |0 |0 |255|
|255|0 |0 |0 |0 |0 |0 |255|
|255|0 |0 |0 |0 |0 |0 |255|
|255|255|255|255|255|255|255|255|

|0|0|0|0|0|0|0|0|
|--|--|--|--|--|--|--|--|
|0|0 |0 |0 |0 |0 |0 |0|
|0|0 |0 |0 |0 |0 |0 |0|
|0|0 |0 |0 |0 |0 |0 |0|
|0|0 |0 |0 |0 |0 |0 |0|
|0|0 |0 |0 |0 |0 |255 |0|
|0|0 |0 |0 |0 |0 |0 |0|
|0|0|0|0|0|0|0|0|

|0|0|0|0|0|0|0|0|
|--|--|--|--|--|--|--|--|
|0|0 |0 |0 |0 |0 |0 |0|
|0|0 |255 |0 |0 |0 |0 |0|
|0|0 |255 |0 |0 |0 |0 |0|
|0|0 |255 |0 |0 |0 |0 |0|
|0|0 |255 |255 |127 |0 |0 |0|
|0|0 |0 |0 |0 |0 |0 |0|
|0|0|0|0|0|0|0|0|
--->

Then we'd convert this to grayscale via the formula gray(R,G,B) = 0.299*R + 0.587*G + 0.114B, achieving:

|.299|.299|.299|.299|.299|.299|.299|.299|
|--|--|--|--|--|--|--|--|
|.299|0 |0 |0 |0 |0 |0 |.299|
|.299|0 |.114 |0 |0 |0 |0 |.299|
|.299|0 |.114 |0 |0 |0 |0 |.299|
|.299|0 |.114 |0 |0 |0 |0 |.299|
|.299|0 |.114 |.114 |.057 |0 |.587 |.299|
|.299|0 |0 |0 |0 |0 |0 |.299|
|.299|.299|.299|.299|.299|.299|.299|.299|

We use the following neural network and hyperparameters

![Natural Structure](./media/simple.png) 

|Param                                              | Number| Description      |
|---------------------------------------------------|-------|------------------|
|Iterations                                         |  10000| Number of full games to play|
|Batch Size                                         |  64   | Mini-batch size for NN|
|Update Frequency                                   |  500  | How often to update target network |
|Double DQN?                                        |  True | Whether or not to use double dqn|
|Prioritized Experience Replay?                     |  False| Whether or not to use a prioritized buffer|
|Buffer Size                                        |  50000| Number of frames stored in the buffer |
|Gamma                                              |  .99  | Factor which prioritizes shorter term rewards |
|Gradient Clip                                      |  None | Maximum gradient allowable|
|Batches per batches                                |  1    | How many minibatches per train step|
|Max epsilon                                        |  1.0  | Probability with which to take random action|
|Min epsilon                                        |  .01  | Probability with which to take random action|
|Epsilon num_steps                                  |  9000 | The number of steps to linearly interpolate between max/min epsilon|
|Start Train t                                      |  25000| How many frames to add to buffer before starting training|
|Frames between learning steps                      |  4    | Number of frames added to buffer between training steps|
|Minimum learning steps per iteration               |  unset| Make sure this number of training steps occurs per iteration|
|Maximum learning steps per iteration               |  unset| Make sure no more than this number of training steps occurs per iteration|
|Maximum number of steps                            |  100  | Number of steps the snake can take before episode completion|    



|Number of steps                                    | Learning Rate|
|---------------------------------------------------|--------------|
| 0                                                 |   1e-3       |
| 20000                                             |   5e-3       |
| 50000                                             |   1e-4       |

Setting seed=3, we yield the following curves:

![Learning Curve Simple](./media/learning_curves_simple.png)
![Loss Curve Simple](./media/loss_curve_simple.png)

Visualizing learning over time:

| Game        | 2500 | 5500      | Post Learning|   
|-------------|------|-----------|--------------|
| |![Simple Start Game](./media/simple_game_2500_epoch_10500.gif)|![Simple Mid Game](./media/simple_game_5500_epoch_67590.gif)|![Simple End Game](./media/simple_game_end_max_cap_300.gif)|
| Frames Seen | 672k | 4300k     | 10000k		     |
| Epsilon     | .75  | .42       | 0 			        | 
| Steps taken | 8    | 100 (max) | 300+ (unbounded)	|
| Food Eaten  | 0    | 5         | 9   			      |

Notice that after 5500 iterations, the snake does well at avoiding walls, but ends itself in an (seemingly) infinite loop around the food.
Similarly, after 10000 interations (post learning) the snake has learned that eating some food is good, but eventually finds itself doing infinite loops near the wall.
This is due to incomplete training; once the snake gets too long it recognizes that it's better to not run into itself than risk getting food and dying.
However, had the snake seen enough samples, it would have been able to get beyond this quirky behavior. This motivated me to implement prioritized experience replay which has been shown before to place priority on samples which improve the model the most rather than hoping to learn through random sampling (which works in expectation, but could take forever).


















### More generalizable
*Note: all of the matrices shown are rotated by 90 degrees around the center of the matrix so that they match the images. In reality (in system memory), they are in unrotated form.*

While before we converted the RGB state of the world to grayscale and used that, here we do something that generalizes beyond color and will (in my hope) work well in describing multi-snake state:

Consider the rendering of the SnakeWorld as:

![Example of SnakeWorld Again](./media/example.jpg) 

The state is given by a list of sparse array of shape (8, 8).

**The odd channel(s) is:**  
  * -1 for boundary
  * 1 for snake body
  * 0 otherwise

**The even channel(s) is:**  
  * -1 for boundary
  * 1 for snake head
  * 0 otherwise

**The last channel is:**  
  * -1 for boundary
  * 1 for food
  * 0 otherwise
  
In the 1 snake case, the list is then of shape (3,8,8). However in the multi-snake case, it's (number of snakes + 1, 8,8).

I believe this will generalize better to multi-snake scenarios (and beyond) because the network need not learn anything about color/gray and
multiple snakes can be represented by adding additional channels using the same stucture.


We use the following neural network and hyperparameters

![General Structure](./media/single.png) 

|Param                                            | Number| Description      |
|-------------------------------------------------|-------|------------------|
|Iterations                                       |  10000| Number of full games to play|
|Batch Size                                       |  128  | Mini-batch size for NN|
|Update Frequency                                 |  500  | How often to update target network |
|Double DQN?                                      |  True | Whether or not to use double dqn|
|Prioritized Experience Replay?  (PER)            |  True | Whether or not to use a prioritized buffer|
|Buffer Size                                      |  50000| Number of frames stored in the buffer |
|Gamma                                            |  .99  | Factor which prioritizes shorter term rewards |
|Gradient Clip                                    |  None | Maximum gradient allowable|
|Batches per batches                              |  1    | How many minibatches per train step|
|Max epsilon                                      |  1.0  | Probability with which to take random action|
|Min epsilon                                      |  .01  | Probability with which to take random action|
|Epsilon num_steps                                |  9500 | The number of steps to linearly interpolate between max/min epsilon|
|Start Train t                                    |  25000| How many frames to add to buffer before starting training|
|Frames between learning steps                    |  4    | Number of frames added to buffer between training steps|
|Minimum learning steps per iteration             |  1    | Make sure this number of training steps occurs per iteration|
|Maximum learning steps per iteration             |  12   | Make sure no more than this number of training steps occurs per iteration|
|Maximum number of steps                          |  200  | Number of steps the snake can take before episode completion|
|Max beta                                         |  1.0  | Factor to balance out bias due to PER|
|Min beta                                         |  .4   | Factor to balance out bias due to PER|
|Beta number of steps                             | 50000 | The number of steps to linearly interpolate between max/min beta|
|Alpha                                            | .5    | Interpolation between greedy and uniform prioritization for PER |  
|Alpha epsilon                                    | 1e-8  | A small number to make sure probabilities are non-zero for PER |  

|Number of steps                                    | Learning Rate|
|---------------------------------------------------|--------------|
| 0                                                 |   1e-3       |
| 20000                                             |   5e-3       |
| 50000                                             |   1e-4       |

Setting seed=5, we yield the following curves:

![Learning Curve Single](./media/learning_curves_single.png)
![Loss Curve Single](./media/loss_curve_single.png)

Visualizing learning over time:

| Game        | 2500 | 5500  | Post Learning|   
|-------------|------|-------|--------------|
| |![Single Start Game](./media/single_game_2500_epoch_11000.gif)|![Single Mid Game](./media/single_game_5500_epoch_45090.gif)|![Single End Game](./media/single_game_end_max_cap_300.gif)|
| Frames Seen | 1400k | 5700k | 12300k		    |
| Epsilon     | .75  | .42   | 0 			      | 
| Steps taken | 200 (max)    | 62 | 187	|
| Food Eaten  | 0    | 12     | 35 (all) |

Through prioritized experience replay, we achieved some nice results where the snake is able to win the game, even given the input state structure which I believe to be more general than the "natural" way of using color.



