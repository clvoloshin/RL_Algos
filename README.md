# RL_Algos

## Directory Key

./research/slitherin/ is my research on Slitherin' from OpenAI Request for Research 2.0
./gym-snake is my implementation of the classic snake game, and its generalization
./reinforce.py is a quick replication of a classic RL algorithm: REINFORCE

## Replicating Some Reinforcement Algorithms
To run reinforce.py
Example use:
```
python reinforce.py CartPole-v0 -n 100 -b 1000 --discount .9 -num_exp 1 --animate --seed 16 --use_reward_to_go -bl
```

To use tensorboard:
The ./output directory is created within reinforce.py. Call:

```
tensorboard --logdir ./output/<Experiment path>
```
