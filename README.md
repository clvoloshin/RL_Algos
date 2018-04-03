# RL_Algos
Replicating Reinforcement Algorithms

Example use:
```
python reinforce.py CartPole-v0 -n 100 -b 1000 --discount .9 -num_exp 1 --animate --seed 16 --use_reward_to_go -bl
```

To use tensorboard:
The ./output directory is created within reinforce.py. Call:

```
tensorboard --logdir ./output/<Experiment path>
```
