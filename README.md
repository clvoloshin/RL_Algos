# RL_Algos
OpenAI Requests for Research & Replicating Some Reinforcement Algorithms

*For Slitherin' from OpenAI Request for Research 2.0 see /research/slitherin/*

For a quick replication of REINFORCE algorithm, see reinforce.py
Example use:
```
python reinforce.py CartPole-v0 -n 100 -b 1000 --discount .9 -num_exp 1 --animate --seed 16 --use_reward_to_go -bl
```

To use tensorboard:
The ./output directory is created within reinforce.py. Call:

```
tensorboard --logdir ./output/<Experiment path>
```
