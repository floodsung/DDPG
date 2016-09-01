# DDPG

Reimplementing DDPG from Continuous Control with Deep Reinforcement Learning based on OpenAI Gym and Tensorflow

[http://arxiv.org/abs/1509.02971](http://arxiv.org/abs/1509.02971)

It is still a problem to implement Batch Normalization on the critic network. However the actor network works well with Batch Normalization.

Some Mujoco environments are still unsolved on OpenAI Gym.

## Some Evaluations

1 [https://gym.openai.com/evaluations/eval_mviLO6dZTCmtF1KSmprM1w#reproducibility](InvertedPendulum)
2 [https://gym.openai.com/evaluations/eval_PtYUMaEUSwqS3YUYA6MOQ#reproducibility](InvertedDoublePendulum)
3 [https://gym.openai.com/evaluations/eval_MwvKWh5CSp6SO8IAWU4pqw#reproducibility](Hopper unsolved)


## How to use

```
git clone https://github.com/songrotek/DDPG.git
cd DDPG
python gym_ddpg.py

```

## Reference
1 [https://github.com/rllab/rllab](https://github.com/rllab/rllab)

2 [https://github.com/MOCR/DDPG](https://github.com/MOCR/DDPG)




