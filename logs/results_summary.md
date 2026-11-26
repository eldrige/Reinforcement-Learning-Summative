# Results summary

## DQN

|   run_id | changes                                     |   eval_return |   eval_success |   eval_length |   frames_to_converge |   episodes |   train_time_s |
|---------:|:--------------------------------------------|--------------:|---------------:|--------------:|---------------------:|-----------:|---------------:|
|        0 | baseline                                    |         41.05 |           0.86 |        110.16 |                56000 |       2336 |         286.29 |
|        1 | lr=0.001                                    |         12.23 |           0    |        200    |                84800 |       1500 |         290.56 |
|        2 | lr=0.0001                                   |         53.46 |           1    |         67.5  |                42800 |       2441 |         286.09 |
|        3 | gamma=0.97                                  |         47.97 |           0.92 |         94.16 |                56000 |       2684 |         284.14 |
|        4 | batch_size=32                               |         51.4  |           0.96 |         83.16 |                56100 |       2298 |         272.97 |
|        5 | buffer_size=5000                            |          2.22 |           0    |        200    |                  nan |       1500 |         287.95 |
|        6 | n_step=1                                    |          1.51 |           0    |        200    |               289000 |       1500 |         258.3  |
|        7 | n_step=5                                    |         11.82 |           0    |        200    |                83300 |       1500 |         262.34 |
|        8 | eps_decay_frac=0.6, eps_end=0.1             |         14.39 |           0    |        200    |               178100 |       1500 |         260.69 |
|        9 | target_update=0, tau=0.01, double_dqn=False |         14.7  |           0    |        200    |                73700 |       1500 |         256.12 |

## REINFORCE

|   run_id | changes                                 |   eval_return |   eval_success |   eval_length |   frames_to_converge |   episodes |   train_time_s |
|---------:|:----------------------------------------|--------------:|---------------:|--------------:|---------------------:|-----------:|---------------:|
|        0 | baseline                                |          7    |           0    |        200    |                  nan |       1500 |         158.71 |
|        1 | lr=0.003                                |          5.47 |           0    |        200    |                  nan |       1500 |         159.49 |
|        2 | lr=0.0003                               |        -18.54 |           0    |        200    |                  nan |       1500 |         158.82 |
|        3 | gamma=0.95                              |         49.57 |           1    |         63.42 |               225000 |       2019 |         159.09 |
|        4 | use_baseline=False, normalize_adv=True  |         -2.25 |           0    |        200    |                  nan |       1500 |         160.55 |
|        5 | use_baseline=False, normalize_adv=False |        -18.64 |           0    |        200    |                  nan |       1500 |         159.67 |
|        6 | entropy_coef=0.0                        |        -18.64 |           0    |        200    |                  nan |       1500 |         161.03 |
|        7 | entropy_coef=0.05                       |         49.56 |           1    |         63.52 |               266000 |       1530 |         161.59 |
|        8 | frames_per_batch=3000                   |        -18.54 |           0    |        200    |                  nan |       1500 |         161.19 |
|        9 | hidden=(256, 256)                       |         47.12 |           0.96 |         68.98 |               228000 |       1752 |         162.5  |

## PPO

|   run_id | changes                                                 |   eval_return |   eval_success |   eval_length |   frames_to_converge |   episodes |   train_time_s |
|---------:|:--------------------------------------------------------|--------------:|---------------:|--------------:|---------------------:|-----------:|---------------:|
|        0 | baseline                                                |         49.57 |           1    |         63.42 |                61440 |       3738 |         226.28 |
|        1 | lr=0.001                                                |         18.59 |           0.56 |        142.52 |               270336 |       1712 |         229.8  |
|        2 | lr=0.0001                                               |         49.57 |           1    |         63.42 |               149504 |       2863 |         227.72 |
|        3 | clip_eps=0.1                                            |         49.56 |           1    |         63.46 |               110592 |       3198 |         224.13 |
|        4 | clip_eps=0.3                                            |         49.57 |           1    |         63.42 |               188416 |       2422 |         227.44 |
|        5 | epochs=4                                                |         49.57 |           1    |         63.42 |               110592 |       3292 |         183.91 |
|        6 | gae_lambda=0.8, gamma=0.95                              |         53.42 |           1    |         67.46 |                28672 |       4136 |         216.07 |
|        7 | entropy_coef=0.0                                        |         49.57 |           1    |         63.42 |                71680 |       3750 |         217.08 |
|        8 | entropy_coef=0.05                                       |         50.02 |           1    |         64.08 |                55296 |       3367 |         218.1  |
|        9 | frames_per_batch=1024, minibatch=128, hidden=(256, 256) |          0.95 |           0    |        200    |                  nan |       1500 |         280.4  |

## A2C

|   run_id | changes                           |   eval_return |   eval_success |   eval_length |   frames_to_converge |   episodes |   train_time_s |
|---------:|:----------------------------------|--------------:|---------------:|--------------:|---------------------:|-----------:|---------------:|
|        0 | baseline                          |         49.66 |              1 |         61.42 |                52544 |       3581 |         156.36 |
|        1 | lr=0.0003                         |         49.57 |              1 |         63.42 |               168000 |       2300 |         159.11 |
|        2 | lr=0.002                          |         -6.25 |              0 |        200    |                  nan |       1500 |         207.76 |
|        3 | frames_per_batch=16               |         49.57 |              1 |         63.42 |                17264 |       4121 |         223.07 |
|        4 | frames_per_batch=256              |        -18.84 |              0 |        200    |                  nan |       1500 |         199.71 |
|        5 | gae_lambda=0.95                   |         49.57 |              1 |         63.42 |                33152 |       3275 |         203.19 |
|        6 | gamma=0.95                        |         49.57 |              1 |         63.42 |                36480 |       3932 |         201.06 |
|        7 | entropy_coef=0.0                  |         49.57 |              1 |         63.42 |                92288 |       2745 |         204.54 |
|        8 | entropy_coef=0.05                 |         49.57 |              1 |         63.42 |                79360 |       3135 |         202.89 |
|        9 | optimizer=adam, hidden=(256, 256) |         12.82 |              0 |        200    |               165696 |       1500 |         211.31 |

## Generalisation (best model per algorithm, 100 unseen episodes)

| algo      | test                            |   mean_return |   success_rate |   mean_steps |
|:----------|:--------------------------------|--------------:|---------------:|-------------:|
| DQN       | unseen risk draws, clinic start |         52.1  |           1    |         67.4 |
| DQN       | unseen risk draws, random start |         14.24 |           0.6  |        123.3 |
| REINFORCE | unseen risk draws, clinic start |         48.08 |           1    |         63.2 |
| REINFORCE | unseen risk draws, random start |         38.33 |           0.83 |         84.4 |
| PPO       | unseen risk draws, clinic start |         52.04 |           1    |         67.6 |
| PPO       | unseen risk draws, random start |         18.54 |           0.62 |        116.8 |
| A2C       | unseen risk draws, clinic start |         48.17 |           1    |         61.2 |
| A2C       | unseen risk draws, random start |         43.3  |           1    |         92   |