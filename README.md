# Reinforcement Learning
## Deep Deterministic Policy Gradients

We present our implementation of an Reinforcement Learning Algorithm called DeepDGP, presented by [Lillicrap et
al.](https://arxiv.org/abs/1509.02971).

### Usage Instruction
We recommend to use python 3.
Install necessary dependencies like openAI-gym by
``` python
pip3 install gym
pip3 install tensorflow
pip3 install tqdm

``` python
cd src
python3 train.py --env_id='<any openAI-gym continuous environments>'
```

To run our pretrained models
```
cd src
python3 run.py --env_id=Pendulum-v0 --model_dir=../trained_models/Pendulum-v0/
```
