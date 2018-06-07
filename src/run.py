import gym
import argparse
from ddpg import DDPG
from train import parseArgs, params
from envs import Fourbar

FLAGS = parseArgs()
env_id = FLAGS.env_id
model_dir = FLAGS.model_dir

params={'tau':FLAGS.tau, 'gamma':FLAGS.gamma, 'lr_act':FLAGS.lr_act, 'lr_crit':FLAGS.lr_crit, 'batch_size':FLAGS.batch_size, 'buffer_size': FLAGS.buffer_size, 'num_epochs' : FLAGS.num_epochs, 'num_cycles' : FLAGS.num_cycles, 'num_rollouts' : FLAGS.num_rollouts, 'train_steps' : FLAGS.train_steps, 'model_dir' : FLAGS.model_dir, 'stddev': FLAGS.stddev, 'hidden_size':FLAGS.hidden_size, 'critic_l2_reg':FLAGS.critic_l2_reg, 'num_episodes': num_episodes}


if env_id == 'path':
    env = Fourbar('path')
    eval_env = Fourbar('path')
elif env_id == 'motion':
    env = Fourbar('motion')
    eval_env = Fourbar('motion')
else:
    env = gym.make(env_id)
    eval_env = gym.make(env_id)


algo = DDPG(env, eval_env, params)
algo.play(render_eval=True)
