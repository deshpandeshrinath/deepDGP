import gym
from ddpg import DDPG
import argparse

#env_id = 'HalfCheetah-v2'
#env_id = 'Humanoid-v2'
#env_id = 'Ant-v2'
#env_id = 'InvertedDoublePendulum-v2'
#env_id = 'Reacher-v2'
#env_id = 'Swimmer-v2'
#env = gym.make('FetchPickAndPlace-v0')
#env = gym.make('Acrobot-v1')
#eval_env = gym.make('Acrobot-v1')


def parseArgs():
    parser = argparse.ArgumentParser(description='Directory to save model')
    parser.add_argument('--env_id', action="store", dest="env_id", default='Pendulum-v0', help='OpenAI-Gym Env ID')
    parser.add_argument('--model_dir', action="store", dest="model_dir", default='../trained_models/')
    parser.add_argument('--gamma', action="store", type=float, dest="gamma", default=0.99, help='default 0.99')
    parser.add_argument('--tau', action="store", type=float, dest="tau", default=0.001, help='default 0.001')
    parser.add_argument('--lr_crit', action="store", type=float, dest="lr_crit", default=0.001, help='default 0.001')
    parser.add_argument('--lr_act', action="store", type=float, dest="lr_act", default=0.0001, help='default 0.0001')
    parser.add_argument('--batch_size', action="store", type=int, dest="batch_size", default=64, help='default 64')
    parser.add_argument('--buffer_size', action="store", type=int, dest="buffer_size", default=1e5, help='default 100000')
    parser.add_argument('--critic_l2_reg', action="store", type=float, dest="critic_l2_reg", default=1e-2, help='default 0.01')
    parser.add_argument('--num_rollouts', action="store", type=int, dest="num_rollouts", default=30, help='default 100')
    parser.add_argument('--num_epochs', action="store", type=int, dest="num_epochs", default=100, help='default 100')
    parser.add_argument('--num_cycles', action="store", type=int, dest="num_cycles", default=20, help='default 20')
    parser.add_argument('--train_steps', action="store", type=int, dest="train_steps", default=50, help='default 50')
    parser.add_argument('--stddev', action="store", type=float, dest="stddev", default=0.1, help='default 0.1')
    parser.add_argument('--hidden_size', action="store", type=int, dest="hidden_size", default=64, help='default 64')
    parser.add_argument('-version', action="store", dest="version", default='', help='default is blank')
    return parser.parse_args()


FLAGS = parseArgs()
env_id = FLAGS.env_id
model_dir = FLAGS.model_dir

params={'tau':FLAGS.tau, 'gamma':FLAGS.gamma, 'lr_act':FLAGS.lr_act, 'lr_crit':FLAGS.lr_crit, 'batch_size':FLAGS.batch_size, 'buffer_size': FLAGS.buffer_size, 'num_epochs' : FLAGS.num_epochs, 'num_cycles' : FLAGS.num_cycles, 'num_rollouts' : FLAGS.num_rollouts, 'train_steps' : FLAGS.train_steps, 'model_dir' : FLAGS.model_dir + FLAGS.env_id + FLAGS.version, 'stddev': FLAGS.stddev, 'hidden_size':FLAGS.hidden_size, 'critic_l2_reg':FLAGS.critic_l2_reg}

if __name__ == '__main__':
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    algo = DDPG(env, eval_env, params)
    algo.train(render=False, render_eval=False)
    algo.play(render_eval=True)
