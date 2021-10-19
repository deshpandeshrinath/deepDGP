import tensorflow.compat.v1 as tf
import tf_slim as tc
import os
from copy import copy
import time
import numpy as np
from tqdm import tqdm
from functools import reduce
import pickle

from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 8}
matplotlib.rc('font', **font)

''' Our Files
'''
from models import Actor, Critic
from noise import OU_ActionNoise
from ReplayBuffer import ReplayBuffer

class DDPG:
    def __init__(self, env, eval_env=None, params={'tau':0.001, 'gamma':0.99, 'lr_act':0.0001, 'lr_crit':0.001, 'batch_size':32, 'buffer_size': 1000000, 'num_epochs' : 500, 'num_cycles' : 20, 'num_rollouts' : 100, 'train_steps' : 50, 'model_dir' : "./model", 'stddev': 0.3, 'hidden_size':64, 'critic_l2_reg': 0.0}, render_graphs=False):
        tf.disable_v2_behavior() 
        self.env = env
        self.eval_env = eval_env
        self.tau = params['tau']
        self.gamma = params['gamma']
        self.batch_size = params['batch_size']
        self.lr_act = params['lr_act']
        self.lr_crit = params['lr_crit']
        self.num_rollouts = params['num_rollouts']
        self.num_epochs = params['num_epochs']
        self.num_cycles = params['num_cycles']
        self.train_steps = params['train_steps']
        self.model_dir = params['model_dir']
        # Std Deviation of noise
        stddev = params['stddev']
        self.render_graphs = render_graphs
        """ Actor and Critic (along with their target copies)
        """
        self.actor = Actor(self.env.action_space.shape[-1], hidden_units=params['hidden_size'])
        self.actor_target = Actor(self.env.action_space.shape[-1], name='actor_target', hidden_units=params['hidden_size'])
        self.actor_target.name = 'act_target'

        self.critic = Critic(hidden_units=params['hidden_size'])
        self.critic_target = Critic(name='crit_target', hidden_units=params['hidden_size'])
        self.critic_l2_reg = params['critic_l2_reg']
        """ ReplayBuffer where we store the experiences
        """
        if os.path.isfile(os.path.join(self.model_dir, "buffer.pkl")):
            print('Loading buffer from model...')
            with open(os.path.join(self.model_dir, "buffer.pkl"), 'rb') as f:
                self.buffer = pickle.load(f)
                f.close()
        else:
            self.buffer = ReplayBuffer(params['buffer_size'])
        self.num_actions = self.env.action_space.shape[-1]
        self.action_noise = OU_ActionNoise(mu=np.zeros(self.num_actions), sigma=float(stddev) * np.ones(self.num_actions))

        # For some task goal state is given in observations
        if self.env.observation_space.shape is None:
            self.state_size = self.env.reset()['observation'].shape[-1]
            self.goal_size = self.env.reset()['desired_goal'].shape[-1]
            self.has_goal_state = True
        else:
            self.state_size = self.env.observation_space.shape[-1]
            self.has_goal_state = False
            self.goal_size = 0

        ''' Inputs to DDPG from experiences
            self.current_state
            self.next_state
            self.goal_
            self.current_reward
            self.current_action
            self.yt_const : to be loaded from yt, so as to keep it as constant in grad computation
        '''
        self.current_state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size], name='current_state')
        self.goal_state = tf.placeholder(dtype=tf.float32, shape=[None, self.goal_size], name='goal_state')
        self.next_state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_size], name='next_state')

        # if goal exists concat it with current state to form input for prediction
        if self.goal_size == 0:
            self.input_state =  self.current_state
            self.next_input_state =  self.next_state
        else:
            self.input_state = tf.concat((self.current_state, self.goal_state), axis=1)
            self.next_input_state = tf.concat((self.next_state, self.goal_state), axis=1)


        self.current_reward = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self.current_action = tf.placeholder(tf.float32, shape=(None, self.env.action_space.shape[-1]), name='rewards')
        self.yt_const = tf.placeholder(tf.float32, shape=(None, 1), name='place_for_target_Q')

        # Actor's action prediction and critic's Q value prediction on predicted action
        self.predicted_current_action = self.actor.predict(self.input_state)

        self.predicted_Q_on_predicted_current_action = self.critic.predict(self.input_state, self.predicted_current_action)

        # Target Q value
        # Q (s_t,a_t) : Here a_t is not variable but taken from placeholder to keep away actor variables
        self.target_Q = self.critic.predict(self.input_state, self.current_action, reuse=True)

        #Q value according to bellman equation
        self.predicted_next_action = self.actor_target.predict(self.next_input_state)
        # Q (s_(t+1), a_(t+1)
        self.predicted_next_Q = self.critic_target.predict(self.next_input_state, self.predicted_next_action)
        # yt from paper
        self.yt = self.current_reward + self.gamma * self.predicted_next_Q

        # Setting up pipline for optimizers
        self.optimizer_setup()

        # saving
        self.saver = tf.train.Saver(max_to_keep=3)

    def optimizer_setup(self):
        """ optimizing ops
        1. self.critic_train_op
        2. self.actor_train_op
        3. self.target_init_updates
        4. self.target_soft_updates
        """
        # training op for critic
        print('setting up critic optimizer')
        critic_shapes = [var.get_shape().as_list() for var in self.critic.trainable_vars]
        critic_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in critic_shapes])
        print('  critic shapes: {}'.format(critic_shapes))
        print('  critic params: {}'.format(critic_nb_params))
        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.yt_const))
        if self.critic_l2_reg > 0.0:
            critic_reg_vars = [var for var in self.critic.trainable_vars if 'kernel' in var.name and 'output' not in var.name]
            for var in critic_reg_vars:
                print('  regularizing: {}'.format(var.name))
            print('  applying l2 regularization with {}'.format(self.critic_l2_reg))
            critic_reg = tc.apply_regularization(
                tc.l2_regularizer(self.critic_l2_reg),
                weights_list=critic_reg_vars
            )
            self.critic_loss += critic_reg

        self.critic_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_crit).minimize(self.critic_loss)
        # training op for actor
        # action_grads = dQ/da
        action_grads = tf.gradients(self.predicted_Q_on_predicted_current_action, self.predicted_current_action)
        ''' action_grads is the list of dim 1. so taking first element to get underlying Tensor
        '''
        ### policy_gradient delta J = d(mu)/d(theta_u) * dQ/da
        ##policy_gradients = tf.gradients(self.predicted_current_action, self.actor.trainable_vars, -action_grads[0])
        ##actor_grads_and_vars = zip(policy_gradients, self.actor.trainable_vars)
        ##self.actor_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_act).apply_gradients(actor_grads_and_vars)

        print('setting up actor optimizer')
        self.actor_loss = -tf.reduce_mean(self.predicted_Q_on_predicted_current_action)
        actor_shapes = [var.get_shape().as_list() for var in self.actor.trainable_vars]
        actor_nb_params = sum([reduce(lambda x, y: x * y, shape) for shape in actor_shapes])
        print('  actor shapes: {}'.format(actor_shapes))
        print('  actor params: {}'.format(actor_nb_params))
        self.actor_grads = tf.gradients(self.actor_loss, self.actor.trainable_vars)
        actor_grads_and_vars = zip(self.actor_grads, self.actor.trainable_vars)
        self.actor_train_op = tf.train.AdamOptimizer(learning_rate=self.lr_act).apply_gradients(actor_grads_and_vars)
        """ setting Pipeline for Updating the target networks
        """
        actor_init_updates, actor_soft_updates = self.get_target_updates(self.actor.trainable_vars, self.actor_target.trainable_vars)
        critic_init_updates, critic_soft_updates = self.get_target_updates(self.critic.trainable_vars, self.critic_target.trainable_vars)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def get_target_updates(self, vars, target_vars):
        print('setting up target updates ...')
        soft_updates = []
        init_updates = []
        assert len(vars) == len(target_vars)
        for var, target_var in zip(vars, target_vars):
            print('  {} <- {}'.format(target_var.name, var.name))
            init_updates.append(tf.assign(target_var, var))
            soft_updates.append(tf.assign(target_var, (1. - self.tau) * target_var + self.tau * var))
        assert len(init_updates) == len(vars)
        assert len(soft_updates) == len(vars)
        return tf.group(*init_updates), tf.group(*soft_updates)

        """ Now updating the target networks' weights
        """

    def train_from_batch(self, sess):
        """ Trianing includes taking a batch from replay and computing gradients, updating weights
        """
        replay_batch = self.buffer.getBatch(self.batch_size)
        yt = sess.run(self.yt, feed_dict={
            self.next_state: replay_batch['next_state'],
            self.current_reward: replay_batch['current_reward']
            })

        ops = [self.critic_train_op, self.actor_train_op, self.target_soft_updates]

        _, critic_loss, actor_loss = sess.run((ops, self.critic_loss, self.actor_loss), feed_dict={
            self.current_state: replay_batch['current_state'],
            self.next_state: replay_batch['next_state'],
            self.current_reward: replay_batch['current_reward'],
            self.current_action: replay_batch['current_action'],
            self.yt_const: yt
        })
        return critic_loss, actor_loss

    def train(self, render=False, render_eval=False):
        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        epoch = 0
        start_time = time.time()
        total_actor_losses = []
        total_critic_losses = []
        epoch_episode_rewards = []
        epoch_episode_steps = []
        #epoch_episode_eval_rewards = []
        #epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0
        max_action = self.env.action_space.high
        if self.render_graphs:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(211)
            ax.set_title("Actor Loss")
            ax2 = fig.add_subplot(212)
            ax2.set_title("Critic Loss")
            line = ax.plot(np.zeros(1), np.zeros(1), 'b-')[0]
            line2 = ax2.plot(np.zeros(1), np.zeros(1), 'r-')[0]
            fig.canvas.draw()

        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        with tf.Session() as sess:
            try:
                print('Restoring from checkpoint...')
                self.saver.restore(sess, os.path.join(self.model_dir, "model.ckpt"))
                sess.graph.finalize()
            except:
                print('No checkpoints found')
                sess.run(tf.global_variables_initializer())
                sess.graph.finalize()
            sess.run(self.target_init_updates)
            obs = self.env.reset()
            obs_eval = self.eval_env.reset()
            for i in tqdm(range(self.num_epochs), total = self.num_epochs):
                for j in range(self.num_cycles):
                    # Start making some random steps
                    for t in range(self.num_rollouts):
                        # Predict next action
                        if obs is dict:
                            current_state = np.reshape(obs['observation'], [1,-1])
                            goal_state = np.reshape(obs['desired_goal'],[1,-1])
                            feed_dict = {
                            self.current_state: current_state,
                            self.goal_state: goal_state
                                    }
                        else:
                            current_state = np.reshape(obs, [1,-1])
                            goal_state = None
                            feed_dict = {
                            self.current_state: current_state,
                                    }
                        current_action, Q_value = sess.run((self.predicted_current_action, self.predicted_Q_on_predicted_current_action), feed_dict=feed_dict)
                        current_action = current_action[0]
                        Q_value = Q_value[0]
                        # Add noise
                        noise = self.action_noise()
                        assert(noise.shape == current_action.shape)
                        current_action += noise
                        current_action = np.clip(current_action, self.env.action_space.low, self.env.action_space.high)
                        #

                        assert(current_action.shape == self.env.action_space.shape)

                        obs, current_reward, done, info = self.env.step(current_action*max_action)
                        if obs is dict:
                            next_state = np.reshape(obs['observation'], [1,-1])
                            #next_goal_state = np.reshape(obs['desired_goal'],[1,-1])
                        else:
                            next_state = np.reshape(obs, [1,-1])
                            #next_goal_state = None
                        t += 1
                        if render:
                            self.env.render()
                        episode_reward += current_reward
                        episode_step += 1

                        # Book-Keeping
                        epoch_actions.append(current_action)
                        epoch_qs.append(Q_value)
                        self.buffer.add(current_state, current_action, current_reward, next_state, done, goal=goal_state)

                        if done:
                            # Episode done.
                            epoch_episode_rewards.append(episode_reward)
                            epoch_episode_steps.append(episode_step)
                            episode_reward = 0.
                            episode_step = 0
                            epoch_episodes += 1

                            obs = self.env.reset()
                            self.action_noise.reset()
                    # Train
                    epoch_actor_losses = []
                    epoch_critic_losses = []
                    #epoch_adaptive_distances = []
                    for step in range(self.train_steps):
                        critic_loss, actor_loss = self.train_from_batch(sess)
                        epoch_actor_losses.append(actor_loss)
                        epoch_critic_losses.append(critic_loss)
                    total_actor_losses.append(np.mean(epoch_actor_losses))
                    total_critic_losses.append(np.mean(epoch_critic_losses))

                    # Evaluate
                    if self.eval_env is not None:
                        eval_episode_rewards = []
                        eval_qs = []
                        if self.eval_env is not None:
                            eval_episode_reward = 0.
                            for t_eval in range(self.num_rollouts):
                                if obs_eval is dict:
                                    current_state = np.reshape(obs_eval['observation'], [1,-1])
                                    goal_state = np.reshape(obs_eval['desired_goal'],[1,-1])
                                    feed_dict = {
                                    self.current_state: current_state,
                                    self.goal_state: goal_state
                                            }
                                else:
                                    current_state = np.reshape(obs_eval, [1,-1])
                                    goal_state = None
                                    feed_dict = {
                                    self.current_state: current_state,
                                            }
                                eval_action, eval_q = sess.run((self.predicted_current_action, self.predicted_Q_on_predicted_current_action), feed_dict=feed_dict)
                                eval_action = eval_action[0]
                                obs_eval, eval_r, eval_done, eval_info = self.eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                                if render_eval:
                                    self.eval_env.render()
                                eval_episode_reward += eval_r

                                eval_qs.append(eval_q)
                                if eval_done:
                                    obs_eval = self.eval_env.reset()
                                    eval_episode_rewards.append(eval_episode_reward)
                                    eval_episode_reward = 0.
                self.saver.save(sess, os.path.join(self.model_dir, "model.ckpt"))

                if self.render_graphs:
                    ax.autoscale_view()
                    ax2.autoscale_view()
                    ax.relim()
                    ax2.relim()
                    line.set_data(np.arange(len(total_actor_losses)), np.array(total_actor_losses))
                    line2.set_data(np.arange(len(total_critic_losses)), np.array(total_critic_losses))
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(1e-7)

            duration = time.time() - start_time
            if self.render_graphs:
                plt.savefig(os.path.join(self.model_dir, "plots.png"))
            train_stats = np.array([total_actor_losses, total_critic_losses, epoch_episode_rewards, epoch_episode_steps])
            filename = os.path.join(self.model_dir, "train_stats.npy")
            if os.path.isfile(filename):
                old_stats = np.load(filename)
                stats = [np.concatenate((old_stat, new_stat), axis=0) for new_stat, old_stat in zip(train_stats, old_stats)]
                np.save(os.path.join(self.model_dir, "train_stats.npy"), stats)
            else:
                np.save(os.path.join(self.model_dir, "train_stats.npy"), train_stats)

            with open(os.path.join(self.model_dir, "buffer.pkl"), 'wb') as f:
                pickle.dump(self.buffer, f)
                f.close()
            print(duration)
            print(np.mean(epoch_episode_rewards))

    def play(self, render_eval=True):
        #tf.reset_default_graph()
        #saver = tf.train.import_meta_graph(os.path.join(self.model_dir, "model.meta"))
        max_action = self.eval_env.action_space.high
        with tf.Session() as sess:
            self.saver.restore(sess, os.path.join(self.model_dir, "model.ckpt"))
            obs_eval = self.eval_env.reset()
            if self.eval_env is not None:
                eval_episode_rewards = []
                eval_qs = []
                if self.eval_env is not None:
                    eval_episode_reward = 0.
                    # Action Loop
                    while True:
                        if obs_eval is dict:
                            current_state = np.reshape(obs_eval['observation'], [1,-1])
                            goal_state = np.reshape(obs_eval['desired_goal'],[1,-1])
                            feed_dict = {
                            self.current_state: current_state,
                            self.goal_state: goal_state
                                    }
                        else:
                            current_state = np.reshape(obs_eval, [1,-1])
                            goal_state = None
                            feed_dict = {
                            self.current_state: current_state,
                                    }

                        eval_action, eval_q = sess.run((self.predicted_current_action, self.predicted_Q_on_predicted_current_action), feed_dict=feed_dict)
                        eval_action = eval_action[0]
                        obs_eval, eval_r, eval_done, eval_info = self.eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            self.eval_env.render()
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        if eval_done:
                            eval_episode_rewards.append(eval_episode_reward)
                            obs_eval = self.eval_env.reset()

