import tensorflow.compat.v1 as tf
import tf_slim as tc
import numpy as np

class Model:
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Actor(Model):
    """Actor model takes states as the input and returns actions
    This is used for systems with actions in continuous domain.
    """
    def __init__(self, num_actions, hidden_units=64, num_layers=2, name='actor', layer_norm=True):
        """ setting placeholders and output pipeline
        """
        super(Actor, self).__init__(name=name)

        self.num_actions =  num_actions
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.layer_norm = layer_norm

    def predict(self, state, reuse=False):
        """ returns actions for given state
        """
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            hidden = state
            for _ in range(self.num_layers):
                hidden = tf.layers.dense(hidden, self.hidden_units)
                if self.layer_norm:
                    hidden = tc.layers.layer_norm(hidden, center=True, scale=True)
                hidden = tf.nn.relu(hidden)

            hidden = tf.layers.dense(hidden, self.num_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            actions = tf.nn.tanh(hidden)
        return actions

class Critic(Model):
    """Critic takes states, actions and (goals if exists) as the input and returns estimated Q(s,a| paramss) value.
    """
    def __init__(self, hidden_units=64, num_layers=2, name='critic', layer_norm=True):
        """ setting placeholders and output pipeline
        """
        super(Critic, self).__init__(name=name)

        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.layer_norm = layer_norm

    def predict(self, state, action, reuse=False):
        """ returns actions for given state
        """
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            hidden = state
            for i in range(self.num_layers):
                if i == (self.num_layers-1):
                    hidden = tf.concat([hidden, action], axis=-1)
                hidden = tf.layers.dense(hidden, self.hidden_units)
                if self.layer_norm:
                    hidden = tc.layers.layer_norm(hidden, center=True, scale=True)
                hidden = tf.nn.relu(hidden)
                # added action just before the final layer

            Q = tf.layers.dense(hidden, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return Q


