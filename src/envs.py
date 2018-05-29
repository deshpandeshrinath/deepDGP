from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from utils import simulate_fourbar, getMotionError, getPathError
import random

class Env:
    def __init__(self, env_id):
        self.env_id = env_id

class Fourbar(Env):
    ''' Fourbar environment which tries solve path generation problem
    '''
    def __init__(self, mode='path', state_is='params'):
        super(Fourbar, self).__init__(env_id='Fourbar-'+mode)
        self.mode = mode
        '''
        State is either linkage params (i.e. dim = 5)
        or
        Coupler Curves (which are of variable size)
        '''
        self.state_is = state_is

        self.action_space = spaces.Box(low=-1, high=1, shape=(5,))
        self.observation_space = spaces.Box(low=0.2, high=3, shape=(5,))

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        self.line1 = self.ax1.plot(np.arange(1), np.arange(1), label='Task')[0]
        self.line2 = self.ax1.plot(np.arange(1), np.arange(1), label='Achieved')[0]
        self.fig.canvas.draw()

        '''
        Nature of task depends upon mode
        It can be either motion or path signature
        TODO: Add obstacle avoidance as well
        '''
    def _compute_random_coupler_curves(self):
        ''' take random params and calculates coupler curves
        '''
        success = False
        # setting random parameters from lognormal distribution (0, 5)
        params = np.zeros((5,))
        while not success:
            for i in range(3):
                redo = True
                while redo:
                    a = np.random.lognormal(0, 2, 1)
                    # making sure that we get ration less than 5 and greater than 0.2
                    if a < 5 and a > 0.2:
                        redo = False
                        params[i] = a
            for i in range(3, 5):
                params[i] = np.random.normal(0, 2, 1)
            self.params = params
            self.coupler_curves = simulate_fourbar(self.params)

            for curve in self.coupler_curves.curves:
                if len(curve) > 5:
                    success = True
                    break

    def reset(self):
        '''
        Resets link parameters and returns obervation, reward, done flag and info
        Reseting the task
        Taking part (first 70 points) of a random trajectory as target
        This is because 70 points make non trivial path/motion
        '''
        redo = True
        while redo:
            self._compute_random_coupler_curves()
            task_sign_lens = [len(sign[self.mode + '_sign']) for sign in self.coupler_curves.signs]
            task = self.coupler_curves.signs[np.argmax(task_sign_lens)][self.mode + '_sign']
            if len(task) >= 70:
                self.task = {self.mode + '_sign': task[:70]}
                redo = False

        '''
        Evaluating coupler curves
        '''
        self._evaluate_step()
        if self.state_is == 'params':
            self.state = self.params

        '''
        TODO: goal state should be of fixed dimensions, which currently is not.
        for goal based RL algorithms use commented
        '''
        #return {'observation': self.state, 'achieved_goal':self.achieved_goal, 'desired_goal':self.task[self.mode + '_sign']}
        return self.state

    def step(self, action):
        self.params = self.params + action
        self.params[:3] = np.clip(self.params[:3], 0.2, 5)
        self.params[3:5] = np.clip(self.params[3:5], -3, 3)
        if self.state_is == 'params':
            self.state = self.params

        self.is_success = False
        self._evaluate_step()
        '''
        TODO: goal state should be of fixed dimensions, which currently is not.
        for goal based RL algorithms use commented
        '''
        #return ({'observation': self.state, 'achieved_goal':self.achieved_goal, 'desired_goal':self.task[self.mode + '_sign']}, self.reward, self.is_success, {'is_success': self.is_success})

        return (self.state, self.reward, self.is_success, {'is_success': self.is_success})


    def _evaluate_step(self):
        ''' Calculates coupler curves, reward for current parameters and task
        '''
        if self.mode == 'path':
            self.reward, index, self.coupler_curves = getPathError(self.params, self.task)
            if self.reward > -0.005:
                self.is_success = True
        elif self.mode == 'motion':
            self.reward, index, self.coupler_curves = getMotionError(self.params, self.task)
            if self.reward > -0.01:
                self.is_success = True

        self.reward = -self.reward

        if len(self.coupler_curves.signs) == 0:
            self.achieved_goal = None
        else:
            self.achieved_goal = self.coupler_curves.signs[index][self.mode + '_sign']

    def render(self):
        self.ax1.autoscale_view()
        self.ax1.relim()
        self.line1.set_data(np.arange(len(self.task[self.mode+'_sign'])), self.task[self.mode+'_sign'])
        if self.achieved_goal is not None:
            self.line2.set_data(np.arange(len(self.achieved_goal)), self.achieved_goal)
        #for sign in self.coupler_curves.signs:
        #    ax1.plot(sign['motion_sign'])
        self.ax1.legend(loc='best')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.03)


if __name__ == '__main__':
    fb = Fourbar('path')
    fb.reset()
    for _ in range(50):
        fb.step(fb.action_space.sample())
        fb.render()

