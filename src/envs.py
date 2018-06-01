from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from utils import simulate_fourbar, getMotionError, getPathError, normalized_cross_corelation, motion_cross_corelation
import random
import pickle
from scipy.interpolate import BSpline, splprep, splev

class Env:
    def __init__(self, env_id):
        self.env_id = env_id

'''
Create a function that computes a bunch of random valid fourbar parameters and stores them in array. so that we dont have to calculate it each time.
'''

class Fourbar(Env):
    ''' Fourbar environment which tries solve path generation problem
    '''
    def __init__(self, mode='path', state_is='coordinates', compare_all_branches=False):
        super(Fourbar, self).__init__(env_id='Fourbar-'+mode)
        self.mode = mode
        '''
        State is either linkage params (i.e. dim = 5)
        or
        Coupler Curves (which are of variable size)
        '''
        self.state_is = state_is

        ''' consider only first branch of fourbar motion for RL or compare everyone
        '''
        self._compare_all_branches = compare_all_branches

        self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(5,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(245,))

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111)
        self.line1 = self.ax1.plot(np.arange(1), np.arange(1), 'o', label='Task')[0]
        self.line2 = self.ax1.plot(np.arange(1), np.arange(1), '-', label='Achieved')[0]
        self.fig.canvas.draw()

        with open('../dataset.pkl', 'rb') as f:
            self.dataset = pickle.load(f)
            print(len(self.dataset))

        ''' We can do this
        Always start from init params l1 = 1, l2 = 1, l3 = 1, l4 = 0, l5 = 0
        '''
        '''
        Nature of task depends upon mode
        It can be either motion or path signature
        TODO: Add obstacle avoidance as well
        '''
        self._load_init_data()

    def _load_init_data(self):
        fbdataset = FourbarDataset(self._compare_all_branches)
        init_ = fbdataset.compute(num=1, random_params=False)
        self.state = init_[0]['state']
        self.params = init_[0]['params']
        self.coupler_curves  = init_[0]['coupler_curves']

    def _load_random_coupler_curves(self):
        ''' take random params and calculates coupler curves
        '''
        sample = random.sample(self.dataset, 1)[0]
        params = sample['params']
        coupler_curves = sample['coupler_curves']
        state = sample['state']
        return params, coupler_curves, state

    def reset(self):
        '''
        Resets link parameters and returns obervation, reward, done flag and info
        Reseting the task
        Taking part (first 70 points) of a random trajectory as target
        This is because 70 points make non trivial path/motion
        '''
        params, coupler_curves, state = self._load_random_coupler_curves()
        print(len(coupler_curves.signs))
        task_sign_lens = [len(sign[self.mode + '_sign']) for sign in coupler_curves.signs]
        task = coupler_curves.signs[np.argmax(task_sign_lens)]
        self.task = {self.mode + '_sign': task[self.mode + '_sign']}
        self.goal = task['fixed_' + self.mode + '_sign'].flatten()
        self.goal_trajectory = coupler_curves.curv1

        '''
        Reinitializing coupler curves
        Evaluating coupler curves
        '''
        #self._load_init_data()
        self.params, self.coupler_curves, self.state = self._load_random_coupler_curves()
        self.full_state = np.concatenate((self.state, self.params), axis=0)
        self.full_state = np.concatenate((self.full_state, self.goal), axis=0)
        print(self.full_state.shape)
        assert self.full_state.shape == (245,)

        '''
        TODO: goal state should be of fixed dimensions, which currently is not.
        for goal based RL algorithms use commented
        '''
        #return {'observation': self.state, 'achieved_goal':self.achieved_goal, 'desired_goal':self.task[self.mode + '_sign']}
        return self.full_state

    def step(self, action):
        self.params = self.params + action
        self.params[:3] = np.clip(self.params[:3], 0.2, 5)
        self.params[3:5] = np.clip(self.params[3:5], -3, 3)

        self.is_success = False
        self._calculate_state()
        self._evaluate_step()

        self.full_state = np.concatenate((self.state, self.params), axis=0)
        self.full_state = np.concatenate((self.full_state, self.goal), axis=0)
        '''
        TODO: goal state should be of fixed dimensions, which currently is not.
        for goal based RL algorithms use commented
        '''
        #return ({'observation': self.state, 'achieved_goal':self.achieved_goal, 'desired_goal':self.task[self.mode + '_sign']}, self.reward, self.is_success, {'is_success': self.is_success})

        return (self.full_state, self.reward, self.is_success, {'is_success': self.is_success})

    def _calculate_state(self):
        if self._compare_all_branches:
            self.coupler_curves, full_data = simulate_fourbar(self.params, all_joints=True)
        else:
            self.coupler_curves, full_data = simulate_fourbar(self.params, both_branches=False, only_first_curve=True, all_joints=True)

        if len(self.coupler_curves.curv1) > 5:
            success = True
            self.state = np.zeros((7,20))
            state = []
            for data in full_data:
                state.append([
                    data['B0'][0][0], data['B0'][0][1],
                    data['B1'][0][0], data['B1'][0][1],
                    data['C'][0][0], data['C'][0][1],
                    data['theta'][0],
                    ])
            state = np.transpose(np.array(state))
            # B-spline fitting state to creat a fixed length entity
            for i in range(0, state.shape[0], 2):
                if i != state.shape[0] - 1:
                    try:
                        tck, _ = splprep(x=[state[i], state[i+1]], k=3, s=0)
                        u = np.arange(0, 1, 0.05)
                        op = np.array(splev(u, tck))
                        #plt.plot(op[0,:], op[1,:], 'o')
                        self.state[i:i+2, :] = op
                    except ValueError as e:
                        print(str(e))
                else:
                    try:
                        angle = self.coupler_curves.signs[0]['normalized_angle']
                        tck, _ = splprep(x=[np.arange(len(angle)), angle], k=3, s=0)
                        u = np.arange(0, 1, 0.05)
                        op = np.array(splev(u, tck))
                        #plt.plot(state[i],'-')
                        #plt.plot(op[1,:],'o')
                        #print(op.shape)
                        self.state[i, :] = op[1,:]
                    except ValueError as e:
                        print(e)
            self.state = np.reshape(self.state, [7*20])
            return success
        else:
            False

    def _evaluate_step(self):
        ''' Calculates reward for current parameters and task
        '''
        if len(self.coupler_curves.signs) != 0:
            if self.mode == 'path': #and len(self.coupler_curves.signs[0]['path_sign'])*1.3 > len(self.task['path_sign']):
                result = normalized_cross_corelation(self.coupler_curves.signs[0], self.task)
                self.reward = result['score']
                if self.reward > 0.99:
                    self.is_success = True
            elif self.mode == 'motion': # and len(self.coupler_curves.signs[0]['motion_sign'])*1.3 > len(self.task['motion_sign']):
                result = -motion_cross_corelation(self.coupler_curves.signs[0], self.task)
                self.reward = result['distance']

                if self.reward > -0.01:
                    self.is_success = True
        else:
            self.reward = -1

        if len(self.coupler_curves.signs) == 0:
            self.achieved_goal = None
        else:
            self.achieved_goal = self.coupler_curves.signs[0][self.mode + '_sign']

    def render(self):
        ''' Should render fourbar and its coupler curve
        '''
        self.ax1.autoscale_view()
        self.ax1.relim()
        self.line1.set_data(self.goal_trajectory[:,0], self.goal_trajectory[:,1])
        if len(self.coupler_curves.curv1) > 5:
            self.line2.set_data(self.coupler_curves.curv1[:,0], self.coupler_curves.curv1[:,1])

        self.ax1.legend(loc='best')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

class FourbarDataset:
    def __init__(self, compare_all_branches=False):
        self._compare_all_branches = compare_all_branches

    def compute(self, num = 1000, random_params=True):
        ''' take random params and calculates coupler curves
        '''
        # setting random parameters from lognormal distribution (0, 5)
        params = np.zeros((5,))
        self.dataset = []
        for _ in range(num):
            success = False
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
                if not random_params:
                    self.params= np.array([1, 1, 1, 0, 0])
                success = self.calculate_state()
                if success:
                    self.dataset.append({'params': self.params, 'coupler_curves': self.coupler_curves, 'state': self.state})

        return self.dataset

    def calculate_state(self):
        if self._compare_all_branches:
            self.coupler_curves, full_data = simulate_fourbar(self.params, all_joints=True)
        else:
            self.coupler_curves, full_data = simulate_fourbar(self.params, both_branches=False, only_first_curve=True, all_joints=True)

        if len(self.coupler_curves.curv1) > 5:
            success = True
            self.state = np.zeros((7,20))
            state = []
            for data in full_data:
                state.append([
                    data['B0'][0][0], data['B0'][0][1],
                    data['B1'][0][0], data['B1'][0][1],
                    data['C'][0][0], data['C'][0][1],
                    data['theta'][0],
                    ])
            state = np.transpose(np.array(state))
            # B-spline fitting state to creat a fixed length entity
            for i in range(0, state.shape[0], 2):
                if i != state.shape[0] - 1:
                    try:
                        tck, _ = splprep(x=[state[i], state[i+1]], k=3, s=0)
                        u = np.arange(0, 1, 0.05)
                        op = np.array(splev(u, tck))
                        #plt.plot(op[0,:], op[1,:], 'o')
                        self.state[i:i+2, :] = op
                    except ValueError as e:
                        print(str(e))
                else:
                    try:
                        angle = self.coupler_curves.signs[0]['normalized_angle']
                        tck, _ = splprep(x=[np.arange(len(angle)), angle], k=3, s=0)
                        u = np.arange(0, 1, 0.05)
                        op = np.array(splev(u, tck))
                        #plt.plot(state[i],'-')
                        #plt.plot(op[1,:],'o')
                        #print(op.shape)
                        self.state[i, :] = op[1,:]
                    except ValueError as e:
                        print(e)
            self.state = np.reshape(self.state, [7*20])
            return success
        else:
            False

    def save(self):
        with open('../dataset.pkl', 'wb') as f:
            #print(self.dataset)
            pickle.dump(self.dataset, f)

if __name__ == '__main__':
    fb = Fourbar('path')
    state = fb.reset()
    print(state.shape)
    for _ in range(100):
        state, reward, _, _ = fb.step(fb.action_space.sample())
        print(state.shape)
        print(reward)
        fb.render()


