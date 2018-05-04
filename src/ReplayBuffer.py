import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        if self.num_experiences < batch_size:
            batch = random.sample(self.buffer, self.num_experiences)
            current_batch_size = self.num_experiences
        else:
            batch = random.sample(self.buffer, batch_size)
            current_batch_size = batch_size

        replay_batch = dict()
        for key in batch[0].keys():
            replay_batch[key]=[]

        for e in batch:
            for key in replay_batch.keys():
                replay_batch[key].append(e[key])

        for key in replay_batch.keys():
            replay_batch[key] = np.array(replay_batch[key], dtype=np.float32)
            replay_batch[key] = np.reshape(replay_batch[key], [current_batch_size, -1])

        return replay_batch

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done, goal=None):
        if goal is not None:
            self.has_goal_state = True
            experience = {'current_state':state,
                    'current_action':action,
                    'current_reward': reward,
                    'next_state':new_state,
                    'done':done,
                    'goal':goal
                    }
        else:
            self.has_goal_state = False
            experience = {'current_state':state,
                    'current_action':action,
                    'current_reward': reward,
                    'next_state':new_state,
                    'done':done,
                    }
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
