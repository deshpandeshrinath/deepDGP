import numpy as np
import sys
import os

files = []
statsa = []
labels = []

for i in range(1, len(sys.argv)):
    files.append(sys.argv[i])
    statsa.append(np.load(files[i-1]))
    name = os.path.split(files[i-1])[0]
    labels.append(os.path.split(name)[1])

#statsa.append(np.load('./model_Ant-v2/train_stats.npy'))
#statsa.append(np.load('./model_HalfCheetah-v2/train_stats.npy'))
#statsa.append(np.load('./model_InvertedDoublePendulum-v2/train_stats.npy'))
#statsa.append(np.load('./model_Reacher-v2/train_stats.npy'))

#labels = ['Ant', 'Half-Cheetah', 'Inverted Double Pendulum', 'Reacher']
#colors = ['red', 'blue', 'magenta', 'green']
from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 8}
matplotlib.rc('font', **font)

def trendline(y):
    x = np.arange(len(y))
    p = np.poly1d(np.polyfit(x, y, 3))
    z = [p(xx) for xx in x]
    return x, z

fig = plt.figure()
ax = fig.add_subplot(211)
ax.set_title("Actor Loss")
ax.set_xlabel("Train Steps")
ax2 = fig.add_subplot(212)
ax2.set_title("Critic Loss")
ax2.set_xlabel("Train Steps")

fig2 = plt.figure()
ax3 = fig2.add_subplot(111)
ax3.set_title("Rewards per Episode")
ax3.set_xlabel("Episodes")

i = 0
for stats in statsa:
    #stats[2] = stats[2] + np.abs(np.min(stats[2]))
    actor_losses = stats[0]
    critic_losses = stats[1]
    rewards = stats[2]#/np.max(stats[2])
    steps = stats[3]
    ax.plot(np.arange(len(actor_losses)), np.array(actor_losses), label=labels[i])
    ax2.plot(np.arange(len(critic_losses)), np.array(critic_losses), label=labels[i])

    x,y = trendline(rewards)
    ax3.plot(np.arange(len(rewards)), np.array(rewards), alpha=0.3)
    ax3.plot(x, y, label=labels[i])
    i += 1


ax.legend(loc='best')
ax2.legend(loc='best')
ax3.legend(loc='best')
plt.show()
fig.savefig("train_errors.png")
fig2.savefig("rewards.png")

