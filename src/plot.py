import numpy as np
import os
from train import parseArgs
FLAGS = parseArgs()
model_dir = FLAGS.model_dir

from matplotlib import pyplot as plt
import matplotlib
font = {'size'   : 8}
matplotlib.rc('font', **font)


fig = plt.figure()
ax = fig.add_subplot(211)
ax.set_title("Actor Loss")
ax.set_xlabel("Train Steps")
ax2 = fig.add_subplot(212)
ax2.set_title("Critic Loss")
ax2.set_xlabel("Train Steps")

filename = os.path.join(model_dir, "train_stats.npy")
stats = np.load(filename)

#stats[2] = stats[2] + np.abs(np.min(stats[2]))
total_actor_losses = stats[0]
total_critic_losses = stats[1]
rewards = stats[2]#/np.max(stats[2])
steps = stats[3]

ax.plot(np.arange(len(total_actor_losses)), np.array(total_actor_losses))
ax2.plot(np.arange(len(total_critic_losses)), np.array(total_critic_losses))


fig2 = plt.figure()
ax3 = fig2.add_subplot(111)
ax3.set_title("Rewards per Episode")
ax3.set_xlabel("Episodes")
ax3.plot(np.arange(len(rewards)), np.array(rewards))

plt.show()

fig.savefig(os.path.join(model_dir, "train_errors.png"))
fig2.savefig(os.path.join(model_dir, "rewards.png"))
