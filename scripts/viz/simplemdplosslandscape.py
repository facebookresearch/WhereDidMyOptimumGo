# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
MAX_STEPS = 100
torch.manual_seed(0)
class TinyMDP(object):

    def reset(self):
        self.state = 0
        return self.state

    def step(self):
        # no matter the action we will get further on
        self.state += 1
        if self.state >= MAX_STEPS:
            done = True
        else:
            done = False

        if self.state > 20 and self.state < 30:
            reward = -1
        else:
            reward = 1
        return self.state, reward, done,  {}


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

class MLPBase(nn.Module):
    def __init__(self, num_inputs=1):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, 5)),
            nn.Tanh(),
            init_(nn.Linear(5, 5)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(5, 1))

    @property
    def state_size(self):
        return 1

    def forward(self, inputs):
        hidden_critic = self.critic(inputs)

        return self.critic_linear(hidden_critic)



critic = MLPBase()
mdp = TinyMDP()
state = mdp.reset()
gamma = .7
optimizer = torch.optim.Adam(critic.parameters())
USE_LOSSES=  False
def loss_landscape(approx):
    ls = []
    ys = []
    for i in range(MAX_STEPS):
        result = approx(torch.FloatTensor(np.array([float(i) / MAX_STEPS])))
        true = result + gamma * approx(torch.FloatTensor(np.array([float(i+1) / MAX_STEPS])))
        ys.append(true.data.numpy()[0])
        l = (result - true).pow(2).data.numpy()
        ls.append(l)
    return np.arange((MAX_STEPS)), ys, np.array([l[0] for l in ls])

fig, ax = plt.subplots()
if USE_LOSSES:
    ax.set(ylim=(0,10.0))
else:
    ax.set(ylim=(0,2.0))

xs, ys, losses = loss_landscape(critic)

if USE_LOSSES:
    line, = ax.plot(xs, losses)
else:
    line, = ax.plot(xs, ys)
all_xs, all_ys, lossess = [xs], [ys], [losses]

def animate(i):
    if USE_LOSSES:
        line.set_ydata(lossess[i])  # update the data
    else:
        line.set_ydata(all_ys[i])  # update the data

    return line,

for i in tqdm(range(1000)):
    next_state, reward, done, _ = mdp.step()
    prediction = critic.forward(torch.FloatTensor(np.array([float(state) / MAX_STEPS])))
    prediction_next = critic.forward(torch.FloatTensor(np.array([float(next_state) / MAX_STEPS])))
    val =  reward + gamma * (1 - int(done)) * Variable(prediction_next.data)

    loss = (prediction - val).pow(2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    xs, ys, losses = loss_landscape(critic)
    all_xs.append(xs)
    all_ys.append(ys)
    lossess.append(losses)

    if done:
        next_state = mdp.reset()

Writer = animation.writers['ffmpeg']
writer = Writer(fps=14, metadata=dict(artist='Me'), bitrate=1800)


ani = animation.FuncAnimation(fig, animate, np.arange(1, 1000), interval=25)
ani.save('im.mp4', writer=writer)
# plt.show()
