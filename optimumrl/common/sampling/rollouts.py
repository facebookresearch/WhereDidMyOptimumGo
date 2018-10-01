# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

def rollout_episode(env, agent, seed):
    env.seed(seed)
    state = torch.Tensor([env.reset()])
    episode_return = 0
    while True:
        action = agent.select_action(state)

        next_state, reward, done, _ = env.step(action.numpy()[0])
        episode_return += reward

        next_state = torch.Tensor([next_state])

        state = next_state
        if done:
            break

    return episode_return
