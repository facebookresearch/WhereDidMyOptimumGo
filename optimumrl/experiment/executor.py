# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import glob
import os
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from optimumrl.environment.wrappers.general import make_env
from optimumrl.models.ppo_a2c_acktr import Policy
from optimumrl.common.storage.pg_storage import RolloutStorage
from optimumrl.common.visualization.visualize import visdom_plot
from optimumrl.common.random.prng import set_all_seeds
import optimumrl.algorithms as algo


def launch_experiment(config):
    # 1. Create experiment directory
    try:
        os.makedirs(config["log_dir"])
    except OSError:
        files = glob.glob(os.path.join(config["log_dir"], '*.monitor.csv'))
        for f in files:
            os.remove(f)

    # 2. Save config into experiment directory
    with open(os.path.join(config["log_dir"], "configuration.json"), 'w') as outfile:
        json.dump(config, outfile)

    assert config["algorithm"]["name"] in ['a2c', 'ppo', 'acktr']
    if config["algorithm"]["params"]["recurrent_policy"]:
        assert config["algorithm"]["name"] in ['a2c', 'ppo'], \
            'Recurrent policy is not implemented for ACKTR'

    set_all_seeds(config["training_seed"]["agent_seed"])
    torch.set_num_threads(1)

    if config["viz"]:
        from visdom import Visdom
        viz = Visdom(port=config["viz_port"])
        win = None

    envs = [make_env(config["environment"]["name"], config["training_seed"]["environment_seed"], i, config["log_dir"], config["algorithm"]["params"]["add_timestep"])
            for i in range(config["algorithm"]["params"]["num_processes"])]

    num_updates = int(config["algorithm"]["params"]["num_frames"]) // config["algorithm"]["params"]["num_steps"] // config["algorithm"]["params"]["num_processes"]

    # TODO: move most of this to separate per-algo file

    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")





    # TODO: don't allow multiple processes, this is essentially multi-agent as is,
    # because you're not living up to the TD thing so it doesn't really seem comparable
    # Or I guess make this an explicit option labeling it as multi-agent
    if config["algorithm"]["params"]["num_processes"] > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * config["algorithm"]["params"]["num_stack"], *obs_shape[1:])

    actor_critic = Policy(obs_shape, envs.action_space,
                          config["algorithm"]["params"]["recurrent_policy"])

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if config["algorithm"]["params"]["cuda"]:
        actor_critic.cuda()

    # TODO: better instantiation method
    if config["algorithm"]["name"] == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, config["algorithm"]["params"]["value_loss_coef"],
                               config["algorithm"]["params"]["entropy_coef"],
                               max_grad_norm=config["algorithm"]["params"]["max_grad_norm"], optimizer=config["algorithm"]["params"]["optimizer"])
    elif config["algorithm"]["name"] == 'ppo':
        agent = algo.PPO(actor_critic, config["algorithm"]["params"]["clip_param"], config["algorithm"]["params"]["ppo_epoch"], config["algorithm"]["params"]["num_mini_batch"],
                         config["algorithm"]["params"]["value_loss_coef"],config["algorithm"]["params"]["entropy_coef"],
                         max_grad_norm=config["algorithm"]["params"]["max_grad_norm"], optimizer=config["algorithm"]["params"]["optimizer"])
    elif config["algorithm"]["name"] == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, config["algorithm"]["params"]["value_loss_coef"],
                               config["algorithm"]["params"]["entropy_coef"], acktr=True)

    rollouts = RolloutStorage(config["algorithm"]["params"]["num_steps"], config["algorithm"]["params"]["num_processes"],
                              obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(config["algorithm"]["params"]["num_processes"], *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if config["algorithm"]["params"]["num_stack"] > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([config["algorithm"]["params"]["num_processes"], 1])
    final_rewards = torch.zeros([config["algorithm"]["params"]["num_processes"], 1])

    if config["algorithm"]["params"]["cuda"]:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(config["algorithm"]["params"]["num_steps"]):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states = actor_critic.act(
                    rollouts.observations[step],
                    rollouts.states[step],
                    rollouts.masks[step])
            cpu_actions = action.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(
                np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if config["algorithm"]["params"]["cuda"]:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)
            rollouts.insert(current_obs, states, action,
                            action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1],
                                                rollouts.states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(
            next_value, config["algorithm"]["params"]["use_gae"], config["algorithm"]["params"]["gamma"], config["algorithm"]["params"]["tau"])

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % config["algorithm"]["params"]["save_interval"]== 0 and config["algorithm"]["params"]["save_model"]:
            save_path = os.path.join(config["algorithm"]["log_dir"], "trained_models")
            save_path = os.path.join(save_path, config["algorithm"]["name"])
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if config["algorithm"]["params"]["cuda"]:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, config["environment"]["name"] + ".pt"))

        if j % config["algorithm"]["params"]["log_interval"] == 0:
            end = time.time()
            total_num_steps = (j + 1) * config["algorithm"]["params"]["num_processes"] * config["algorithm"]["params"]["num_steps"]
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                  format(j, total_num_steps,
                         int(total_num_steps / (end - start)),
                         final_rewards.mean(),
                         final_rewards.median(),
                         final_rewards.min(),
                         final_rewards.max(), dist_entropy,
                         value_loss, action_loss))
        if config["viz"] and j % config["viz_interval"] == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, config["log_dir"], config["environment"]["name"],
                                  config["algorithm"]["name"], config["algorithm"]["params"]["num_frames"])
            except IOError:
                pass
