from PIL import Image
from collections import deque
from datetime import datetime
from pathlib import Path
import copy
import cv2
import imageio
import numpy as np
import random, os
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
#import multiprocessing as mp
from torchvision import transforms as T
import gc

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from src.environment import *
from src.memory import *
from src.model import *
from src.running_mean_std import *

class Agent():
    def __init__(self, envs, world, stage, action_type, num_envs, state_dim, action_dim, save_dir, save_model_step,
                 save_figure_step, learn_step, total_step_or_episode, total_step, total_episode, model,
                 random_encoder_model, gamma, gamma_int, learning_rate, entropy_coef, V_coef, max_grad_norm,
                 clip_param, batch_size, num_epoch, is_normalize_advantage, V_loss_type, target_kl, gae_lambda, int_adv_coef,
                 ext_adv_coef, additional_bonus_state_8_4_option, embedding_dim,
                 k, average_entropy, capacity, replay_buffer, norm_int_type, device):
        self.world = world
        self.stage = stage
        self.action_type = action_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.learn_step = learn_step
        self.total_step_or_episode = total_step_or_episode
        self.total_step = total_step
        self.total_episode = total_episode

        self.current_step = 0
        self.current_episode = 0

        self.save_model_step = save_model_step
        self.save_figure_step = save_figure_step

        self.device = device
        self.save_dir = save_dir

        self.num_envs = num_envs
        self.envs = envs
        self.model = model.to(self.device)
        self.random_encoder_model = random_encoder_model.to(self.device)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gamma_int = gamma_int
        self.entropy_coef = entropy_coef
        self.V_coef = V_coef
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.max_grad_norm = max_grad_norm
        self.clip_param = clip_param
        self.batch_size = batch_size
        self.num_epoch = num_epoch

        self.memory = Memory(self.num_envs)
        self.replay_buffer = replay_buffer
        self.is_completed = False

        self.env = None
        self.max_test_score = -1e9
        self.is_normalize_advantage = is_normalize_advantage
        self.V_loss_type = V_loss_type
        self.target_kl = target_kl
        self.gae_lambda = gae_lambda
        self.int_adv_coef = int_adv_coef
        self.ext_adv_coef = ext_adv_coef
        self.additional_bonus_state_8_4_option = additional_bonus_state_8_4_option
        self.embedding_dim = embedding_dim
        self.k = k
        self.average_entropy = average_entropy
        self.capacity = capacity
        self.norm_int_type = norm_int_type

        # I just log 1000 lastest update and print it to log.
        self.V_loss = np.zeros((1000,)).reshape(-1)
        self.V_int_loss = np.zeros((1000, )).reshape(-1)
        self.P_loss = np.zeros((1000,)).reshape(-1)
        self.E_loss = np.zeros((1000,)).reshape(-1)
        self.approx_kl_divs = np.zeros((1000,)).reshape(-1)
        self.total_loss = np.zeros((1000,)).reshape(-1)
        self.loss_index = 0
        self.len_loss = 0

        self.rewards_int = np.zeros((1000,)).reshape(-1)
        self.rewards_int_norm = np.zeros((1000,)).reshape(-1)
        self.rewards_int_idx = 0
        self.rewards_int_len = 0

        # self.obs_rms = RunningMeanStd(shape=(1, 1, 84, 84), device=self.device)
        self.int_rms = RunningMeanStd(shape=(1,))

    def save_figure(self, is_training = False):
        # test current model and save model/figure if model yield best total rewards.
        # create env for testing, reset test env
        if self.env is None:
            self.env = create_env(self.world, self.stage, self.action_type, self.additional_bonus_state_8_4_option, True)
        state = self.env.reset()
        done = False

        images = []
        total_reward = 0
        total_step = 0
        num_repeat_action = 0
        old_action = -1

        episode_time = datetime.now()

        # play 1 episode, just get loop action with max probability from model until the episode end.
        while not done:
            with torch.no_grad():
                logit, value, value_in = self.model(torch.tensor(np.array(state), dtype = torch.float, device = self.device).unsqueeze(0))
            action = logit.argmax(-1).item()
            next_state, reward, done, trunc, info = self.env.step(action)
            state = next_state
            img = Image.fromarray(self.env.current_state)
            images.append(img)
            total_reward += reward
            total_step += 1

            if action == old_action:
                num_repeat_action += 1
            else:
                num_repeat_action = 0
            old_action = action
            if num_repeat_action == 200:
                break

        #logging, if model yield better result, save figure (test_episode.mp4) and model (best_model.pth)
        if is_training:
            f_out = open(f"logging_test.txt", "a")
            f_out.write(f'episode_reward: {total_reward:.4f} episode_step: {total_step} current_step: {self.current_step} \
loss_p: {(self.P_loss.sum()/self.len_loss):.4f} loss_v: {(self.V_loss.sum()/self.len_loss):.4f} \
loss_v_int: {(self.V_int_loss.sum()/self.len_loss):.4f} loss_e: {(self.E_loss.sum()/self.len_loss):.4f} \
loss: {(self.total_loss.sum()/self.len_loss):.4f} approx_kl_div: {(self.approx_kl_divs.sum()/self.len_loss):.4f} \
reward_int: {(self.rewards_int.sum()/self.rewards_int_len):.4f} reward_int_norm: {(self.rewards_int_norm.sum()/self.rewards_int_len):.4f} \
episode_time: {datetime.now() - episode_time}\n')
            f_out.close()

        if total_reward > self.max_test_score or info['flag_get']:
            imageio.mimsave('test_episode.mp4', images)
            self.max_test_score = total_reward
            if is_training:
                torch.save(self.model.state_dict(), f"best_model.pth")

        if info['flag_get']:
            self.is_completed = True

    def save_model(self):
        torch.save(self.model.state_dict(), f"model_{self.current_step}.pth")

    def load_model(self, model_path = None):
        if model_path is None:
            model_path = f"model_{self.current_step}.pth"
        self.model.load_state_dict(torch.load(model_path))

    def update_loss_statis(self, loss_p, loss_v, loss_v_int, loss_e, loss, approx_kl_div):
        # update loss for logging, just save 1000 latest updates.
        self.V_loss[self.loss_index] = loss_v
        self.V_int_loss[self.loss_index] = loss_v_int
        self.P_loss[self.loss_index] = loss_p
        self.E_loss[self.loss_index] = loss_e
        self.total_loss[self.loss_index] = loss
        self.approx_kl_divs[self.loss_index] = approx_kl_div
        self.loss_index = (self.loss_index + 1)%1000
        self.len_loss = min(self.len_loss+1, 1000)

    def update_reward_int_log(self, rewards_int, rewards_int_norm):
        for i in range(len(rewards_int)):
            self.rewards_int[self.rewards_int_idx] = rewards_int[i].mean()
            self.rewards_int_norm[self.rewards_int_idx] = rewards_int_norm[i].mean()
            self.rewards_int_idx = (self.rewards_int_idx + 1)%1000
            self.rewards_int_len = min(self.rewards_int_len+1, 1000)

    def select_action(self, states):
        # select action when training, we need use Categorical distribution to make action base on probability from model
        states = torch.tensor(np.array(states), device = self.device)

        with torch.no_grad():
            logits, Values, values_int = self.model(states)
            policy = F.softmax(logits, dim=1)
            distribution = torch.distributions.Categorical(policy)
            actions = distribution.sample().cpu().numpy().tolist()
        return actions, logits, Values, values_int

    def compute_reward_int(self, embeddings):
        """embeddings: (B, D), buffer: (N, D) → rewards (B,)"""

        with torch.no_grad():
            replay_buffer = self.replay_buffer.embeddings.clone()[:self.replay_buffer.size]
            dists = torch.cdist(
                        embeddings.float(),
                        replay_buffer.float(),
                        compute_mode='donot_use_mm_for_euclid_dist',
                        p=2.0
                    )

            if self.average_entropy:
                intrinsic_reward = 0.
                k = min(self.k+1, dists.shape[1])
                knn_dists, _ = dists.topk(k, dim=-1, largest=False, sorted=True)  # (B, k)
                knn_dists = knn_dists[:, 1:] if self.k + 1 <= dists.shape[1] else knn_dists
                intrinsic_reward = torch.log(knn_dists + 1.0).mean(dim=-1)  # (B,)
            else:
                knn_dists, _ = dists.topk(self.k+1, dim=-1, largest=False, sorted=True) # (B, k)
                intrinsic_reward = torch.log(knn_dists[:, -1] + 1.0)  # (B,)

        return intrinsic_reward.reshape(-1).cpu().numpy()

    def compute_emb(self, next_state):
        next_state = torch.tensor(np.array(next_state), device = self.device)
        next_state = next_state[:, 3, :, :].reshape(-1, 1, next_state.shape[2], next_state.shape[3])

        with torch.no_grad():
            embeddings = self.random_encoder_model(next_state)

        return embeddings

    def learn(self):
        # get all data
        states, actions, next_states, rewards, rewards_int, rewards_int_norm, dones, old_logits, old_values, old_values_int = self.memory.get_data()

        # predict next_value and next_value_int for calculate advantage and td target
        targets = []
        targets_int = []
        with torch.no_grad():
            _, next_value, next_value_int = self.model(torch.tensor(np.array(next_states[-1]), device = self.device))
        target = next_value
        target_int = next_value_int
        advantage = 0
        advantage_int = 0

        rewards_int = np.array(rewards_int)
        rewards_int_norm = np.array(rewards_int_norm)

        if self.norm_int_type == 'min_max':
            rewards_int_norm = rewards_int
            rewards_int_norm = (rewards_int_norm - rewards_int_norm.min()) / (rewards_int_norm.max() - rewards_int_norm.min() + 1e-11)
            rewards_int_norm = rewards_int_norm.astype(np.float32)

        self.update_reward_int_log(rewards_int, rewards_int_norm)

        if self.norm_int_type != 'no':
            rewards_int = rewards_int_norm

        # calculate advantage and td target. We need calculate for both reward and intrinsic rewards.
        for reward, reward_int, done, value, value_int in zip(rewards[::-1], rewards_int[::-1], dones[::-1], old_values[::-1], old_values_int[::-1]):
            done = torch.tensor(done, device = self.device, dtype = torch.float).reshape(-1, 1)
            reward = torch.tensor(reward, device = self.device).reshape(-1, 1)
            reward_int = torch.tensor(reward_int, device = self.device).reshape(-1, 1)

            target = next_value * self.gamma * (1-done) + reward
            advantage = target + self.gamma * advantage * (1-done) * self.gae_lambda
            targets.append(advantage)
            advantage = advantage - value.detach()
            next_value = value.detach()

            target_int = next_value_int * self.gamma_int * (1-done) + reward_int
            advantage_int = target_int + self.gamma_int * advantage_int * (1-done) * self.gae_lambda
            targets_int.append(advantage_int)
            advantage_int = advantage_int - value_int.detach()
            next_value_int = value_int.detach()

        # convert all data to tensor
        targets = targets[::-1]
        targets_int = targets_int[::-1]

        states = torch.tensor(np.array(states), device = self.device)
        states = states.reshape((-1,  states.shape[2], states.shape[3], states.shape[4]))

        # self.obs_rms.update(next_states)

        action_index = torch.tensor(actions, dtype = torch.int64, device = self.device)
        action_index = torch.flatten(action_index)

        old_values = torch.cat(old_values, 0)
        old_values_int = torch.cat(old_values_int, 0)

        targets = torch.cat(targets, 0).view(-1, 1)
        targets_int = torch.cat(targets_int, 0).view(-1, 1)

        old_logits = torch.cat(old_logits, 0)
        old_probs = torch.softmax(old_logits, -1)
        index = torch.arange(0, len(old_probs), device = self.device)
        old_log_probs = (old_probs[index, action_index] + 1e-9).log()
        advantages = (targets - old_values).reshape(-1)
        advantages_int = (targets_int - old_values_int).reshape(-1)

        early_stopping = False

        #train num_epoch time
        for epoch in range(self.num_epoch):
            #shuffle data for each epoch
            shuffle_ids = torch.randperm(len(targets), dtype = torch.int64)
            start_id = -1

            for i in range(len(old_values)//self.batch_size):
                #train with batch_size data
                self.optimizer.zero_grad()

                start_id = i * self.batch_size
                end_id = min(len(shuffle_ids), (i+1) * self.batch_size)
                batch_ids = shuffle_ids[start_id:end_id]

                #predict logits and values from model
                batch_state = states[batch_ids]
                logits, value, value_int = self.model(batch_state)

                #calculate entropy and value loss (using mse or huber based on config)
                probs =  torch.softmax(logits, -1)
                entropy = (- (probs * (probs + 1e-9).log()).sum(-1)).mean()
                if self.V_loss_type == 'huber':
                    loss_V = F.smooth_l1_loss(value, targets[batch_ids])
                    loss_V_int = F.smooth_l1_loss(value_int, targets_int[batch_ids])
                else:
                    loss_V = F.mse_loss(value, targets[batch_ids])
                    loss_V_int = F.mse_loss(value_int, targets_int[batch_ids])
                index = torch.arange(0, len(probs), device = self.device)
                batch_action_index = action_index[batch_ids]

                log_probs = (probs[index, batch_action_index] + 1e-9).log()

                #approx_kl_div copy from https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html#PPO
                #if approx_kl_div larger than 1.5 * target_kl (if target_kl in config is not None), stop training because policy change so much
                with torch.no_grad():
                    log_ratio = log_probs - old_log_probs[batch_ids]
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    early_stopping = True

                #calculate policy loss
                ratio = torch.exp(log_probs - old_log_probs[batch_ids])

                if self.current_step <= self.learn_step * 10:
                    loss_V_int = loss_V_int * 0
                    batch_advantages = self.ext_adv_coef * advantages[batch_ids].detach()
                else:
                    batch_advantages = self.ext_adv_coef * advantages[batch_ids].detach() + self.int_adv_coef * advantages_int[batch_ids].detach()

                if self.is_normalize_advantage:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-9)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                loss_P = -torch.min(surr1, surr2).mean()

                # update model
                loss = loss_V * self.V_coef + loss_V_int * self.V_coef + loss_P - entropy * self.entropy_coef

                if early_stopping:
                    break
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    self.update_loss_statis(loss_P.item(), loss_V.item(), loss_V_int.item(), entropy.item(), loss.item(), approx_kl_div.item())

    def train(self):
        episode_reward = [0] * self.num_envs
        episode_step = [0] * self.num_envs
        max_episode_reward = 0
        max_episode_step = 0
        episode_time = [datetime.now() for _ in range(self.num_envs)]
        total_time = datetime.now()

        last_episode_rewards = []

        #reset envs
        states = self.envs.reset()

        while True:
            # finish training if agent reach total_step or total_episode base on what type of total_step_or_episode is step or episode
            self.current_step += 1

            if self.total_step_or_episode == 'step':
                if self.current_step >= self.total_step:
                    break
            else:
                if self.current_episode >= self.total_episode:
                    break

            actions, logit, value, value_int = self.select_action(states)

            next_states, next_states_, rewards, dones, truncs, infos = self.envs.step(actions)

            with torch.no_grad():
                state_embs = self.compute_emb(states)
            self.replay_buffer.pushs(state_embs)

            rewards_int = self.compute_reward_int(state_embs)
            rewards_int_norm = rewards_int / (np.sqrt(self.int_rms.var.cpu().numpy()) + 1e-11)

            if self.current_step > self.learn_step * 10:
                self.int_rms.update(torch.tensor(rewards_int).reshape(-1))

            # save to memory
            self.memory.save(states, actions, rewards, rewards_int, rewards_int_norm, next_states, dones, logit, value, value_int)

            episode_reward = [x + reward for x, reward in zip(episode_reward, rewards)]
            episode_step = [x+1 for x in episode_step]

            # logging after each step, if 1 episode is ending, I will log this to logging.txt
            for i, done in enumerate(dones):
                if done:
                    self.current_episode += 1
                    max_episode_reward = max(max_episode_reward, episode_reward[i])
                    max_episode_step = max(max_episode_step, episode_step[i])
                    last_episode_rewards.append(episode_reward[i])
                    f_out = open(f"logging.txt", "a")
                    f_out.write(f'episode: {self.current_episode} agent: {i} rewards: {episode_reward[i]:.4f} steps: {episode_step[i]} \
complete: {infos[i]["flag_get"]==True} mean_rewards: {np.array(last_episode_rewards[-min(len(last_episode_rewards), 100):]).mean():.4f} \
max_rewards: {max_episode_reward:.4f} max_steps: {max_episode_step} current_step: {self.current_step} \
loss_p: {(self.P_loss.sum()/self.len_loss):.4f} loss_v: {(self.V_loss.sum()/self.len_loss):.4f} \
loss_v_int: {(self.V_int_loss.sum()/self.len_loss):.4f} loss_e: {(self.E_loss.sum()/self.len_loss):.4f} \
loss: {(self.total_loss.sum()/self.len_loss):.4f} approx_kl_div: {(self.approx_kl_divs.sum()/self.len_loss):.4f} \
reward_int: {(self.rewards_int.sum()/self.rewards_int_len):.4f} reward_int_norm: {(self.rewards_int_norm.sum()/self.rewards_int_len):.4f} \
episode_time: {datetime.now() - episode_time[i]} total_time: {datetime.now() - total_time}\n')

                    f_out.close()
                    episode_reward[i] = 0
                    episode_step[i] = 0
                    episode_time[i] = datetime.now()

            # training agent every learn_step
            if self.current_step % self.learn_step == 0:
                self.learn()
                self.memory.reset()
                gc.collect()
                torch.cuda.empty_cache()

            # eval agent every save_figure_step
            if self.current_step % self.save_figure_step == 0:
                self.save_figure(is_training=True)
                if self.is_completed:
                    return

            if self.current_step % self.save_model_step == 0:
                self.save_model()

            states = list(next_states_)

        f_out = open(f"logging.txt", "a")
        f_out.write(f' mean_rewards: {np.array(last_episode_rewards[-min(len(last_episode_rewards), 100):]).mean()} \
max_rewards: {max_episode_reward} max_steps: {max_episode_step} current_step: {self.current_step} \
total_time: {datetime.now() - total_time}\n')
        f_out.close()