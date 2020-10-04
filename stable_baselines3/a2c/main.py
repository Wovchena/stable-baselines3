# import collections
import datetime
import itertools

import cv2
import gym
import gym.wrappers
import gym.envs.atari
import numpy as np
import torch
from torch.utils import tensorboard


class CropGrayScaleResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.shape = (84, 75)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, observation):
        return cv2.resize(cv2.cvtColor(observation[-180:, :160], cv2.COLOR_RGB2GRAY),
                          (75, 84), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def breakout():
    return gym.wrappers.FrameStack(CropGrayScaleResizeWrapper(gym.wrappers.TimeLimit(gym.envs.atari.AtariEnv('breakout', obs_type='image',
        frameskip=4, repeat_action_probability=0.25), max_episode_steps=60000)), 4)


class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, n_alternatives):
        """n_alternatives == 1 -> Normal"""
        super().__init__()
        self.c1 = torch.nn.Conv2d(state_dim, 32, kernel_size=8, stride=4, padding=4)
        self.c2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=2)
        self.c3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(7680, 512)
        self.actor = torch.nn.Linear(512, n_alternatives)
        self.critic = torch.nn.Linear(512, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), 7e-4)  # TODO 1-e3

    def forward(self, state):
        # TODO Mish, BatchNorm
        conved = torch.flatten(torch.nn.functional.relu(self.c3(torch.nn.functional.relu(self.c2(torch.nn.functional.relu(self.c1(state)))))), 1)
        features = torch.nn.functional.relu(self.fc1(conved))
        return torch.distributions.Categorical(logits=self.actor(features)), self.critic(features)


def transpose_reshape(x):
    return x.transpose(0, 1).reshape(x.shape[0] * x.shape[1], *x.shape[2:])


def train_step(mem, last_values, last_dones, policy, device, gamma, gae_lambda):
    observations, actions, rewards, dones, values, log_probs = mem
    gae = 0.0
    advantages = []
    next_value = last_values
    next_done = last_dones
    for step_id in reversed(range(len(observations))):
        delta = rewards[step_id] + gamma * (1.0 - next_done) * next_value - values[step_id]
        gae = delta + gamma * gae_lambda * (1.0 - next_done) * gae
        advantages.append(gae)
        next_value = values[step_id]
        next_done = dones[step_id]
    advantages.reverse()

    actions = transpose_reshape(torch.stack(actions))
    observations = transpose_reshape(torch.stack(observations))
    advantages = transpose_reshape(torch.stack(advantages))
    rb_values = transpose_reshape(torch.stack(values))
    returns = advantages + rb_values

    actions = actions.long().flatten()

    distr, values = policy(observations)
    log_prob = distr.log_prob(actions)
    entropy = distr.entropy()
    values = values.flatten()

    # Policy gradient loss
    policy_loss = -(advantages * log_prob).mean()

    # Value loss using the TD(gae_lambda) target
    value_loss = torch.nn.functional.mse_loss(returns, values)
    entropy_loss = -entropy.mean()

    loss = policy_loss + 0.01 * entropy_loss + 0.25 * value_loss  # TODO try 1 as value_loss coeff

    # Optimization step
    policy.optimizer.zero_grad()
    loss.backward()

    # Clip grad norm
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)  # TODO does it work without it?
    policy.optimizer.step()
    return value_loss.item()


def main():
    DEVICE = torch.device('cuda')
    GAMMA = 0.99
    envs = gym.vector.async_vector_env.AsyncVectorEnv((breakout,) * 16)  # 4 + (lambda: RenderWrapper(breakout()),)
    policy = ActorCritic(envs.observation_space.shape[1], envs.action_space[0].n).to(DEVICE)

    current_scores = np.zeros((envs.num_envs,), dtype=np.float64)  # VectorEnv returns as np.float64
    last_scores = []
    losses = []

    observations = envs.reset()
    prev_dones = np.zeros((envs.num_envs,), dtype=np.bool)
    mem = [], [], [], [], [], []  # obs, actions, rewards, dones, values, log_probs
    writer = tensorboard.SummaryWriter(datetime.datetime.now().strftime('logs/main/%d-%m-%Y %H-%M'))
    for step_id in itertools.count():
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            distr, values = policy(obs_tensor)
        actions = distr.sample()
        log_probs = distr.log_prob(actions)
        actions = actions.cpu().numpy()

        new_obs, rewards, dones, infos = envs.step(actions)

        # Reshape in case of discrete action
        actions = actions.reshape(-1, 1)
        mem[0].append(obs_tensor)
        mem[1].append(torch.as_tensor(actions, device=DEVICE))
        mem[2].append(torch.as_tensor(rewards, device=DEVICE))
        mem[3].append(torch.as_tensor(prev_dones, dtype=torch.float32, device=DEVICE))
        mem[4].append(values.squeeze())
        mem[5].append(log_probs)
        observations = new_obs
        prev_dones = dones

        current_scores += rewards
        if dones.any():
            last_scores.extend(current_scores[dones])
            current_scores[dones] = 0.0
            if len(last_scores) > 999:
                scores = np.mean(last_scores)
                val_loss = np.mean(losses)
                writer.add_scalar('Scalars/Score', scores, step_id)
                writer.add_scalar('Scalars/Value loss', val_loss, step_id)
                print(step_id, scores, val_loss)
                last_scores = []
                losses = []

        if len(mem[0]) > 5:
            with torch.no_grad():
                _, last_value = policy(torch.as_tensor(new_obs, dtype=torch.float32, device=DEVICE))
            loss = train_step(mem, last_value.squeeze(), torch.as_tensor(dones, dtype=torch.float32, device=DEVICE), policy, DEVICE, GAMMA, 0.92)
            mem = [], [], [], [], [], []
            losses.append(loss)


if __name__ == '__main__':
    main()

    # envs = vec_env.vec_transpose.VecTransposeImage(
    #     VecFrameStack(make_atari_env('BreakoutNoFrameskip-v4', n_envs=16, seed=0), n_stack=4))
    # model = stable_baselines3.A2C(CnnPolicy, envs)
    # model.learn(1e7)
