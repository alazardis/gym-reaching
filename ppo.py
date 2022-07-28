import torch

from torch.distribution import MultivariateNormal
from network import FeedForwardNN

class PPO:
    def __init__(self, env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self.__init__hyperparameters()
        self.cov_var= torch.full(size=(self.act_dim, ), fill_value=0.5)

        self.cov_mat = torch.diag(self.cov_var)

    def learm(self, total_timesteps):
        t_so_far = 0
        while t_so_far < total_timesteps:

    def __init__hyperparameters(self):
        self.timesteps_per_batch = 10000
        self.max_timesteps_per_episode = 100

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs = self.env.reset()
            done = False
            for ep_t in range(self.max_timesteps_per_episode):

                t += 1

                batch_obs.append(obs)
                action = self.env.action_space.sample()
                obs, rew, done, _ = self.env.step(action)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)

        batch-rtgs = self.compute_rtgs(batch_rews)

    def get_action(self, obs):
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(action)
