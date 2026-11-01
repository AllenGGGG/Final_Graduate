"""Candidate-aware masked Double DQN; target actions use online-network argmax."""
from collections import deque
import numpy as np
import torch
from torch import nn


class CandidateQ(nn.Module):
    def __init__(self, state_dim, candidate_dim):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.candidate = nn.Sequential(nn.Linear(candidate_dim, 64), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs):
        state = self.state(obs['state'].float())
        candidate = self.candidate(obs['candidates'].float())
        q = self.score(torch.cat([state[:, None, :].expand(-1, candidate.shape[1], -1), candidate], -1)).squeeze(-1)
        return q.masked_fill(obs['mask'] < .5, -1e9)


def tensor_observation(obs):
    return {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}


class DoubleDQN:
    def __init__(self, state_dim, candidate_dim, seed=11):
        torch.manual_seed(seed)
        self.network = CandidateQ(state_dim, candidate_dim)
        self.target = CandidateQ(state_dim, candidate_dim)
        self.target.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        self.rng = np.random.default_rng(seed)
        self.replay = deque(maxlen=10000)
        self.dims = (state_dim, candidate_dim)
        self.updates = 0

    def predict(self, obs, deterministic=True):
        with torch.no_grad():
            action = int(self.network(tensor_observation(obs)).argmax(1).item())
        return action, None

    def explore(self, obs, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(obs['mask'])))
        return self.predict(obs)[0]

    def remember(self, obs, action, reward, next_obs, done):
        self.replay.append((obs, action, reward, next_obs, done))

    def update(self, batch_size=128):
        rows = [self.replay[i] for i in self.rng.choice(len(self.replay), batch_size, replace=False)]
        obs = {k: torch.as_tensor(np.stack([r[0][k] for r in rows])) for k in rows[0][0]}
        next_obs = {k: torch.as_tensor(np.stack([r[3][k] for r in rows])) for k in rows[0][3]}
        action = torch.tensor([r[1] for r in rows], dtype=torch.long)
        reward = torch.tensor([r[2] for r in rows], dtype=torch.float32)
        done = torch.tensor([r[4] for r in rows], dtype=torch.float32)
        prediction = self.network(obs).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            best = self.network(next_obs).argmax(1)
            target = reward + (1.-done)*self.target(next_obs).gather(1, best[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(prediction, target)
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 10.)
        self.optimizer.step()
        self.updates += 1
        if self.updates % 250 == 0:
            self.target.load_state_dict(self.network.state_dict())
        return float(loss.detach())

    def save(self, path):
        torch.save({'dims': self.dims, 'weights': self.network.state_dict()}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        model = cls(*data['dims'])
        model.network.load_state_dict(data['weights'])
        model.target.load_state_dict(data['weights'])
        return model
