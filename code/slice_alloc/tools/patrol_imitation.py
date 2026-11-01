"""Behavior-cloning warm start: pretrain on price's decisions, then fine-tune on
the real SLA-success objective. Price is used only to generate an initial
dataset of (observation, chosen_action) pairs; it is never read during later
RL training/evaluation, and the fine-tuning target is real settlement success,
not price's score. This is a third attempt after reward shaping (helped
stability, capped at price's own quality) and candidate-feature engineering
(no improvement this round); the goal is a positive starting bias rather than
random exploration, which may or may not surpass price after fine-tuning.
"""
from __future__ import annotations
import concurrent.futures
import numpy as np
from slice_alloc.patrol import PatrolEnv, PROFILES, STATE_DIM, CANDIDATE_DIM, baseline
from slice_alloc.tools.patrol_run import OUT, scenario

STEPS = 32768
PRETRAIN_EPISODES = 400
PROBE_SEEDS = [11, 23]
# Disjoint from both the primary test seeds (4100-4129) and the holdout
# probe seeds (5000-5014) used by prior diagnostics in this session.
PRETRAIN_SEED_RANGE = (20000, 20000 + PRETRAIN_EPISODES)
HOLDOUT_SEEDS = list(range(5000, 5015))


def collect_demonstrations():
    """(state, candidates, mask, chosen_action) from price's own decisions."""
    rows = []
    for seed in range(*PRETRAIN_SEED_RANGE):
        rng = np.random.default_rng(seed)
        env = PatrolEnv(seed=seed, loads=(24, 40), profiles=PROFILES)
        obs, _ = env.reset(options=dict(scenario=scenario(seed, int(rng.choice((24, 40))),
                                                           str(rng.choice(PROFILES)))))
        done = False
        while not done:
            action = baseline('price', env)
            rows.append((obs['state'].copy(), obs['candidates'].copy(), obs['mask'].copy(), action))
            obs, reward, done, _, _ = env.step(action)
        env.close()
    return rows


def pretrain_ddqn(seed, demos):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    from slice_alloc.src.qlearning import DoubleDQN
    agent = DoubleDQN(STATE_DIM, CANDIDATE_DIM, seed)
    opt = torch.optim.Adam(agent.network.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(demos))
    for epoch in range(15):
        rng.shuffle(idx)
        for b in range(0, len(idx), 128):
            batch = [demos[i] for i in idx[b:b + 128]]
            state = torch.as_tensor(np.stack([x[0] for x in batch]))
            cand = torch.as_tensor(np.stack([x[1] for x in batch]))
            mask = torch.as_tensor(np.stack([x[2] for x in batch]))
            action = torch.tensor([x[3] for x in batch], dtype=torch.long)
            logits = agent.network({'state': state, 'candidates': cand, 'mask': mask})
            loss = F.cross_entropy(logits, action)
            opt.zero_grad(); loss.backward(); opt.step()
    agent.target.load_state_dict(agent.network.state_dict())
    return agent


def pretrain_ppo_policy(seed, demos):
    """Clone price's decisions into a CandidatePolicy via its own actor head."""
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from slice_alloc.src.policy import CandidatePolicy
    env = PatrolEnv(seed=seed + 100, loads=(24, 40), profiles=PROFILES)
    model = PPO(CandidatePolicy, env, learning_rate=.0003, n_steps=256, batch_size=128,
               n_epochs=5, gamma=1., seed=seed, device='cpu', verbose=0)
    opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(demos))
    for epoch in range(15):
        rng.shuffle(idx)
        for b in range(0, len(idx), 128):
            batch = [demos[i] for i in idx[b:b + 128]]
            obs = {'state': torch.as_tensor(np.stack([x[0] for x in batch])),
                   'candidates': torch.as_tensor(np.stack([x[1] for x in batch])),
                   'mask': torch.as_tensor(np.stack([x[2] for x in batch]))}
            action = torch.tensor([x[3] for x in batch], dtype=torch.long)
            distribution, _ = model.policy._outputs(obs)
            loss = F.nll_loss(torch.log_softmax(distribution.distribution.logits, dim=-1), action)
            opt.zero_grad(); loss.backward(); opt.step()
    env.close()
    return model


def finetune_and_save(spec):
    import torch
    torch.set_num_threads(1)
    kind, seed, demos = spec
    env = PatrolEnv(seed=seed + 100, loads=(24, 40), profiles=PROFILES)
    out = OUT / 'imitation_probe' / f'{kind}-{seed}'
    out.parent.mkdir(parents=True, exist_ok=True)
    if kind == 'ddqn':
        agent = pretrain_ddqn(seed, demos)
        obs, _ = env.reset()
        for step in range(STEPS):
            epsilon = max(.08, .9 * (1 - step / (STEPS * .8)))
            action = agent.explore(obs, epsilon)
            nxt, reward, done, _, _ = env.step(action)
            agent.remember(obs, action, reward, nxt, done)
            if len(agent.replay) >= 128 and step % 4 == 0:
                agent.update()
            obs = env.reset()[0] if done else nxt
        agent.save(str(out) + '.pt')
    else:
        model = pretrain_ppo_policy(seed, demos)
        model.set_env(env)
        model.learn(total_timesteps=STEPS, reset_num_timesteps=False)
        model.save(str(out) + '.zip')
    env.close()
    return (kind, seed)


def main():
    print('collecting price demonstrations...', flush=True)
    demos = collect_demonstrations()
    print(f'collected {len(demos)} (state,action) pairs from price', flush=True)
    specs = [('ddqn', s, demos) for s in PROBE_SEEDS] + [('ppo', s, demos) for s in PROBE_SEEDS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for kind, seed in pool.map(finetune_and_save, specs):
            print(f'{kind}-{seed} done', flush=True)


if __name__ == '__main__':
    main()
