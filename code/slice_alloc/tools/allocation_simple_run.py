"""Minimal final pipeline for research point 1: intelligent slice allocation.

The controller makes one initial decision when a request arrives.  It chooses
reject or one legal (modality, edge, mapping/cache) candidate.  The native
simulator produces the traffic and the reward is the number of requests whose
actual packet trace satisfies the request SLA.  There are no resource tiers,
overbooking, prices, Lagrange multipliers, prediction or runtime migration.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from slice_alloc import modal_binding as b
from slice_alloc import allocation_final as environment
from slice_alloc.allocation_value import MaskedDoubleDQN
from slice_alloc.distributional_allocation import (
    CandidateQRPolicy,
    NStepDictReplay,
)
from slice_alloc.tools import modal_binding_run as native

ROOT = b.p.ROOT
OUT = Path(os.environ.get("LOWALT_SIMPLE_OUTPUT", str(ROOT / "slice_alloc/results/simple"))).resolve()
INITS = (11, 23, 37, 53, 71)
LOADS = (48, 80)
VALIDATION = tuple(range(92000, 92030))
TEST = tuple(range(93000, 93030))
BUDGETS = (8192, 16384, 32768)
METHODS = ("double_dqn", "greedy", "nearest", "first_fit")
SOURCES = (
    "allocation_final.py",
    "allocation_value.py",
    "modal_binding.py",
    "modal_binding_rl.py",
    "distributional_allocation.py",
    "patrol.py",
    "initial_allocation.py",
    "patrol_contract.schema.json",
    "tools/modal_binding_run.py",
    "tests/test_allocation_simple.py",
    "docs/REFERENCE_TRAINING.md",
    "execution/allocation_final_scenario.cc",
    "tools/allocation_simple_run.py",
)
HYPER = dict(
    learning_rate=1e-4,
    buffer_size=20000,
    learning_starts=1024,
    batch_size=128,
    train_freq=4,
    gradient_steps=1,
    gamma=1.0,
    target_update_interval=1000,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    exploration_steps=16384,
    max_grad_norm=10.0,
    policy_kwargs=dict(n_quantiles=1, net_arch=[96]),
    replay_buffer_class=NStepDictReplay,
    replay_buffer_kwargs=dict(horizon=8, discount=1.0),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def parameters():
    return copy.deepcopy(HYPER)


def freeze():
    protocol = dict(
        title="面向低空智联的多模态网络切片技术",
        research="智能分配",
        environment="12 UAV, two same-channel APs, two edge nodes, ns-3 native IP/EID-RLOC/NDN execution",
        sources={name: sha(ROOT / "slice_alloc" / name) for name in SOURCES},
        binary_sha256=sha(b.p.NS3 / "build/scratch/allocation-final-scenario"),
        initializations=INITS,
        training_seed_range=[41000000, 42000000],
        loads=LOADS,
        validation=VALIDATION,
        test=TEST,
        budgets=BUDGETS,
        methods=METHODS,
        primary="masked Double-DQN",
        decision="reject or choose one legal modality, edge, mapping/server and cache placement",
        reward="number of newly settled requests whose native packet trace satisfies SLA; rejection scores zero",
        baselines="Greedy minimum normalized reservation, nearest-edge, first legal candidate",
        scope="initial allocation only; no prediction, migration, resource tiers, overbooking or QoS controller",
        hyperparameters={**parameters(), "replay_buffer_class": "slice_alloc.distributional_allocation.NStepDictReplay"},
        checkpoint_selection="highest validation SLA success, tie earliest checkpoint",
    )
    path = OUT / "protocol.json"
    normalized = json.loads(json.dumps(protocol))
    if path.exists():
        if read(path) != normalized:
            raise AssertionError("frozen simple protocol changed")
    else:
        write(path, normalized)
        for name in SOURCES:
            target = OUT / "frozen" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "slice_alloc" / name, target)
    return protocol


def scenario(seed, load):
    return environment.scenario(seed, load, "balanced")


def greedy_action(env):
    if not env.options:
        return 0
    return min(range(len(env.options)), key=lambda i: float(np.sum(env.options[i].peak / env.ledger.capacity))) + 1


def nearest_action(env):
    if not env.options:
        return 0
    request = env.requests[env.current["request_id"]]
    attached = [env.current["associations"][str(n)] - 2 for n in request.members]
    return min(
        range(len(env.options)),
        key=lambda i: (sum(env.options[i].edge != edge for edge in attached),
                       float(np.sum(env.options[i].peak / env.ledger.capacity))),
    ) + 1


def choose(method, env, model=None):
    if model is not None:
        return int(model.predict(env.encode(), deterministic=True)[0])
    if method == "greedy":
        return greedy_action(env)
    if method == "nearest":
        return nearest_action(env)
    if method == "first_fit":
        return 1 if env.options else 0
    raise ValueError(method)


def episode(cfg, method, model=None, prefix=None):
    env = environment.AllocationEnv(no_calendar=True)
    started = time.perf_counter()
    try:
        obs, _ = env.reset(options={"scenario": cfg, "prefix": prefix, "trace": prefix is not None})
        done = False
        reward = 0.0
        decision_times = []
        while not done:
            decision_start = time.perf_counter()
            action = choose(method, env, model)
            decision_times.append(time.perf_counter() - decision_start)
            if obs["mask"][action] <= 0.5:
                raise AssertionError("policy selected an infeasible action")
            obs, value, done, _, _ = env.step(action)
            reward += value
        result = env.result()
        if abs(reward - result["success"]) > 1e-9:
            raise AssertionError("reward is not native SLA success count")
        result.update(
            method=method,
            episode_seconds=time.perf_counter() - started,
            decision_ms=1000 * float(np.mean(decision_times)),
            scenario_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        )
        if prefix is not None:
            # Reuse the independent packet/event audit from the native runner.
            raw_result = Path(str(prefix) + "-result.json")
            write(raw_result, result)
            raw_events = Path(str(prefix) + "-events.csv")
            if raw_events.exists():
                with raw_events.open("rb") as source, gzip.open(str(raw_events) + ".gz", "wb", compresslevel=3) as target:
                    shutil.copyfileobj(source, target)
                raw_events.unlink()
            result.update(native.audit(prefix, result, cfg))
            write(str(prefix) + "-metrics.json", {k: v for k, v in result.items()
                if k not in ("decisions", "settled", "observations")})
        return result
    finally:
        env.close()


def score(model):
    rows = []
    for seed in VALIDATION:
        for load in LOADS:
            result = episode(scenario(seed, load), "double_dqn", model)
            rows.append({"seed": seed, "load": load, "requests": result["requests"], "success": result["success"]})
    return dict(rate=sum(r["success"] for r in rows) / sum(r["requests"] for r in rows), rows=rows)


class Checkpoints(BaseCallback):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.records = []

    def _on_step(self):
        if self.num_timesteps not in BUDGETS:
            return True
        path = self.folder / f"{self.num_timesteps}.zip"
        self.model.save(path)
        validation = score(self.model)
        self.records.append(dict(step=self.num_timesteps, path=str(path.relative_to(OUT)), sha256=sha(path), validation=validation))
        write(self.folder / "progress.json", self.records)
        print("CHECKPOINT", self.num_timesteps, validation["rate"], flush=True)
        return True


def train(init):
    torch.set_num_threads(1)
    freeze()
    folder = OUT / "training" / str(init)
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / "training.json"
    if cached.exists():
        result = read(cached)
        assert result["protocol_sha256"] == sha(OUT / "protocol.json")
        assert result["model_sha256"] == sha(OUT / result["model"])
        return result
    env = environment.AllocationEnv(seed=init + 100, loads=LOADS, no_calendar=True)
    model = MaskedDoubleDQN(CandidateQRPolicy, env, **parameters(), seed=init, device="cpu", verbose=0)
    callback = Checkpoints(folder)
    before = model.quantile_net.deployment[-1].weight.detach().clone()
    started = time.perf_counter()
    try:
        model.learn(max(BUDGETS), callback=callback)
        if not callback.records:
            raise AssertionError("no validation checkpoint was produced")
        if torch.equal(before, model.quantile_net.deployment[-1].weight.detach()):
            raise AssertionError("policy weights did not change")
        best = max(callback.records, key=lambda item: (item["validation"]["rate"], -item["step"]))
        selected = OUT / "models" / f"double_dqn-{init}.zip"
        selected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT / best["path"], selected)
        again = MaskedDoubleDQN.load(selected, device="cpu")
        assert score(again) == best["validation"], "reloaded model changed validation"
        result = dict(init=init, steps=model.num_timesteps, updates=model._n_updates,
                      seconds=time.perf_counter() - started, selected=best,
                      checkpoints=callback.records, model=str(selected.relative_to(OUT)),
                      training_seeds=env.episode_specs,
                      model_sha256=sha(selected), protocol_sha256=sha(OUT / "protocol.json"))
        write(cached, result)
        return result
    finally:
        env.close()


def evaluate(method, init=0, seeds=TEST, load_values=LOADS):
    model = None if method in ("greedy", "nearest", "first_fit") else MaskedDoubleDQN.load(
        OUT / "models" / f"double_dqn-{init}.zip", device="cpu")
    torch.set_num_threads(1)
    rows = []
    for seed in seeds:
        for load in load_values:
            prefix = OUT / "native" / f"{method}-{init}" / f"{load}-{seed}"
            metrics = Path(str(prefix) + "-metrics.json")
            result = read(metrics) if metrics.exists() else episode(scenario(seed, load), method, model, prefix)
            rows.append({"method": method, "init": init, "seed": seed, "load": load,
                         "requests": result["requests"], "success": result["success"],
                         "admitted": result["admitted"], "by_service": result["by_service"],
                         "audited": result.get("audited", False)})
    return rows


def evaluate_task(spec):
    return evaluate(*spec)


def ensure_tests():
    """Only a successful subprocess may create a completion marker."""
    marker = OUT / "tests_complete.json"
    expected = dict(protocol_sha256=sha(OUT / "protocol.json"))
    if marker.exists() and read(marker) == expected:
        return
    with (OUT / "tests.log").open("w") as stream:
        subprocess.run([sys.executable, "-m", "unittest", "slice_alloc.tests.test_allocation_simple"],
                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    write(marker, expected)


def prepare_evaluation():
    completion = read(OUT / "training_complete.json")
    assert completion["protocol_sha256"] == sha(OUT / "protocol.json")
    assert {r["init"] for r in completion["models"]} == set(INITS)
    for row in completion["models"]:
        assert row["protocol_sha256"] == sha(OUT / "protocol.json")
        assert row["model_sha256"] == sha(OUT / row["model"])
    snapshot = {name: sha(OUT / name) for name in
                ("protocol.json", "training_complete.json", "tests_complete.json")}
    marker = OUT / "test_started.json"
    if marker.exists():
        assert read(marker) == snapshot, "formal evaluation choices changed"
    else:
        write(marker, snapshot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "train", "evaluate", "all"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    freeze()
    if args.command != "freeze":
        ensure_tests()
    if args.command in ("train", "all"):
        assert not (OUT / "test_started.json").exists(), "training after opening formal test is not allowed"
        with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            trained = list(pool.map(train, INITS))
        write(OUT / "training_complete.json", {"models": trained, "protocol_sha256": sha(OUT / "protocol.json")})
    if args.command in ("evaluate", "all"):
        if not (OUT / "training_complete.json").exists():
            raise SystemExit("run train before evaluate")
        prepare_evaluation()
        rows = []
        with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            tasks = [("double_dqn", i) for i in INITS] + [(m, 0) for m in METHODS if m != "double_dqn"]
            for group in pool.map(evaluate_task, tasks):
                rows.extend(group)
        write(OUT / "evaluation.json", rows)
        print("EVALUATED", len(rows), flush=True)


if __name__ == "__main__":
    main()
