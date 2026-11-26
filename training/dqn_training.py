"""Deep Q-Network training on CommunityHealthEnv with TorchRL.

    uv run python -m training.dqn_training                 # one run, default config
    uv run python -m training.dqn_training --sweep         # 10 hyperparameter runs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field

import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.postprocs import MultiStep
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from training.common import (
    LOG_DIR, MODEL_DIR, RunLogger, evaluate, finished_episode_returns, frames_to_first_success,
    make_env, make_q_actor,
)


@dataclass
class DQNConfig:
    lr: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 50_000
    eps_init: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.25     # fraction of training over which epsilon is annealed
    target_update: int = 500         # hard update period (optim steps); 0 -> soft update
    tau: float = 0.005               # soft update rate (used when target_update == 0)
    hidden: tuple = (128, 128)
    double_dqn: bool = True
    n_step: int = 3                  # multi-step TD targets (1 = classic one-step DQN)
    frames_per_batch: int = 100
    updates_per_batch: int = 25
    init_random_frames: int = 2_000
    total_frames: int = 300_000
    seed: int = 0


# 10 configurations used for the hyperparameter study
SWEEP: list[dict] = [
    {},                                                         # baseline
    {"lr": 1e-3},
    {"lr": 1e-4},
    {"gamma": 0.97},
    {"batch_size": 32},
    {"buffer_size": 5_000},
    {"n_step": 1},                                              # classic one-step targets
    {"n_step": 5},
    {"eps_decay_frac": 0.6, "eps_end": 0.1},                    # slow exploration decay
    {"target_update": 0, "tau": 0.01, "double_dqn": False},     # soft target, vanilla DQN
]


def train_dqn(cfg: DQNConfig, run_id: int = 0, verbose: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    env = make_env(seed=cfg.seed)
    actor = make_q_actor(env, cfg.hidden)
    greedy = EGreedyModule(env.action_spec, eps_init=cfg.eps_init, eps_end=cfg.eps_end,
                           annealing_num_steps=int(cfg.eps_decay_frac * cfg.total_frames))
    explore_policy = TensorDictSequential(actor, greedy)

    collector = Collector(env, explore_policy, frames_per_batch=cfg.frames_per_batch,
                          total_frames=cfg.total_frames, init_random_frames=cfg.init_random_frames,
                          postproc=MultiStep(gamma=cfg.gamma, n_steps=cfg.n_step) if cfg.n_step > 1 else None)
    rb = ReplayBuffer(storage=LazyTensorStorage(cfg.buffer_size), batch_size=cfg.batch_size)

    loss_fn = DQNLoss(actor, action_space=env.action_spec, delay_value=True,
                      double_dqn=cfg.double_dqn, loss_function="smooth_l1")
    loss_fn.make_value_estimator(gamma=cfg.gamma)
    target = (HardUpdate(loss_fn, value_network_update_interval=cfg.target_update)
              if cfg.target_update > 0 else SoftUpdate(loss_fn, tau=cfg.tau))
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg.lr)

    logger = RunLogger("dqn", run_id, asdict(cfg))
    frames, best_eval, it = 0, -1e9, 0
    best_path = MODEL_DIR / "dqn" / f"run_{run_id:02d}.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    for data in collector:
        it += 1
        frames += data.numel()
        rb.extend(data.reshape(-1))
        losses = []
        if frames >= cfg.init_random_frames:
            for _ in range(cfg.updates_per_batch):
                sample = rb.sample()
                loss = loss_fn(sample)["loss"]
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 10.0)
                optim.step()
                target.step()
                losses.append(loss.item())
            greedy.step(data.numel())
        eps_returns = finished_episode_returns(data)
        logger.log(frames, eps_returns, loss=sum(losses) / max(len(losses), 1), epsilon=greedy.eps.item())

        if it % 100 == 0:
            ev_ret, ev_succ, ev_len = evaluate(actor, n_episodes=10)
            logger.rows[-1].update(eval_return=ev_ret, eval_success=ev_succ)
            if ev_ret > best_eval:
                best_eval = ev_ret
                torch.save(actor.state_dict(), best_path)
            if verbose:
                print(f"[DQN run {run_id}] frames={frames:>7} eps={greedy.eps.item():.2f} "
                      f"train_ret={logger.rows[-1]['mean_return']:7.2f} eval_ret={ev_ret:7.2f} "
                      f"success={ev_succ:.0%} len={ev_len:.0f}", flush=True)
    collector.shutdown()

    actor.load_state_dict(torch.load(best_path))
    final_ret, final_succ, final_len = evaluate(actor, n_episodes=50, seed=777)
    summary = {
        "algo": "dqn", "run_id": run_id, "eval_return": final_ret, "eval_success": final_succ,
        "eval_length": final_len, "episodes": logger.episodes,
        "frames_to_converge": frames_to_first_success(logger.rows, threshold=15.0),
        "train_time_s": logger.rows[-1]["time_s"],
    }
    logger.save(summary)
    if verbose:
        print(f"[DQN run {run_id}] FINAL eval_return={final_ret:.2f} success={final_succ:.0%}")
    return summary


def run_sweep(runs: list[int] | None = None, total_frames: int | None = None, workers: int = 1):
    from training.sweep import sweep
    return sweep("dqn", runs, total_frames, workers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--runs", type=int, nargs="*")
    p.add_argument("--frames", type=int)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    if a.sweep:
        run_sweep(a.runs, a.frames, a.workers)
    else:
        cfg = DQNConfig(**({"total_frames": a.frames} if a.frames else {}))
        print(json.dumps(train_dqn(cfg), indent=2))


if __name__ == "__main__":
    main()
