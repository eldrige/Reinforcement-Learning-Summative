"""Policy-gradient training (REINFORCE, PPO, A2C) on CommunityHealthEnv with TorchRL.

    uv run python -m training.pg_training --algo ppo              # one run, default config
    uv run python -m training.pg_training --algo a2c --sweep      # 10 hyperparameter runs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE

from training.common import (
    MODEL_DIR, RunLogger, evaluate, finished_episode_returns, frames_to_first_success,
    make_critic, make_env, make_policy,
)


# --------------------------------------------------------------------------- configs
@dataclass
class ReinforceConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    hidden: tuple = (128, 128)
    frames_per_batch: int = 1_000     # ~5-15 full episodes per policy update
    use_baseline: bool = True         # learned value baseline (REINFORCE with baseline)
    normalize_adv: bool = True
    entropy_coef: float = 0.01
    total_frames: int = 300_000
    seed: int = 0


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 10
    minibatch: int = 256
    frames_per_batch: int = 2_048
    entropy_coef: float = 0.01
    critic_coef: float = 0.5
    hidden: tuple = (128, 128)
    total_frames: int = 300_000
    seed: int = 0


@dataclass
class A2CConfig:
    lr: float = 7e-4
    gamma: float = 0.99
    gae_lambda: float = 1.0           # 1.0 -> plain n-step advantage (classic A2C)
    frames_per_batch: int = 64        # rollout length (n-steps) per update
    entropy_coef: float = 0.01
    critic_coef: float = 0.5
    hidden: tuple = (128, 128)
    optimizer: str = "rmsprop"
    max_grad_norm: float = 0.5
    total_frames: int = 300_000
    seed: int = 0


CONFIGS = {"reinforce": ReinforceConfig, "ppo": PPOConfig, "a2c": A2CConfig}

SWEEPS: dict[str, list[dict]] = {
    "reinforce": [
        {},
        {"lr": 3e-3},
        {"lr": 3e-4},
        {"gamma": 0.95},
        {"use_baseline": False, "normalize_adv": True},     # vanilla REINFORCE, normalised returns
        {"use_baseline": False, "normalize_adv": False},    # vanilla REINFORCE, raw returns
        {"entropy_coef": 0.0},
        {"entropy_coef": 0.05},
        {"frames_per_batch": 3_000},                         # more episodes per update
        {"hidden": (256, 256)},
    ],
    "ppo": [
        {},
        {"lr": 1e-3},
        {"lr": 1e-4},
        {"clip_eps": 0.1},
        {"clip_eps": 0.3},
        {"epochs": 4},
        {"gae_lambda": 0.8, "gamma": 0.95},
        {"entropy_coef": 0.0},
        {"entropy_coef": 0.05},
        {"frames_per_batch": 1_024, "minibatch": 128, "hidden": (256, 256)},
    ],
    "a2c": [
        {},
        {"lr": 3e-4},
        {"lr": 2e-3},
        {"frames_per_batch": 16},
        {"frames_per_batch": 256},
        {"gae_lambda": 0.95},
        {"gamma": 0.95},
        {"entropy_coef": 0.0},
        {"entropy_coef": 0.05},
        {"optimizer": "adam", "hidden": (256, 256)},
    ],
}


# --------------------------------------------------------------------------- helpers
def monte_carlo_returns(data, gamma: float):
    """Discounted return-to-go G_t inside a flat [T] rollout; keeps only finished episodes."""
    rew = data["next", "reward"].squeeze(-1)
    done = data["next", "done"].squeeze(-1)
    ends = torch.nonzero(done).squeeze(-1)
    if len(ends) == 0:
        return None
    data = data[: int(ends[-1]) + 1]
    rew, done = rew[: len(data)], done[: len(data)]
    g = torch.zeros_like(rew)
    running = 0.0
    for t in range(len(rew) - 1, -1, -1):
        running = rew[t] + gamma * running * (~done[t]).float()
        g[t] = running
    data.set("value_target", g.unsqueeze(-1))
    return data


def _entropy(actor, td):
    return actor.get_dist(td).entropy().mean()


# --------------------------------------------------------------------------- trainer
def train_pg(algo: str, cfg, run_id: int = 0, verbose: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    env = make_env(seed=cfg.seed)
    actor = make_policy(env, cfg.hidden)
    critic = make_critic(env, cfg.hidden)

    collector = Collector(env, actor, frames_per_batch=cfg.frames_per_batch,
                          total_frames=cfg.total_frames,
                          reset_at_each_iter=(algo == "reinforce"))

    if algo == "ppo":
        adv = GAE(gamma=cfg.gamma, lmbda=cfg.gae_lambda, value_network=critic, average_gae=False)
        loss_fn = ClipPPOLoss(actor, critic, clip_epsilon=cfg.clip_eps, entropy_coeff=cfg.entropy_coef,
                              critic_coeff=cfg.critic_coef, normalize_advantage=True,
                              loss_critic_type="smooth_l1")
        rb = ReplayBuffer(storage=LazyTensorStorage(cfg.frames_per_batch),
                          sampler=SamplerWithoutReplacement(), batch_size=cfg.minibatch)
    elif algo == "a2c":
        adv = GAE(gamma=cfg.gamma, lmbda=cfg.gae_lambda, value_network=critic, average_gae=False)
        loss_fn = A2CLoss(actor, critic, entropy_coeff=cfg.entropy_coef, critic_coeff=cfg.critic_coef,
                          loss_critic_type="smooth_l1")
    else:
        loss_fn = ReinforceLoss(actor, critic, loss_critic_type="smooth_l1")

    params = list(actor.parameters()) + list(critic.parameters())
    if algo == "a2c" and cfg.optimizer == "rmsprop":
        optim = torch.optim.RMSprop(params, lr=cfg.lr, alpha=0.99, eps=1e-5)
    else:
        optim = torch.optim.Adam(params, lr=cfg.lr)
    max_norm = getattr(cfg, "max_grad_norm", 1.0)

    logger = RunLogger(algo, run_id, asdict(cfg))
    best_path = MODEL_DIR / "pg" / f"{algo}_run_{run_id:02d}.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    frames, best_eval, it = 0, -1e9, 0
    eval_every = max(1, 20_000 // cfg.frames_per_batch)
    pending_returns: list[float] = []

    for data in collector:
        it += 1
        frames += data.numel()
        pending_returns += finished_episode_returns(data)
        metrics = {}

        if algo == "ppo":
            for _ in range(cfg.epochs):
                with torch.no_grad():
                    adv(data)
                rb.empty()
                rb.extend(data.reshape(-1))
                for _ in range(cfg.frames_per_batch // cfg.minibatch):
                    sub = rb.sample()
                    out = loss_fn(sub)
                    loss = out["loss_objective"] + out["loss_critic"] + out.get("loss_entropy", 0.0)
                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm)
                    optim.step()
            metrics = {"loss_policy": out["loss_objective"].item(), "loss_value": out["loss_critic"].item(),
                       "entropy": _entropy(actor, data).item() if "entropy" not in out.keys() else out["entropy"].item()}
        elif algo == "a2c":
            with torch.no_grad():
                adv(data)
            out = loss_fn(data)
            loss = out["loss_objective"] + out["loss_critic"] + out.get("loss_entropy", 0.0)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm)
            optim.step()
            metrics = {"loss_policy": out["loss_objective"].item(), "loss_value": out["loss_critic"].item(),
                       "entropy": _entropy(actor, data).item() if "entropy" not in out.keys() else out["entropy"].item()}
        else:  # REINFORCE (Monte-Carlo policy gradient)
            batch = monte_carlo_returns(data, cfg.gamma)
            if batch is not None:
                g = batch["value_target"]
                if cfg.use_baseline:
                    with torch.no_grad():
                        a = g - critic(batch.clone())["state_value"]
                else:
                    a = g.clone()
                if cfg.normalize_adv:
                    a = (a - a.mean()) / (a.std() + 1e-8)
                batch.set("advantage", a)
                out = loss_fn(batch)
                ent = _entropy(actor, batch)
                loss = out["loss_actor"] - cfg.entropy_coef * ent
                if cfg.use_baseline:
                    loss = loss + 0.5 * out["loss_value"]
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm)
                optim.step()
                metrics = {"loss_policy": out["loss_actor"].item(), "loss_value": out["loss_value"].item(),
                           "entropy": ent.item()}

        if pending_returns or metrics:
            logger.log(frames, pending_returns, **metrics)
            pending_returns = []

        if it % eval_every == 0:
            ev_ret, ev_succ, ev_len = evaluate(actor, n_episodes=10)
            logger.rows[-1].update(eval_return=ev_ret, eval_success=ev_succ)
            if ev_ret > best_eval:
                best_eval = ev_ret
                torch.save(actor.state_dict(), best_path)
            if verbose:
                print(f"[{algo.upper()} run {run_id}] frames={frames:>7} "
                      f"train_ret={logger.rows[-1]['mean_return']:7.2f} eval_ret={ev_ret:7.2f} "
                      f"success={ev_succ:.0%} len={ev_len:.0f} entropy={metrics.get('entropy', float('nan')):.3f}",
                      flush=True)
    collector.shutdown()

    actor.load_state_dict(torch.load(best_path))
    final_ret, final_succ, final_len = evaluate(actor, n_episodes=50, seed=777)
    summary = {
        "algo": algo, "run_id": run_id, "eval_return": final_ret, "eval_success": final_succ,
        "eval_length": final_len, "episodes": logger.episodes,
        "frames_to_converge": frames_to_first_success(logger.rows, threshold=15.0),
        "train_time_s": logger.rows[-1]["time_s"],
    }
    logger.save(summary)
    if verbose:
        print(f"[{algo.upper()} run {run_id}] FINAL eval_return={final_ret:.2f} success={final_succ:.0%}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=list(CONFIGS), required=True)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--runs", type=int, nargs="*")
    p.add_argument("--frames", type=int)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args()
    if a.sweep:
        from training.sweep import sweep
        sweep(a.algo, a.runs, a.frames, a.workers)
    else:
        cfg = CONFIGS[a.algo](**({"total_frames": a.frames} if a.frames else {}))
        print(json.dumps(train_pg(a.algo, cfg), indent=2))


if __name__ == "__main__":
    main()
