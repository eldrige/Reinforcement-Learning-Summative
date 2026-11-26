"""Shared helpers: TorchRL env construction, evaluation, logging."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch.distributions import OneHotCategorical
from torchrl.envs import GymWrapper, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, QValueActor, ValueOperator

from environment.custom_env import MAX_STEPS, CommunityHealthEnv

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"


def make_env(seed: int | None = None, render_mode: str | None = None, renderer: str = "opengl",
             random_start: bool = False):
    """CommunityHealthEnv wrapped for TorchRL (one-hot actions, episode stats)."""
    base = GymWrapper(CommunityHealthEnv(render_mode=render_mode, renderer=renderer, random_start=random_start))
    env = TransformedEnv(base)
    env.append_transform(StepCounter())
    env.append_transform(RewardSum())
    if seed is not None:
        env.set_seed(seed)
    return env


# ------------------------------------------------------------------ networks
def make_mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...], act=torch.nn.ReLU):
    return MLP(in_features=in_dim, out_features=out_dim, num_cells=list(hidden), activation_class=act)


def make_q_actor(env, hidden):
    obs_dim = env.observation_spec["observation"].shape[-1]
    n_act = env.action_spec.shape[-1]
    return QValueActor(make_mlp(obs_dim, n_act, hidden), in_keys=["observation"], spec=env.action_spec)


def make_policy(env, hidden, act=torch.nn.Tanh):
    obs_dim = env.observation_spec["observation"].shape[-1]
    n_act = env.action_spec.shape[-1]
    module = TensorDictModule(make_mlp(obs_dim, n_act, hidden, act), in_keys=["observation"], out_keys=["logits"])
    return ProbabilisticActor(
        module, spec=env.action_spec, in_keys=["logits"], distribution_class=OneHotCategorical,
        return_log_prob=True, default_interaction_type=ExplorationType.RANDOM,
    )


def make_critic(env, hidden, act=torch.nn.Tanh):
    obs_dim = env.observation_spec["observation"].shape[-1]
    return ValueOperator(make_mlp(obs_dim, 1, hidden, act), in_keys=["observation"])


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(policy, n_episodes: int = 20, seed: int = 10_000, deterministic: bool = True,
             random_start: bool = False):
    """Roll out the policy; returns mean return, success rate, mean episode length."""
    env = make_env(seed=seed, random_start=random_start)
    returns, success, lengths = [], [], []
    mode = ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
    with set_exploration_type(mode):
        for _ in range(n_episodes):
            td = env.rollout(MAX_STEPS + 1, policy, break_when_any_done=True)
            returns.append(td["next", "reward"].sum().item())
            lengths.append(td.shape[-1])
            success.append(bool(td["next", "terminated"][-1].item()))
    env.close()
    return float(np.mean(returns)), float(np.mean(success)), float(np.mean(lengths))


def finished_episode_returns(data) -> list[float]:
    done = data["next", "done"].squeeze(-1)
    return data["next", "episode_reward"].squeeze(-1)[done].tolist()


class RunLogger:
    """Collects training curves (frames vs episode return / losses) for one run."""

    def __init__(self, algo: str, run_id: int, config: dict):
        self.algo, self.run_id, self.config = algo, run_id, config
        self.rows: list[dict] = []
        self.t0 = time.time()
        self.episodes = 0

    def log(self, frames: int, ep_returns: list[float], **metrics):
        self.episodes += len(ep_returns)
        row = {"frames": frames, "episodes": self.episodes,
               "mean_return": float(np.mean(ep_returns)) if ep_returns else np.nan,
               "time_s": time.time() - self.t0}
        row.update({k: float(v) for k, v in metrics.items()})
        self.rows.append(row)

    def save(self, summary: dict):
        import pandas as pd

        d = LOG_DIR / self.algo
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(d / f"run_{self.run_id:02d}.csv", index=False)
        (d / f"run_{self.run_id:02d}.json").write_text(json.dumps({"config": self.config, **summary}, indent=2))


def frames_to_first_success(rows: list[dict], threshold: float) -> int | None:
    """Frames until the running-mean training return first exceeds ``threshold`` (convergence proxy)."""
    import pandas as pd

    if not rows:
        return None
    s = pd.DataFrame(rows).set_index("frames")["mean_return"].ffill().rolling(5, min_periods=1).mean()
    hit = s[s >= threshold]
    return int(hit.index[0]) if len(hit) else None
