"""Run the best trained agent in the Community Health Outreach simulation.

    uv run main.py                         # best agent overall, 3D OpenGL window + verbose terminal log
    uv run main.py --algo ppo --episodes 3
    uv run main.py --algo random           # untrained random agent (environment demo)
    uv run main.py --compare               # evaluate all four best models (no GUI)
    uv run main.py --algo a2c --gif assets/best_agent.gif
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from environment.custom_env import ACTION_NAMES, HOUSEHOLDS, MAX_STEPS
from training.common import MODEL_DIR, evaluate, make_env, make_policy, make_q_actor

ALGOS = ["dqn", "reinforce", "ppo", "a2c"]


def model_path(algo: str):
    return MODEL_DIR / ("dqn" if algo == "dqn" else "pg") / f"{algo}_best.pt"


def load_agent(algo: str, env):
    path = model_path(algo)
    if not path.exists():
        sys.exit(f"No trained model at {path}. Train first: uv run python -m training.sweep --algo {algo}")
    meta = json.loads(path.with_suffix(".json").read_text())
    hidden = tuple(meta["config"]["hidden"])
    agent = make_q_actor(env, hidden) if algo == "dqn" else make_policy(env, hidden)
    agent.load_state_dict(torch.load(path))
    agent.eval()
    return agent, meta


def best_overall() -> str:
    scores = {}
    for a in ALGOS:
        p = model_path(a).with_suffix(".json")
        if p.exists():
            m = json.loads(p.read_text())
            scores[a] = (m["eval_success"], m["eval_return"])
    if not scores:
        sys.exit("No trained models found. Run: uv run python -m training.sweep --algo all")
    return max(scores, key=scores.get)


def run(algo: str, episodes: int, renderer: str, seed: int, gif: str | None, render: bool):
    mode = "rgb_array" if gif else ("human" if render else None)
    env = make_env(seed=seed, render_mode=mode, renderer=renderer)
    base = env.base_env._env
    agent = None
    if algo != "random":
        agent, meta = load_agent(algo, env)
        print(f"Loaded {algo.upper()} (sweep run {meta['run_id']}): eval_return={meta['eval_return']:.2f} "
              f"success={meta['eval_success']:.0%}  config={meta['config']}")
    frames, stats = [], []
    for ep in range(episodes):
        td = env.reset()
        print(f"\n=== Episode {ep + 1}/{episodes} | algo={algo.upper()} | high-risk households this day: "
              f"{[f'H{i}' for i in np.flatnonzero(base.high_risk)]} (hidden from agent) ===")
        print(f"{'step':>4} {'pos':>7} {'kits':>4}  {'action':<17} {'reward':>7} {'return':>8}  event")
        done = False
        while not done:
            if agent is None:
                td = env.rand_action(td)
            else:
                with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                    td = agent(td)
            td = env.step(td)
            a = int(td["action"].argmax())
            r = float(td["next", "reward"])
            print(f"{base.t:>4} {str(tuple(base.pos)):>7} {base.kits:>4}  {ACTION_NAMES[a]:<17} {r:>+7.2f} "
                  f"{base.episode_return:>8.2f}  {base.last_event}", flush=True)
            if gif:
                frames.append(base.render())
            done = bool(td["next", "done"])
            td = td["next"]
        info = base._info()
        stats.append((base.episode_return, info["success"], base.t))
        print(f"--- Episode {ep + 1}: return={base.episode_return:.2f} steps={base.t}/{MAX_STEPS} "
              f"screened={info['screened']}/{len(HOUSEHOLDS)} referred={info['referred']}/{info['high_risk_total']} "
              f"educated={info['educated']} success={info['success']}")
    if gif:
        import imageio
        imageio.mimsave(gif, frames[::1], duration=0.18, loop=0)
        print(f"Saved {gif}")
    env.close()
    r, s, l = map(np.mean, zip(*stats))
    print(f"\nSummary over {episodes} episode(s): mean return={r:.2f} success rate={s:.0%} mean steps={l:.1f}")


def compare(n: int):
    print(f"{'algo':<10} {'mean return':>12} {'success':>8} {'steps':>6}   ({n} unseen episodes, greedy policy)")
    for a in ALGOS:
        if not model_path(a).exists():
            continue
        env = make_env()
        agent, _ = load_agent(a, env)
        ret, succ, length = evaluate(agent, n_episodes=n, seed=2024)
        print(f"{a:<10} {ret:>12.2f} {succ:>8.0%} {length:>6.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=ALGOS + ["random", "best"], default="best")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--renderer", choices=["opengl", "2d"], default="opengl")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gif", help="save the run as a GIF instead of opening a window")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--compare", action="store_true", help="evaluate all trained agents, no GUI")
    a = p.parse_args()
    os.environ["RENDER_FPS"] = str(a.fps)
    if a.compare:
        compare(100)
        return
    algo = best_overall() if a.algo == "best" else a.algo
    run(algo, a.episodes, a.renderer, a.seed, a.gif, not a.no_render)


if __name__ == "__main__":
    main()
