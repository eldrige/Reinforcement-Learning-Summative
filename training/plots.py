"""Figures + summary tables for the report (reads logs/, writes assets/ and logs/results_summary.md).

    uv run python -m training.plots
"""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from training.common import LOG_DIR, MODEL_DIR, ROOT, evaluate, make_env, make_policy, make_q_actor

warnings.filterwarnings("ignore")
ASSETS = ROOT / "assets"
ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COLOR = {"dqn": "#2a78d6", "reinforce": "#eb6834", "ppo": "#1baf7a", "a2c": "#eda100"}

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6, "axes.edgecolor": "#9a9a96",
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2,
})


def best_run(algo):
    meta = json.loads((MODEL_DIR / ("dqn" if algo == "dqn" else "pg") / f"{algo}_best.json").read_text())
    df = pd.read_csv(LOG_DIR / algo / f"run_{meta['run_id']:02d}.csv")
    return meta, df


def smooth(s, w=20):
    return s.astype(float).interpolate(limit_direction="both").rolling(w, min_periods=1).mean()


def fig_rewards():
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True, sharey=True)
    for ax, a in zip(axes.flat, ALGOS):
        meta, df = best_run(a)
        w = 50 if a in ("dqn", "a2c") else 10
        ax.plot(df["frames"], df["mean_return"], color=COLOR[a], alpha=0.18, linewidth=1)
        ax.plot(df["frames"], smooth(df["mean_return"], w), color=COLOR[a], label=LABEL[a])
        ax.axhline(0, color="#9a9a96", linewidth=0.8)
        ax.set_title(f"{LABEL[a]} (best run {meta['run_id']})")
    for ax in axes[1]:
        ax.set_xlabel("environment steps")
    for ax in axes[:, 0]:
        ax.set_ylabel("episode return")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_reward_curves.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for a in ALGOS:
        _, df = best_run(a)
        w = 50 if a in ("dqn", "a2c") else 10
        y = smooth(df["mean_return"], w)
        ax.plot(df["frames"], y, color=COLOR[a], label=LABEL[a])
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return (smoothed)")
    ax.set_title("Training return, best configuration per algorithm")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_reward_compare.png")


def fig_stability():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    _, df = best_run("dqn")
    axes[0].plot(df["frames"], smooth(df["loss"], 50), color=COLOR["dqn"])
    axes[0].set_title("DQN TD loss (smooth L1)")
    axes[0].set_xlabel("environment steps")
    axes[0].set_ylabel("loss")
    for a in ["reinforce", "ppo", "a2c"]:
        _, df = best_run(a)
        axes[1].plot(df["frames"], smooth(df["entropy"], 10 if a != "a2c" else 50), color=COLOR[a], label=LABEL[a])
    axes[1].set_title("Policy entropy (PG methods)")
    axes[1].set_xlabel("environment steps")
    axes[1].set_ylabel("entropy (nats)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_stability.png")


def fig_eval():
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for a in ALGOS:
        _, df = best_run(a)
        d = df.dropna(subset=["eval_success"])
        ax.plot(d["frames"], d["eval_success"] * 100, color=COLOR[a], marker="o", markersize=3, label=LABEL[a])
    ax.set_xlabel("environment steps")
    ax.set_ylabel("greedy success rate (%)")
    ax.set_title("Mission success during training (10 evaluation episodes per point)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_eval_success.png")


def fig_sweeps():
    fig, axes = plt.subplots(1, 4, figsize=(11, 3), sharey=True)
    for ax, a in zip(axes, ALGOS):
        df = pd.read_csv(LOG_DIR / f"{a}_sweep.csv")
        ax.bar(df["run_id"].astype(str), df["eval_return"], color=COLOR[a], width=0.7)
        ax.axhline(0, color="#9a9a96", linewidth=0.8)
        ax.set_title(LABEL[a])
        ax.set_xlabel("sweep run")
    axes[0].set_ylabel("final eval return (50 ep.)")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_sweeps.png")


def generalisation(n=100):
    rows = []
    for a in ALGOS:
        meta, _ = best_run(a)
        env = make_env()
        hidden = tuple(meta["config"]["hidden"])
        agent = make_q_actor(env, hidden) if a == "dqn" else make_policy(env, hidden)
        agent.load_state_dict(torch.load(MODEL_DIR / ("dqn" if a == "dqn" else "pg") / f"{a}_best.pt"))
        for name, rs in [("unseen risk draws, clinic start", False), ("unseen risk draws, random start", True)]:
            ret, succ, length = evaluate(agent, n_episodes=n, seed=31337, random_start=rs)
            rows.append({"algo": LABEL[a], "test": name, "mean_return": round(ret, 2),
                         "success_rate": succ, "mean_steps": round(length, 1)})
    df = pd.DataFrame(rows)
    df.to_csv(LOG_DIR / "generalisation.csv", index=False)
    return df


def summary_md(gen):
    lines = ["# Results summary\n"]
    for a in ALGOS:
        df = pd.read_csv(LOG_DIR / f"{a}_sweep.csv")
        cols = [c for c in ["run_id", "changes", "eval_return", "eval_success", "eval_length",
                            "frames_to_converge", "episodes", "train_time_s"] if c in df.columns]
        if "changes" not in df.columns:
            cols = [c for c in df.columns if c not in ("algo", "seed", "total_frames")]
        lines.append(f"## {LABEL[a]}\n")
        lines.append(df[cols].round(2).to_markdown(index=False))
        lines.append("")
    lines.append("## Generalisation (best model per algorithm, 100 unseen episodes)\n")
    lines.append(gen.to_markdown(index=False))
    (LOG_DIR / "results_summary.md").write_text("\n".join(lines))


def main():
    ASSETS.mkdir(exist_ok=True)
    fig_rewards()
    fig_stability()
    fig_eval()
    fig_sweeps()
    gen = generalisation()
    print(gen.to_string(index=False))
    summary_md(gen)
    print("figures in assets/, tables in logs/results_summary.md")


if __name__ == "__main__":
    main()
