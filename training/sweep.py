"""Hyperparameter sweep runner shared by DQN and policy-gradient trainers.

    uv run python -m training.sweep --algo all --workers 6
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil

import pandas as pd

from training.common import LOG_DIR, MODEL_DIR

ALGOS = ["dqn", "reinforce", "ppo", "a2c"]


def _job(args):
    algo, run_id, overrides, frames = args
    import torch

    torch.set_num_threads(1)
    if frames:
        overrides = {**overrides, "total_frames": frames}
    if algo == "dqn":
        from training.dqn_training import DQNConfig, train_dqn
        return {**train_dqn(DQNConfig(**overrides), run_id, verbose=True), "changes": json.dumps(overrides)}
    from training.pg_training import CONFIGS, train_pg
    return {**train_pg(algo, CONFIGS[algo](**overrides), run_id, verbose=True), "changes": json.dumps(overrides)}


def sweep_table(algo: str) -> list[dict]:
    if algo == "dqn":
        from training.dqn_training import SWEEP
        return SWEEP
    from training.pg_training import SWEEPS
    return SWEEPS[algo]


def summarise(algo: str):
    """Collect every run's json into logs/<algo>_sweep.csv and copy the best model."""
    rows = []
    table = sweep_table(algo)
    for f in sorted((LOG_DIR / algo).glob("run_*.json")):
        d = json.loads(f.read_text())
        cfg = d.pop("config")
        ch = table[d["run_id"]] if d["run_id"] < len(table) else {}
        d["changes"] = ", ".join(f"{k}={v}" for k, v in ch.items()) or "baseline"
        rows.append({**d, **{k: (str(v) if isinstance(v, (list, tuple)) else v) for k, v in cfg.items()}})
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("run_id")
    df.to_csv(LOG_DIR / f"{algo}_sweep.csv", index=False)
    best = df.sort_values(["eval_success", "eval_return"], ascending=False).iloc[0]
    rid = int(best["run_id"])
    src = MODEL_DIR / ("dqn" if algo == "dqn" else "pg") / (
        f"run_{rid:02d}.pt" if algo == "dqn" else f"{algo}_run_{rid:02d}.pt")
    dst = MODEL_DIR / ("dqn" if algo == "dqn" else "pg") / f"{algo}_best.pt"
    shutil.copy(src, dst)
    (dst.with_suffix(".json")).write_text((LOG_DIR / algo / f"run_{rid:02d}.json").read_text())
    print(f"[{algo}] best run {rid}: eval_return={best['eval_return']:.2f} "
          f"success={best['eval_success']:.0%} -> {dst.relative_to(MODEL_DIR.parent)}")
    return df


def sweep(algo: str, runs: list[int] | None = None, frames: int | None = None, workers: int = 1):
    algos = ALGOS if algo == "all" else [algo]
    jobs = []
    for a in algos:
        table = sweep_table(a)
        for rid in (runs if runs else range(len(table))):
            jobs.append((a, rid, table[rid], frames))
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            for r in pool.imap_unordered(_job, jobs):
                print(f"DONE {r['algo']} run {r['run_id']}: return={r['eval_return']:.2f} "
                      f"success={r['eval_success']:.0%} ({r['train_time_s']:.0f}s)", flush=True)
    else:
        for j in jobs:
            _job(j)
    for a in algos:
        summarise(a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="all", choices=ALGOS + ["all"])
    p.add_argument("--runs", type=int, nargs="*")
    p.add_argument("--frames", type=int)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    p.add_argument("--summarise-only", action="store_true")
    a = p.parse_args()
    if a.summarise_only:
        for al in (ALGOS if a.algo == "all" else [a.algo]):
            summarise(al)
    else:
        sweep(a.algo, a.runs, a.frames, a.workers)


if __name__ == "__main__":
    main()
