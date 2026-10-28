"""Vẽ đường học (learning curve) thật của zombie PPO: train từ đầu ở một màn, cứ mỗi `--every` bước thì
đánh giá policy (deterministic) trên 100 episode cố định, ghi cùng mốc của 3 baseline.

Chạy:  python tools/learning_curve.py --level 1 --steps 250000 --seed 0
Ra:    models/learning_curve_lv1_seed0.csv  (không đụng vào models/zombie_lv*.zip)
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from config import MODEL_DIR
from env_zombie import ZombieEnv
from train import PPO_KW, BASELINES, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--steps", type=int, default=250_000)
    ap.add_argument("--every", type=int, default=20_480)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    lv = args.level
    out = os.path.join(MODEL_DIR, f"learning_curve_lv{lv}_seed{args.seed}.csv")
    env = make_vec_env(lambda: ZombieEnv(level=lv), n_envs=args.envs, seed=args.seed)
    model = PPO("MlpPolicy", env, seed=args.seed, **PPO_KW)
    pol = lambda o, e: int(model.predict(o, deterministic=True)[0])

    rows = []
    base = {name: evaluate(p, lv, args.episodes) for name, p in BASELINES.items()}
    print({k: round(v["mean_reward"], 2) for k, v in base.items()}, flush=True)

    def record(steps):
        r = evaluate(pol, lv, args.episodes)
        buf = [ep["r"] for ep in (model.ep_info_buffer or [])]
        train_r = float(np.mean(buf)) if buf else float("nan")
        rows.append(dict(steps=steps, eval_reward=r["mean_reward"], catch=r["catch"], killed=r["killed"],
                         timeout=r["timeout"], train_ep_rew_mean=train_r))
        print(f"{steps:>8} eval_reward {r['mean_reward']:.3f} catch {r['catch']:.2f} train_ep_rew {train_r:.3f}", flush=True)

    t0 = time.time()
    record(0)
    done = 0
    while done < args.steps:
        model.learn(total_timesteps=args.every, reset_num_timesteps=False)
        done = model.num_timesteps
        record(done)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["steps", "eval_reward", "catch", "killed", "timeout", "train_ep_rew_mean",
                    "random_reward", "chase_reward", "path_reward", "random_catch", "chase_catch", "path_catch"])
        for r in rows:
            w.writerow([r["steps"], f"{r['eval_reward']:.4f}", f"{r['catch']:.3f}", f"{r['killed']:.3f}", f"{r['timeout']:.3f}",
                        f"{r['train_ep_rew_mean']:.4f}"] + [f"{base[n]['mean_reward']:.4f}" for n in ("random", "chase", "path")]
                       + [f"{base[n]['catch']:.3f}" for n in ("random", "chase", "path")])
    print(f"Xong sau {(time.time() - t0) / 60:.1f} phút -> {out}")


if __name__ == "__main__":
    main()
