"""Đánh giá lại các model HIỆN HÀNH (models/zombie_lv*.zip) bằng code hiện hành, so với 3 baseline.
Dùng khi config / luật đã đổi sau lần ghi models/eval.csv.

Chạy:  python tools/eval_current.py                 (cả 5 màn)
       python tools/eval_current.py --levels 1,2    (một vài màn)
Ra:    models/eval_current.csv (cột giống eval.csv)
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from stable_baselines3 import PPO

from config import MODEL_DIR
from train import BASELINES, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,2,3,4,5")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(MODEL_DIR, "eval_current.csv"))
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    rows = []
    for lv in [int(x) for x in args.levels.split(",")]:
        model = PPO.load(os.path.join(MODEL_DIR, f"zombie_lv{lv}.zip"), device="cpu")
        pols = dict(BASELINES)
        pols["ppo"] = lambda o, e, m=model: int(m.predict(o, deterministic=True)[0])
        for name, pol in pols.items():
            r = evaluate(pol, lv, args.episodes)
            rows.append([lv, name, args.episodes] + [f"{r[k]:.3f}" for k in
                        ("mean_reward", "catch", "killed", "timeout", "aimed", "stuck", "hp_lost", "hits")])
            print(lv, name, rows[-1][3:5], flush=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "policy", "episodes", "mean_reward", "catch", "killed", "timeout", "aimed", "stuck", "hp_lost", "hits"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
