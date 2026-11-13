"""So sánh nhiều thuật toán RL cho zombie trên CÙNG curriculum 5 màn.

Mỗi thuật toán: cùng env (env_zombie.py), cùng reward, cùng mạng 64x64, cùng số bước mỗi màn
(LEVELS[...]["train_steps"] x --scale), màn sau học tiếp từ model màn trước của CHÍNH thuật toán đó.

"Qua màn" (gate): sau khi train màn k, đánh giá 100 episode (deterministic, seed 10000+i)
và so với baseline "path" (đi theo đường ngắn nhất, vào tầm là đánh) trên CÙNG các seed:
    qua màn k  <=>  return trung bình >= return của path
"Đạt tới màn" = số màn qua LIÊN TIẾP tính từ màn 1 (giống người chơi: kẹt ở màn nào thì dừng ở đó).
Thuật toán vẫn được train tiếp các màn sau để có đủ bảng số liệu.

Chạy:
    python compare_algos.py --baselines                 # đo baseline 1 lần -> models/algo_baselines.csv
    python compare_algos.py --algos ppo,a2c --seeds 0,1 # train + đánh giá -> models/algo_compare.csv
    python compare_algos.py --report                    # in bảng tổng kết + vẽ models/algo_compare.png
Thử cấu hình DQN khác (ghi ra file riêng, không đụng bảng so sánh chính):
    python compare_algos.py --tag tune --algos dqn,dqn_eps,dqn_buf,dqn_tgt --seeds 10,11,12 --max-level 4
Model lưu ở models/algos/<algo>/seed<s>/zombie_lv<k>.zip (xem trong game: python game.py --zombie-algo a2c)
"""
import argparse
import csv
import os
import shutil
import time

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env

from config import MODEL_DIR, LEVELS
from env_zombie import ZombieEnv
from train import evaluate, BASELINES

try:
    from sb3_contrib import TRPO
except ImportError:          # pip install sb3-contrib
    TRPO = None

ALGO_DIR = os.path.join(MODEL_DIR, "algos")
CMP_CSV = os.path.join(MODEL_DIR, "algo_compare.csv")
BASE_CSV = os.path.join(MODEL_DIR, "algo_baselines.csv")
EP_CSV = os.path.join(MODEL_DIR, "algo_episodes.csv")      # reward + thắng/thua của TỪNG episode
OLD_CSV = os.path.join(MODEL_DIR, "algo_compare_100ep.csv")  # lần chạy trước (100 episode): lấy lại thời gian train
METRICS = ("mean_reward", "catch", "killed", "timeout", "aimed", "stuck", "hp_lost", "hits", "dodge")

# Mỗi thuật toán: (class, số env song song, tham số). Mạng mặc định của SB3 cho cả 4 là MLP 64x64.
ALGOS = {
    # On-policy, policy gradient có "kẹp" tỉ lệ xác suất (clip 0.2), 10 epoch trên mỗi rollout
    "ppo": (PPO, 8, dict(n_steps=512, batch_size=256, ent_coef=0.01)),
    # On-policy, actor-critic đồng bộ, 1 lần cập nhật cho mỗi rollout ngắn (5 bước), không giới hạn bước nhảy
    "a2c": (A2C, 8, dict(ent_coef=0.01)),
    # On-policy, trust region "cứng": mỗi cập nhật giữ KL(cũ||mới) <= 0.01 (conjugate gradient + line search)
    "trpo": (TRPO, 8, dict(n_steps=512, batch_size=256)),
    # Off-policy, value-based: học Q(s,a) từ replay buffer, chọn action bằng epsilon-greedy
    "dqn": (DQN, 1, dict(buffer_size=200_000, learning_starts=5_000, batch_size=64, train_freq=4,
                         target_update_interval=5_000, exploration_fraction=0.2,
                         exploration_final_eps=0.05)),
}

# Các biến thể DQN để thử: (thuật toán gốc, tham số đổi, cách học tiếp khi sang màn mới).
# Mỗi biến thể thêm MỘT thay đổi so với biến thể trước để thấy tác dụng riêng của từng thay đổi.
#   eps0/eps_frac : sang màn mới, epsilon bắt đầu từ eps0 (mặc định SB3 là 1.0 = đi ngẫu nhiên lại từ đầu)
#   keep_buffer   : mang replay buffer của màn trước sang (mặc định buffer bị bỏ, phải gom lại 5000 bước)
#   l1_from       : màn 1 giống hệt cấu hình đó nên lấy lại model màn 1 của nó, khỏi train lại
VARIANTS = {
    "dqn_eps": ("dqn", {}, dict(eps0=0.3, eps_frac=0.1, l1_from="dqn_buf")),
    "dqn_buf": ("dqn", {}, dict(eps0=0.3, eps_frac=0.1, keep_buffer=True)),
    "dqn_tgt": ("dqn", dict(target_update_interval=1_000), dict(eps0=0.3, eps_frac=0.1, keep_buffer=True)),
}
VARIANTS["dqn"] = ("dqn", {}, dict(l1_from="dqn_buf"))      # chỉ dùng khi --tag (mặc định: không đổi gì)


def resolve(algo, tagged):
    """-> (class, số env, tham số, tuỳ chọn học tiếp)"""
    if algo in VARIANTS and (tagged or algo != "dqn"):
        base, extra, warm = VARIANTS[algo]
        cls, n_envs, kw = ALGOS[base]
        return cls, n_envs, {**kw, **extra}, warm
    cls, n_envs, kw = ALGOS[algo]
    return cls, n_envs, kw, {}


def buffer_path(algo, seed, lv):
    return os.path.join(ALGO_DIR, algo, f"seed{seed}", f"buffer_lv{lv}.pkl")


def model_path(algo, seed, lv):
    return os.path.join(ALGO_DIR, algo, f"seed{seed}", f"zombie_lv{lv}.zip")


def append(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def log_episodes(algo, seed, lv, r):
    new = not os.path.exists(EP_CSV)
    with open(EP_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["algo", "seed", "level", "episode", "reward", "win"])
        for i, (rew, win) in enumerate(zip(r["ep_rewards"], r["ep_wins"])):
            w.writerow([algo, seed, lv, i, f"{rew:.4f}", win])


def run_baselines(episodes):
    for lv in range(1, len(LEVELS) + 1):
        for name in ("chase", "path", "zigzag"):
            r = evaluate(BASELINES[name], lv, episodes)
            log_episodes(name, -1, lv, r)
            append(BASE_CSV, ["level", "policy"] + list(METRICS), [lv, name] + [f"{r[k]:.4f}" for k in METRICS])
            print(f"màn {lv} {name:6} return {r['mean_reward']:.2f}  thắng {r['catch']:.0%}")


def read_baselines():
    out = {}
    with open(BASE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["level"]), row["policy"]] = {k: float(row[k]) for k in METRICS}
    return out


def run_algo(algo, seed, scale, episodes, max_level=None, tagged=False):
    cls, n_envs, kw, warm = resolve(algo, tagged)
    if cls is None:
        raise SystemExit("Cần sb3-contrib cho TRPO: pip install sb3-contrib")
    base = read_baselines()
    done = set()                       # chạy tiếp từ chỗ dừng: màn đã có kết quả + model thì bỏ qua
    if os.path.exists(CMP_CSV):
        with open(CMP_CSV, encoding="utf-8") as f:
            done = {(r["algo"], int(r["seed"]), int(r["level"])) for r in csv.DictReader(f)}
    old_time = {}
    if os.path.exists(OLD_CSV):
        with open(OLD_CSV, encoding="utf-8") as f:
            old_time = {(r["algo"], int(r["seed"]), int(r["level"])): float(r["train_s"]) for r in csv.DictReader(f)}
    for lv in range(1, (max_level or len(LEVELS)) + 1):
        steps = int(LEVELS[lv - 1]["train_steps"] * scale)
        path = model_path(algo, seed, lv)
        src = warm.get("l1_from")
        if lv == 1 and src and not os.path.exists(path) and os.path.exists(model_path(src, seed, 1)):
            os.makedirs(os.path.dirname(path), exist_ok=True)       # cùng cấu hình màn 1 -> dùng lại model
            shutil.copy(model_path(src, seed, 1), path)
            print(f"[{algo} seed{seed}] màn 1: dùng lại model màn 1 của {src}", flush=True)
        if (algo, seed, lv) in done and os.path.exists(path):
            print(f"[{algo} seed{seed}] màn {lv}: đã có, bỏ qua", flush=True)
            continue
        if os.path.exists(path):
            # model đã train (lần chạy trước) nhưng chưa đánh giá với số episode này: chỉ đánh giá lại
            model = cls.load(path, device="cpu")
            train_s = old_time.get((algo, seed, lv), float("nan"))
        else:
            env = make_vec_env(lambda lv=lv: ZombieEnv(level=lv), n_envs=n_envs, seed=seed + 100 * lv)
            prev = model_path(algo, seed, lv - 1)
            if lv > 1 and os.path.exists(prev):
                over = {}
                if "eps0" in warm:
                    over.update(exploration_initial_eps=warm["eps0"], exploration_fraction=warm["eps_frac"])
                keep = warm.get("keep_buffer") and os.path.exists(buffer_path(algo, seed, lv - 1))
                if keep:
                    over["learning_starts"] = 0        # buffer đã có dữ liệu, học được ngay
                model = cls.load(prev, env=env, device="cpu", **over)  # curriculum: học tiếp từ màn trước
                if keep:
                    model.load_replay_buffer(buffer_path(algo, seed, lv - 1))
            else:
                model = cls("MlpPolicy", env, seed=seed, device="cpu", verbose=0, **kw)
            t0 = time.time()
            model.learn(total_timesteps=steps, reset_num_timesteps=True)
            train_s = time.time() - t0
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model.save(path)
            if warm.get("keep_buffer"):
                model.save_replay_buffer(buffer_path(algo, seed, lv))
                if os.path.exists(buffer_path(algo, seed, lv - 1)):
                    os.remove(buffer_path(algo, seed, lv - 1))     # buffer cũ không cần nữa (~50 MB)

        r = evaluate(lambda o, e: int(model.predict(o, deterministic=True)[0]), lv, episodes)
        log_episodes(algo, seed, lv, r)
        p = base[lv, "path"]
        passed = r["mean_reward"] >= p["mean_reward"]
        append(CMP_CSV, ["algo", "seed", "level", "steps", "train_s"] + list(METRICS) + ["path_reward", "path_catch", "passed"],
               [algo, seed, lv, steps, f"{train_s:.0f}"] + [f"{r[k]:.4f}" for k in METRICS]
               + [f"{p['mean_reward']:.4f}", f"{p['catch']:.4f}", int(passed)])
        print(f"[{algo} seed{seed}] màn {lv}: return {r['mean_reward']:.2f} vs path {p['mean_reward']:.2f} "
              f"-> {'QUA' if passed else 'KHÔNG QUA'}  (thắng {r['catch']:.0%}, né {r['dodge']:.0%}, {train_s:.0f}s)",
              flush=True)


def report():
    import collections
    rows = list(csv.DictReader(open(CMP_CSV, encoding="utf-8")))
    base = read_baselines()
    by = collections.defaultdict(dict)
    for r in rows:
        by[r["algo"], int(r["seed"])][int(r["level"])] = r
    print(f"{'thuật toán':10} {'seed':>4}  {'đạt tới màn':>11}  qua các màn   return từng màn")
    reach = collections.defaultdict(list)
    for (algo, seed), lvs in sorted(by.items()):
        k = 0
        while k + 1 in lvs and lvs[k + 1]["passed"] == "1":
            k += 1
        reach[algo].append(k)
        passed = "".join("✓" if lvs.get(l, {}).get("passed") == "1" else "·" for l in range(1, 6))
        rets = " ".join(f"{float(lvs[l]['mean_reward']):5.2f}" if l in lvs else "  -  " for l in range(1, 6))
        print(f"{algo:10} {seed:>4}  {k:>11}  {passed:11}  {rets}")
    print("path baseline                              " + " ".join(f"{base[l, 'path']['mean_reward']:5.2f}" for l in range(1, 6)))
    _plot(by, base)
    return reach


def _plot(by, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    algos = [a for a in ALGOS if any(k[0] == a for k in by)]
    cols = {"ppo": "#e8453c", "a2c": "#3b7dd8", "trpo": "#2a9d6f", "dqn": "#c98a1b"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(1, 6)
    w = 0.8 / (len(algos) + 1)
    for i, a in enumerate(algos):
        seeds = [s for (al, s) in by if al == a]
        for ax, key in ((axes[0], "mean_reward"), (axes[1], "catch")):
            vals = np.array([[float(by[a, s][l][key]) if l in by[a, s] else np.nan for l in x] for s in seeds])
            ax.bar(x - 0.4 + (i + 0.5) * w, np.nanmean(vals, 0), w, color=cols.get(a), label=a.upper())
            if len(seeds) > 1:
                ax.errorbar(x - 0.4 + (i + 0.5) * w, np.nanmean(vals, 0),
                            yerr=[np.nanmean(vals, 0) - np.nanmin(vals, 0), np.nanmax(vals, 0) - np.nanmean(vals, 0)],
                            fmt="none", ecolor="#333", capsize=2, lw=1)
    for ax, key, t in ((axes[0], "mean_reward", "Return trung bình (100 episode)"),
                       (axes[1], "catch", "Tỉ lệ thắng (áp sát / hạ người chơi giả)")):
        ax.bar(x - 0.4 + (len(algos) + 0.5) * w, [base[l, "path"][key] for l in x], w, color="#999", label="path")
        ax.set_xticks(x, [f"Màn {l}" for l in x])
        ax.set_title(t)
        ax.axhline(0, color="#555", lw=0.6)
    axes[0].legend(ncol=5, fontsize=9)
    fig.suptitle("So sánh thuật toán trên cùng curriculum (cột = trung bình các seed, vạch = min-max)")
    fig.tight_layout()
    out = os.path.join(MODEL_DIR, "algo_compare.png")
    fig.savefig(out, dpi=130)
    print("Đã vẽ", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", default="ppo,a2c,trpo,dqn")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--tag", default="", help="ghi kết quả ra algo_compare_<tag>.csv thay vì bảng chính")
    ap.add_argument("--max-level", type=int, default=None, help="chỉ chạy màn 1..N")
    args = ap.parse_args()
    global CMP_CSV, EP_CSV
    if args.tag:
        CMP_CSV = os.path.join(MODEL_DIR, f"algo_compare_{args.tag}.csv")
        EP_CSV = os.path.join(MODEL_DIR, f"algo_episodes_{args.tag}.csv")
    try:
        import torch
        torch.set_num_threads(1)      # chạy nhiều tiến trình song song không giành nhau CPU
    except ImportError:
        pass
    if args.baselines:
        run_baselines(args.episodes)
    elif args.report:
        report()
    else:
        for algo in args.algos.split(","):
            for s in args.seeds.split(","):
                run_algo(algo.strip(), int(s), args.scale, args.episodes, args.max_level, bool(args.tag))


if __name__ == "__main__":
    main()
