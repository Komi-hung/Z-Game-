"""Train zombie theo curriculum: màn 1 -> 5, mỗi màn tiếp tục từ checkpoint của màn trước.

Chạy tất cả:          python train.py
Chỉ một màn:          python train.py --level 5
Thử nhanh:            python train.py --scale 0.1    (10% số bước trong config)
Chỉnh nhẹ số liệu:    python train.py --finetune --scale 0.4
                      (mỗi màn học tiếp từ chính model của màn đó, không train lại từ đầu)

Kết quả: models/zombie_lv1.zip ... zombie_lv5.zip và bảng so sánh models/eval.csv
"""
import argparse
import csv
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from config import MODEL_DIR, LEVELS
from env_zombie import ZombieEnv
from zombie_brain import baseline_action


PPO_KW = dict(n_steps=512, batch_size=256, ent_coef=0.01, verbose=1, device="cpu")


def load_model(path, env, seed=0):
    """Nạp model; nếu model cũ có 9 action mà env có 18 (thêm 9 action "giữ đòn")
    thì chép toàn bộ trọng số đã học sang model mới: 9 action cũ giữ nguyên, 9 action
    giữ đòn khởi đầu bằng bản sao hơi kém ưu tiên hơn (warm-start: không mất kỹ năng cũ)."""
    old = PPO.load(path, device="cpu")
    n_old, n_new = old.action_space.n, env.action_space.n
    if n_old == n_new:
        return PPO.load(path, env=env, device="cpu")
    print(f"Warm-start {path}: {n_old} -> {n_new} action")
    import torch
    new = PPO("MlpPolicy", env, seed=seed, **PPO_KW)
    old_sd, new_sd = old.policy.state_dict(), new.policy.state_dict()
    for k, v in old_sd.items():
        if k.startswith("action_net."):
            t = new_sd[k].clone()
            t[:n_old] = v
            t[n_old:2 * n_old] = v                  # bản sao cho các action giữ đòn
            if k.endswith("bias"):
                t[n_old:2 * n_old] -= 1.5           # lúc đầu ít giữ đòn, sẽ tự học khi nào nên giữ
            new_sd[k] = t
        elif new_sd[k].shape == v.shape:
            new_sd[k] = v
    new.policy.load_state_dict(new_sd)
    return new


def evaluate(policy, level, n_episodes=100, seed=10_000):
    """Chạy n episode trên bản đồ của màn này, đếm kết quả."""
    env = ZombieEnv(level=level, replay_prob=0.0)
    counts = {"caught": 0, "won": 0, "killed": 0, "timeout": 0}
    total_r = aimed = stuck = hp_lost = hits = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs, env))
            total_r += r
            done = term or trunc
        counts[info["outcome"]] += 1
        aimed += info["aimed_frac"]
        stuck += info["stuck_frac"]
        hp_lost += info["hp_lost"]
        hits += info["hits_dealt"]
    n = n_episodes
    # catch: zombie thường áp sát được / boss hạ được người chơi giả
    return dict(mean_reward=total_r / n, catch=(counts["caught"] + counts["won"]) / n,
                killed=counts["killed"] / n, timeout=counts["timeout"] / n, aimed=aimed / n,
                stuck=stuck / n, hp_lost=hp_lost / n, hits=hits / n)


BASELINES = {
    "random": lambda o, e: e.action_space.sample(),
    # đi thẳng tới người chơi, vào tầm là đánh
    "chase": lambda o, e: baseline_action("chase", e.zx, e.zy, e.bot.x, e.bot.y, e.path,
                                          e.attack.ready, e.st["attack_range"]),
    # đi theo đường ngắn nhất (biết vòng qua tường nhưng không né đạn), vào tầm là đánh
    "path": lambda o, e: baseline_action("path", e.zx, e.zy, e.bot.x, e.bot.y, e.path,
                                         e.attack.ready, e.st["attack_range"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0, help="1-5, mặc định train cả 5 màn")
    ap.add_argument("--scale", type=float, default=1.0, help="nhân số bước train trong config")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--finetune", action="store_true",
                    help="học tiếp từ model hiện có của chính màn đó (dùng khi chỉ chỉnh nhẹ config)")
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    levels = [args.level] if args.level else range(1, len(LEVELS) + 1)

    for lv in levels:
        cfg = LEVELS[lv - 1]
        steps = int(cfg["train_steps"] * args.scale)
        print(f"\n===== {cfg['name']} ({steps:,} bước) =====")
        env = make_vec_env(lambda lv=lv: ZombieEnv(level=lv), n_envs=args.envs, seed=args.seed)
        prev = os.path.join(MODEL_DIR, f"zombie_lv{lv - 1}.zip")
        own = os.path.join(MODEL_DIR, f"zombie_lv{lv}.zip")
        if args.finetune and os.path.exists(own):
            print(f"Fine-tune từ {own}")
            model = load_model(own, env, args.seed)
        elif lv > 1 and os.path.exists(prev):
            print(f"Tiếp tục học từ {prev}")
            model = load_model(prev, env, args.seed)
        else:
            model = PPO("MlpPolicy", env, seed=args.seed, **PPO_KW)
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        path = os.path.join(MODEL_DIR, f"zombie_lv{lv}.zip")
        model.save(path)
        print(f"Đã lưu {path}")

        policies = dict(BASELINES)
        policies["ppo"] = lambda o, e: int(model.predict(o, deterministic=True)[0])
        results = {name: evaluate(pol, lv) for name, pol in policies.items()}
        print("(màn boss: 'bắt được' = hạ được người chơi giả trong trận tay đôi)")
        print(f"{'policy':8} {'reward':>8} {'bắt được':>9} {'bị giết':>8} {'hết giờ':>8} {'bị ngắm':>8} {'kẹt':>6} {'mất máu':>8} {'đòn trúng':>9}")
        for name, r in results.items():
            print(f"{name:8} {r['mean_reward']:8.2f} {r['catch']:9.0%} {r['killed']:8.0%} "
                  f"{r['timeout']:8.0%} {r['aimed']:8.0%} {r['stuck']:6.0%} {r['hp_lost']:8.0%} {r['hits']:9.2f}")

        csv_path = os.path.join(MODEL_DIR, "eval.csv")
        new_file = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["level", "policy", "steps", "mean_reward", "catch", "killed", "timeout", "aimed", "stuck", "hp_lost", "hits"])
            for name, r in results.items():
                w.writerow([lv, name, model.num_timesteps] + [f"{r[k]:.3f}" for k in
                           ("mean_reward", "catch", "killed", "timeout", "aimed", "stuck", "hp_lost", "hits")])


if __name__ == "__main__":
    main()
