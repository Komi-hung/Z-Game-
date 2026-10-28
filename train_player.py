"""Train RL agent đóng vai NGƯỜI CHƠI (PPO) để làm "đối thủ thử" cho zombie.

Agent được tập ĐỀU với cả 3 loại zombie (PPO, viết tay "path", "chase" mỗi loại 1/3) trên cả 5 màn,
để nó không chỉ học cách khai thác riêng zombie PPO -> phép so sánh ở evaluate_match.py công bằng.
Zombie KHÔNG học thêm trong lúc này (model zombie được giữ nguyên).

Chạy:        python train_player.py                  (mặc định 3 triệu bước)
Train tiếp:  python train_player.py --resume --steps 1000000
Kết quả:     models/player_agent.zip (+ checkpoint trong models/player_ckpt/)
"""
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # tránh các tiến trình tranh nhau nhân CPU

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from config import MODEL_DIR
from player_env import PlayerEnv

AGENT_PATH = os.path.join(MODEL_DIR, "player_agent.zip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=min(8, max(4, os.cpu_count() or 1)),
                    help="số env chạy song song (mỗi env một tiến trình)")
    ap.add_argument("--resume", action="store_true", help="train tiếp từ models/player_agent.zip")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    vec_cls = SubprocVecEnv if args.envs > 1 else DummyVecEnv
    env = make_vec_env(PlayerEnv, n_envs=args.envs, seed=args.seed, vec_env_cls=vec_cls)
    if args.resume and os.path.exists(AGENT_PATH):
        print(f"Train tiếp từ {AGENT_PATH}")
        model = PPO.load(AGENT_PATH, env=env, device="cpu")
    else:
        model = PPO("MlpPolicy", env, n_steps=max(256, 4096 // args.envs), batch_size=1024, n_epochs=10,
                    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
                    policy_kwargs=dict(net_arch=[256, 256]), seed=args.seed, verbose=1, device="cpu")
    ckpt = CheckpointCallback(save_freq=max(1, 250_000 // args.envs),
                              save_path=os.path.join(MODEL_DIR, "player_ckpt"), name_prefix="player")
    model.learn(total_timesteps=args.steps, callback=ckpt, reset_num_timesteps=not args.resume)
    model.save(AGENT_PATH)
    env.close()
    print(f"Đã lưu {AGENT_PATH}. Chạy tiếp: python evaluate_match.py")


if __name__ == "__main__":
    main()
