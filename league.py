"""Self-play theo vòng (league): zombie và người chơi RL thay phiên nhau học để vượt mặt nhau.

Mỗi vòng:
  1. ZOMBIE HỌC: mỗi màn, zombie (và boss) luyện ngay trong game thật (zombie_world_env.py)
     với TẤT CẢ các phiên bản người chơi RL đã có (+ người chơi viết tay), bắt đầu từ model tốt nhất.
  2. GIỮ BẢN TỐT NHẤT: bản zombie mới đấu với bản tốt nhất hiện tại trên CÙNG các trận (cùng seed,
     cùng đối thủ). Màn nào bản mới thắng nhiều hơn -> được lên làm bản chính thức trong game
     (models/zombie_lvK.zip). Bản cũ vẫn lưu trong models/league/.
  3. NGƯỜI CHƠI HỌC: agent người chơi luyện tiếp để thắng bản zombie tốt nhất, rồi được thêm vào
     nhóm đối thủ của vòng sau. Zombie nào vẫn thắng được người chơi đã luyện riêng để khắc chế nó
     mới là zombie "khôn thật" (đo bằng tỉ lệ thắng ở bước 4).
  4. ĐO: zombie tốt nhất đấu với người chơi mới -> ghi vào models/league.csv.

Chạy 1 vòng (mặc định):   python league.py
Chạy nhiều vòng:          python league.py --rounds 3
Thử nhanh:                python league.py --scale 0.05 --eval-matches 10
So & giữ bản tốt hơn:     python league.py --promote-from <thư mục có zombie_lvK.zip>
"""
import argparse
import csv
import os
import shutil
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

from multiprocessing import Pool

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from config import MODEL_DIR, LEVELS
from player_env import PlayerEnv, ScriptedPlayer
from train import load_model
from zombie_world_env import ZombieWorldEnv

LEAGUE_DIR = os.path.join(MODEL_DIR, "league")
PLAYER_DIR = os.path.join(LEAGUE_DIR, "players")
LOG_PATH = os.path.join(MODEL_DIR, "league.csv")
STEPS = {1: 150_000, 2: 300_000, 3: 200_000, 4: 300_000, 5: 600_000}   # số bước zombie học mỗi vòng
PLAYER_STEPS = 1_500_000


# ---------------------------------------------------------------- đánh giá
_W = {}


def _init_eval(zombie_dir, opponent_paths, controller="ppo"):
    _W["controller"] = controller
    _W["env"] = PlayerEnv(zombie_dir=zombie_dir)
    _W["opp"] = [PPO.load(p, device="cpu") for p in opponent_paths]
    _W["scripted"] = ScriptedPlayer()


def _play(task):
    level, opp, seed = task
    env = _W["env"]
    env.set_match(level, _W["controller"], seed)
    obs, _ = env.reset()
    while True:
        if opp < 0:
            action = _W["scripted"].act(env.world)
        else:
            action, _ = _W["opp"][opp].predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            return level, opp, seed, info["outcome"] == "zombie_win"


def zombie_win_rates(zombie_dir, levels, opponent_paths, n, seed, workers, controller="ppo", scripted=True):
    """Tỉ lệ zombie thắng (người chơi chết) theo từng màn và từng đối thủ. opp = -1 là người chơi viết tay."""
    opps = list(range(len(opponent_paths))) + ([-1] if scripted else [])
    tasks = [(lv, o, seed + i) for lv in levels for o in opps for i in range(n)]
    with Pool(workers, initializer=_init_eval, initargs=(zombie_dir, opponent_paths, controller)) as pool:
        results = pool.map(_play, tasks, chunksize=2)
    rates = {}
    for lv, o, _, win in results:
        rates.setdefault((lv, o), []).append(win)
    return {k: sum(v) / len(v) for k, v in rates.items()}


def level_score(rates, level):
    """Điểm của một màn = trung bình tỉ lệ thắng trước mọi đối thủ."""
    vals = [v for (lv, _), v in rates.items() if lv == level]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------- huấn luyện
def vec(cls_kwargs, envs, env_cls):
    vec_cls = SubprocVecEnv if envs > 1 else DummyVecEnv
    return make_vec_env(env_cls, n_envs=envs, env_kwargs=cls_kwargs, vec_env_cls=vec_cls)


def train_zombie(level, init_path, opponents, steps, out_path, envs, seed):
    env = vec(dict(level=level, opponents=opponents, teammate_dir=MODEL_DIR), envs, ZombieWorldEnv)
    model = load_model(init_path, env, seed)
    model.learn(total_timesteps=steps, reset_num_timesteps=False)
    model.save(out_path)
    env.close()


def train_player(init_path, steps, out_path, envs, seed):
    env = vec(dict(zombie_dir=MODEL_DIR), envs, PlayerEnv)
    model = PPO.load(init_path, env=env, device="cpu")
    model.learn(total_timesteps=steps, reset_num_timesteps=False)
    model.save(out_path)
    env.close()


def log(rows):
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "round", "phase", "level", "opponent", "zombie_win_rate", "matches"])
        w.writerows(rows)


def compare_and_promote(cand_dir, levels, opponents, matches, seed, workers, round_id, margin=0.03):
    """So bộ zombie trong cand_dir với bản tốt nhất (models/) trên cùng các trận.
    Màn nào bản mới thắng nhiều hơn ÍT NHẤT `margin` thì mới lên (chênh nhỏ hơn có thể chỉ là ngẫu nhiên)."""
    best = zombie_win_rates(MODEL_DIR, levels, opponents, matches, seed, workers)
    cand = zombie_win_rates(cand_dir, levels, opponents, matches, seed, workers)
    rows = []
    for lv in sorted(levels):
        sb, sc = level_score(best, lv), level_score(cand, lv)
        keep = sc >= sb + margin
        print(f"  màn {lv}: tốt nhất {sb:.0%} -> bản mới {sc:.0%}  {'=> LÊN BẢN MỚI' if keep else '=> giữ bản cũ'}")
        if keep:
            shutil.copy(os.path.join(cand_dir, f"zombie_lv{lv}.zip"), os.path.join(MODEL_DIR, f"zombie_lv{lv}.zip"))
        stamp = time.strftime("%Y-%m-%d %H:%M")
        rows.append([stamp, round_id, "zombie_best_before", lv, "all", f"{sb:.3f}", matches])
        rows.append([stamp, round_id, "zombie_candidate", lv, "all", f"{sc:.3f}", matches])
        rows.append([stamp, round_id, "promoted", lv, "", int(keep), ""])
    log(rows)


def player_pool():
    files = sorted(f for f in os.listdir(PLAYER_DIR) if f.endswith(".zip"))
    return [os.path.join(PLAYER_DIR, f) for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--levels", default="5,2,4,3,1", help="thứ tự màn cho zombie học (boss trước)")
    ap.add_argument("--scale", type=float, default=1.0, help="nhân số bước train")
    ap.add_argument("--player-steps", type=int, default=PLAYER_STEPS)
    ap.add_argument("--eval-matches", type=int, default=40, help="số trận cho mỗi (màn, đối thủ) khi so bản mới/cũ")
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.03,
                    help="bản mới phải thắng nhiều hơn ít nhất chừng này mới được lên (chống may rủi)")
    ap.add_argument("--promote-from", default=None, help="chỉ so bộ zombie trong thư mục này với bản tốt nhất rồi thoát")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]
    os.makedirs(PLAYER_DIR, exist_ok=True)

    # Vòng 0: người chơi hiện có là đối thủ đầu tiên
    if not player_pool():
        first = os.path.join(MODEL_DIR, "player_agent.zip")
        if not os.path.exists(first):
            raise SystemExit("Chưa có models/player_agent.zip - chạy python train_player.py trước")
        shutil.copy(first, os.path.join(PLAYER_DIR, "player_r00.zip"))
    start = len(player_pool())

    if args.promote_from:
        print(f"So {args.promote_from} với bản tốt nhất trong {MODEL_DIR}/ ...")
        compare_and_promote(args.promote_from, levels, player_pool(), args.eval_matches, 40_000,
                            args.workers, "manual", args.margin)
        return

    for r in range(start, start + args.rounds):
        rdir = os.path.join(LEAGUE_DIR, f"round_{r:02d}")
        os.makedirs(rdir, exist_ok=True)
        opponents = player_pool()
        # đối thủ mới nhất xuất hiện nhiều gấp đôi khi zombie học
        train_opps = opponents + [opponents[-1]]
        print(f"\n######## VÒNG {r}: {len(opponents)} phiên bản người chơi trong nhóm đối thủ ########")
        t0 = time.time()

        # 1. Zombie học
        for lv in levels:
            steps = int(STEPS[lv] * args.scale)
            print(f"\n== Zombie màn {lv}: học {steps:,} bước trong game thật ==")
            train_zombie(lv, os.path.join(MODEL_DIR, f"zombie_lv{lv}.zip"), train_opps, steps,
                         os.path.join(rdir, f"zombie_lv{lv}.zip"), args.envs, args.seed + r)

        # 2. So bản mới với bản tốt nhất trên cùng các trận, giữ bản tốt hơn
        print("\n== So bản mới với bản tốt nhất (cùng seed, cùng đối thủ) ==")
        cand_dir = os.path.join(rdir, "candidate")
        os.makedirs(cand_dir, exist_ok=True)
        for lv in range(1, len(LEVELS) + 1):
            src = os.path.join(rdir, f"zombie_lv{lv}.zip")
            if not os.path.exists(src):
                src = os.path.join(MODEL_DIR, f"zombie_lv{lv}.zip")
            shutil.copy(src, os.path.join(cand_dir, f"zombie_lv{lv}.zip"))
        eval_seed = 50_000 + 1000 * r
        compare_and_promote(cand_dir, levels, opponents, args.eval_matches, eval_seed, args.workers, r,
                            args.margin)

        # 3. Người chơi học để khắc chế bản zombie tốt nhất
        steps = int(args.player_steps * args.scale)
        print(f"\n== Người chơi học {steps:,} bước để thắng zombie tốt nhất ==")
        new_player = os.path.join(PLAYER_DIR, f"player_r{r:02d}.zip")
        train_player(opponents[-1], steps, new_player, args.envs, args.seed + r)
        shutil.copy(new_player, os.path.join(MODEL_DIR, "player_agent.zip"))

        # 4. Zombie tốt nhất có còn thắng được người chơi vừa luyện để khắc chế nó?
        # Mốc so sánh: zombie viết tay "path" (đi theo đường ngắn nhất) trên CÙNG các trận.
        # Zombie RL chỉ thực sự "khôn" khi thắng nhiều hơn mốc này.
        print("\n== Zombie tốt nhất (RL) vs người chơi mới, so với mốc zombie viết tay ==")
        n_final = args.eval_matches * 2
        rl = zombie_win_rates(MODEL_DIR, sorted(levels), [new_player], n_final, eval_seed + 500,
                              args.workers, "ppo", scripted=False)
        ref = zombie_win_rates(MODEL_DIR, sorted(levels), [new_player], n_final, eval_seed + 500,
                               args.workers, "path", scripted=False)
        rows = []
        for lv in sorted(levels):
            v, b = rl[(lv, 0)], ref[(lv, 0)]
            print(f"  màn {lv}: zombie RL thắng {v:.0%} | zombie viết tay thắng {b:.0%} "
                  f"{'-> RL TỐT HƠN' if v > b else '-> RL chưa vượt mốc'}")
            stamp = time.strftime("%Y-%m-%d %H:%M")
            rows.append([stamp, r, "vs_new_player", lv, f"player_r{r:02d}", f"{v:.3f}", n_final])
            rows.append([stamp, r, "path_baseline_vs_new_player", lv, f"player_r{r:02d}", f"{b:.3f}", n_final])
        log(rows)
        print(f"\nVòng {r} xong sau {(time.time() - t0) / 60:.0f} phút. Kết quả: {LOG_PATH}")


if __name__ == "__main__":
    main()
