"""Zombie RL mạnh tới đâu? Cho người chơi đấu N trận với từng loại zombie và đếm ai thắng.

Logic đánh giá:
  1. Người chơi: RL agent (models/player_agent.zip) và người chơi viết tay (đối chứng).
     Cả hai dùng cùng nút bấm, cùng giới hạn tốc độ quay, cùng nhịp 0.1 giây.
  2. Zombie: "ppo" (zombie RL) so với 2 bản viết tay "path" và "chase".
     -> Zombie RL tốt nếu nó THẮNG NHIỀU HƠN bản viết tay khi gặp cùng một người chơi.
  3. Trận thứ i của mọi cặp dùng CÙNG seed (cùng vị trí xuất hiện, cùng độ lệch đạn ngẫu nhiên),
     nên khác biệt đến từ cách zombie chơi chứ không phải do may rủi (so sánh theo cặp).
  4. Mỗi tỉ lệ thắng có khoảng tin cậy 95% (Wilson). 100 trận chỉ cho độ chính xác khoảng ±10%,
     nên chênh lệch nhỏ hơn mức đó có thể chỉ là ngẫu nhiên -> xem thêm p-value.

Hai chế độ:
  level    : mỗi màn đấu N trận riêng (100 máu đầu mỗi trận)
  campaign : chơi liền màn 1 -> 5 với MỘT mạng như game thật; chết ở màn nào thì zombie thắng

Chạy:  python evaluate_match.py                          (100 trận, cả 2 chế độ)
       python evaluate_match.py --matches 50 --players agent --zombies ppo,path
Kết quả lưu ở models/match_eval.csv
"""
import argparse
import csv
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # tránh các tiến trình tranh nhau nhân CPU
import time
from multiprocessing import Pool

from config import LEVELS, MODEL_DIR
from player_env import PlayerEnv, ScriptedPlayer

AGENT_PATH = os.path.join(MODEL_DIR, "player_agent.zip")
_W = {}     # dữ liệu riêng của mỗi tiến trình con


def _init_worker(agent_path):
    _W["env"] = PlayerEnv()
    _W["scripted"] = ScriptedPlayer()
    _W["agent"] = None
    if agent_path and os.path.exists(agent_path):
        from stable_baselines3 import PPO
        _W["agent"] = PPO.load(agent_path, device="cpu")


def play_level(player, level, controller, seed):
    """Một trận = một màn. Trả về (kết quả, thời gian, số zombie hạ, máu còn lại)."""
    env = _W["env"]
    env.set_match(level, controller, seed)
    obs, _ = env.reset()
    while True:
        if player == "agent":
            action, _ = _W["agent"].predict(obs, deterministic=True)
        else:
            action = _W["scripted"].act(env.world)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            return info["outcome"], info["time"], info["kills"], info["hp"]


def run_task(task):
    mode, player, controller, level, seed = task
    if mode == "level":
        outcome, t, kills, hp = play_level(player, level, controller, seed)
        return dict(mode=mode, player=player, zombies=controller, level=level, seed=seed,
                    outcome=outcome, time=round(t, 1), kills=kills, hp=hp)
    # campaign: màn 1 -> 5, một mạng
    total_t = total_k = 0
    for lv in range(1, len(LEVELS) + 1):
        outcome, t, kills, hp = play_level(player, lv, controller, seed * 10 + lv)
        total_t += t
        total_k += kills
        if outcome != "player_win":
            return dict(mode=mode, player=player, zombies=controller, level=lv, seed=seed,
                        outcome=outcome, time=round(total_t, 1), kills=total_k, hp=hp)
    return dict(mode=mode, player=player, zombies=controller, level=len(LEVELS), seed=seed,
                outcome="player_win", time=round(total_t, 1), kills=total_k, hp=hp)


def wilson(k, n, z=1.96):
    """Khoảng tin cậy 95% cho tỉ lệ k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_p(rows_a, rows_b):
    """So sánh theo cặp (cùng seed): zombie A thắng mà B thua (b) và ngược lại (c).
    Kiểm định McNemar (xấp xỉ chuẩn, có hiệu chỉnh liên tục). p < 0.05: khác biệt có ý nghĩa."""
    a = {(r["level"] if r["mode"] == "level" else 0, r["seed"]): r["outcome"] == "zombie_win" for r in rows_a}
    b = {(r["level"] if r["mode"] == "level" else 0, r["seed"]): r["outcome"] == "zombie_win" for r in rows_b}
    keys = a.keys() & b.keys()
    n01 = sum(1 for k in keys if a[k] and not b[k])
    n10 = sum(1 for k in keys if b[k] and not a[k])
    if n01 + n10 == 0:
        return 1.0, n01, n10
    z = (abs(n01 - n10) - 1) / math.sqrt(n01 + n10)
    return math.erfc(max(z, 0) / math.sqrt(2)), n01, n10


def summarize(rows, title):
    print(f"\n=== {title} ===")
    print(f"{'người chơi':10} {'zombie':7} {'màn':>4} {'zombie thắng':>13} {'95% CI':>14} "
          f"{'người thắng':>12} {'hòa':>5} {'TB giây':>8}")
    groups = {}
    for r in rows:
        key = (r["player"], r["zombies"], r["level"] if r["mode"] == "level" else "1-5")
        groups.setdefault(key, []).append(r)
    for (player, zombies, level), g in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][2]), kv[0][1])):
        n = len(g)
        zw = sum(r["outcome"] == "zombie_win" for r in g)
        pw = sum(r["outcome"] == "player_win" for r in g)
        lo, hi = wilson(zw, n)
        print(f"{player:10} {zombies:7} {str(level):>4} {zw:>6}/{n:<6} {lo:6.0%} - {hi:4.0%} {pw:>12} "
              f"{n - zw - pw:>5} {sum(r['time'] for r in g) / n:8.1f}")


def compare(rows, title):
    """Zombie PPO có thắng nhiều hơn zombie viết tay một cách có ý nghĩa không?"""
    print(f"\n--- {title}: PPO so với zombie viết tay (theo cặp, cùng seed) ---")
    players = sorted({r["player"] for r in rows})
    for player in players:
        ppo = [r for r in rows if r["player"] == player and r["zombies"] == "ppo"]
        if not ppo:
            continue
        for base in ("path", "chase"):
            other = [r for r in rows if r["player"] == player and r["zombies"] == base]
            if not other:
                continue
            p, n01, n10 = mcnemar_p(ppo, other)
            ka = sum(r["outcome"] == "zombie_win" for r in ppo)
            kb = sum(r["outcome"] == "zombie_win" for r in other)
            verdict = "khác biệt CÓ ý nghĩa" if p < 0.05 else "chưa đủ bằng chứng"
            print(f"{player:9} ppo {ka}/{len(ppo)} vs {base} {kb}/{len(other)} | "
                  f"chỉ ppo thắng: {n01}, chỉ {base} thắng: {n10} | p = {p:.3f} -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=100, help="số trận cho mỗi cặp (người chơi, zombie, màn)")
    ap.add_argument("--players", default="agent,scripted", help="agent và/hoặc scripted")
    ap.add_argument("--zombies", default="ppo,path,chase")
    ap.add_argument("--levels", default="1,2,3,4,5")
    ap.add_argument("--mode", choices=["level", "campaign", "both"], default="both")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--agent", default=AGENT_PATH)
    args = ap.parse_args()

    players = [p for p in args.players.split(",") if p]
    if "agent" in players and not os.path.exists(args.agent):
        print(f"[!] Chưa có {args.agent} - bỏ qua agent (chạy train_player.py trước)")
        players.remove("agent")
    zombies = args.zombies.split(",")
    levels = [int(x) for x in args.levels.split(",")]
    tasks = []
    for player in players:
        for z in zombies:
            if args.mode in ("level", "both"):
                tasks += [("level", player, z, lv, args.seed + i) for lv in levels for i in range(args.matches)]
            if args.mode in ("campaign", "both"):
                tasks += [("campaign", player, z, 0, args.seed + i) for i in range(args.matches)]
    print(f"{len(tasks)} trận, {args.workers} tiến trình...")
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker, initargs=(args.agent,)) as pool:
        rows = []
        for k, r in enumerate(pool.imap_unordered(run_task, tasks, chunksize=4), 1):
            rows.append(r)
            if k % 200 == 0:
                print(f"  {k}/{len(tasks)} trận ({time.time() - t0:.0f}s)")
    print(f"Xong sau {time.time() - t0:.0f} giây")

    level_rows = [r for r in rows if r["mode"] == "level"]
    camp_rows = [r for r in rows if r["mode"] == "campaign"]
    if level_rows:
        summarize(level_rows, f"Từng màn: {args.matches} trận mỗi màn")
        compare(level_rows, "Gộp các màn")
    if camp_rows:
        summarize(camp_rows, f"Cả game (màn 1 -> 5, một mạng): {args.matches} lượt")
        compare(camp_rows, "Cả game")
        for player in players:
            for z in zombies:
                g = [r for r in camp_rows if r["player"] == player and r["zombies"] == z and r["outcome"] != "player_win"]
                dist = {lv: sum(r["level"] == lv for r in g) for lv in range(1, len(LEVELS) + 1)}
                print(f"  {player:9} vs {z:6}: người chơi dừng ở màn {dist}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    out = os.path.join(MODEL_DIR, "match_eval.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["mode"], r["player"], r["zombies"], r["level"], r["seed"])))
    print(f"\nĐã lưu từng trận vào {out}")


if __name__ == "__main__":
    main()
