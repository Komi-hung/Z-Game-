"""Xem các thuật toán RL (PPO/TRPO/A2C/DQN) chạy cùng lúc qua từng màn và DỪNG ở màn không qua nổi.

  python watch_algos.py                       # 4 thuật toán, seed 0, mỗi màn xem 3 episode
  python watch_algos.py --seed 3 --speed 8    # seed khác, chạy nhanh hơn
  python watch_algos.py --algos ppo dqn       # chỉ so hai thuật toán
  python watch_algos.py --algos dqn dqn_tgt   # DQN mặc định vs DQN đã chỉnh (mục 5.5 báo cáo)
  python watch_algos.py --episodes 5 --level 3   # bắt đầu từ màn 3
  python watch_algos.py --live                # QUYẾT ĐỊNH DỪNG NGAY TỪ KẾT QUẢ ĐANG CHẠY (không đọc file CSV)

Mỗi ô là một thuật toán, mọi ô chạy CÙNG danh sách episode (cùng seed môi trường) nên so sánh công bằng.
Zombie (mô hình RL) đấu với người chơi giả PlayerBot trên bản đồ của màn, nhìn từ trên xuống.

Chế độ --live: không đọc CSV. Sau mỗi màn, chương trình chạy luôn baseline `path` trên CÙNG các episode đó
rồi so return trung bình của mô hình với path; thấp hơn thì ô đó dừng. Ít episode nên nhiễu hơn bảng 300 episode.

Luật dừng (mặc định): một thuật toán "qua màn" khi return trung bình >= baseline `path` trên 300 episode đánh giá
(models/algo_compare.csv, do compare_algos.py tạo). Ô nào không qua màn thì dừng tại đó và ghi
"đạt tới màn N"; ô nào qua cả 5 màn thì ghi HOÀN THÀNH. Số live chỉ để NHÌN hành vi (ít episode, nhiễu),
còn phán quyết qua/trượt lấy từ 300 episode.

Phím: Space tạm dừng · +/- đổi tốc độ · N bỏ qua episode hiện tại · Esc thoát.
"""
import argparse
import csv
import math
import os

import pygame

from config import LEVELS, MODEL_DIR, DECISION_DT, PLAYER
from env_zombie import ZombieEnv
from geometry import in_fire_line
from numpy_policy import load_policy
from zombie_brain import baseline_action

ALGO_DIR = os.path.join(MODEL_DIR, "algos")
CMP_CSV = os.path.join(MODEL_DIR, "algo_compare.csv")
SEED0 = 10_000                      # cùng seed môi trường với lúc đánh giá (compare_algos.py)
N_LEVELS = len(LEVELS)
COL = {"ppo": (209, 73, 91), "trpo": (80, 190, 120), "a2c": (70, 150, 210), "dqn": (230, 170, 50),
       "dqn_tgt": (240, 120, 40)}
TYPE_COLORS = {"heavy": (170, 130, 255), "vest": (90, 200, 90), "skinny": (240, 200, 80), "boss": (255, 70, 70)}
PANEL_W, PANEL_H, HEAD, CELL = 420, 450, 58, 18


def read_results():
    """{(algo, seed, level): (return, return_path, passed)} từ bảng đánh giá 300 episode."""
    res = {}
    for path in (CMP_CSV, os.path.join(MODEL_DIR, "algo_compare_tuned.csv")):   # + DQN đã chỉnh (dqn_tgt)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                res[r["algo"], int(r["seed"]), int(r["level"])] = (
                    float(r["mean_reward"]), float(r["path_reward"]), int(float(r["passed"])))
    return res


class Panel:
    """Một thuật toán: chạy lần lượt từng màn, dừng khi bảng 300 episode báo không qua."""

    def __init__(self, algo, seed, results, start_level, episodes, live=False):
        self.algo, self.seed, self.results, self.episodes, self.live = algo, seed, results, episodes, live
        self.level = start_level
        self.reached = start_level - 1          # số màn đã qua liên tiếp
        self.done, self.status = False, ""
        self.policies = {}
        self.history = []                       # (màn, return live, return 300ep, path 300ep, qua?)
        self.start_level_run()

    def policy(self, level):
        if level not in self.policies:
            path = os.path.join(ALGO_DIR, self.algo, f"seed{self.seed}", f"zombie_lv{level}.zip")
            self.policies[level] = load_policy(path)
        return self.policies[level]

    def start_level_run(self):
        self.env = ZombieEnv(level=self.level, replay_prob=0.0)
        self.model = self.policy(self.level)
        self.ep, self.live_returns, self.pause = 0, [], 0
        self.new_episode()

    def new_episode(self):
        self.obs, _ = self.env.reset(seed=SEED0 + self.ep)
        self.ep_r, self.outcome = 0.0, None

    def step(self):
        if self.done:
            return
        if self.outcome is not None:                      # giữ khung cuối của episode ~1 giây
            self.pause -= 1
            if self.pause <= 0:
                self.next_episode()
            return
        action, _ = self.model.predict(self.obs, deterministic=True)
        self.obs, r, term, trunc, info = self.env.step(int(action))
        self.ep_r += r
        if term or trunc:
            self.outcome, self.pause = info["outcome"], 10

    def next_episode(self):
        self.live_returns.append(self.ep_r)
        self.ep += 1
        if self.ep < self.episodes:
            self.new_episode()
            return
        self.finish_level()

    def finish_level(self):
        live = sum(self.live_returns) / len(self.live_returns)
        if self.live:                                     # quyết định từ chính các episode vừa chạy
            mean_r, path_r = live, self.path_live_return()
            passed = int(mean_r >= path_r)
        else:
            mean_r, path_r, passed = self.results[self.algo, self.seed, self.level]
        self.history.append((self.level, live, mean_r, path_r, passed))
        if not passed:
            self.done, self.status = True, f"DỪNG ở màn {self.level}  (đạt tới màn {self.reached})"
        else:
            self.reached = self.level
            if self.level >= N_LEVELS:
                self.done, self.status = True, "HOÀN THÀNH 5/5 màn"
            else:
                self.level += 1
                self.start_level_run()

    def path_live_return(self):
        """Return trung bình của baseline `path` trên đúng các seed mà mô hình vừa chơi (chạy ẩn, không vẽ)."""
        env, total = ZombieEnv(level=self.level, replay_prob=0.0), 0.0
        for ep in range(self.episodes):
            env.reset(seed=SEED0 + ep)
            done = False
            while not done:
                a = baseline_action("path", env.zx, env.zy, env.bot.x, env.bot.y, env.path,
                                    env.attack.ready, env.st["attack_range"])
                _, r, term, trunc, _ = env.step(a)
                total += r
                done = term or trunc
        return total / self.episodes

    def skip_episode(self):
        if not self.done and self.outcome is None:
            self.outcome, self.pause = "bỏ qua", 1


def draw_panel(panel, fonts, W=PANEL_W, H=PANEL_H):
    surf = pygame.Surface((W, H))
    surf.fill((18, 18, 24))
    f, fs = fonts
    col = COL.get(panel.algo, (200, 200, 200))
    pygame.draw.rect(surf, col, (0, 0, W, 4))
    surf.blit(f.render(panel.algo.upper(), True, col), (10, 8))
    reached = f"đạt tới màn {panel.reached}/{N_LEVELS}"
    surf.blit(f.render(reached, True, (235, 235, 235)), (W - f.size(reached)[0] - 10, 8))
    env = panel.env
    cur = LEVELS[panel.level - 1]["name"]
    surf.blit(fs.render(f"{cur}   episode {min(panel.ep + 1, panel.episodes)}/{panel.episodes}"
                        f"   return {panel.ep_r:+.2f}", True, (190, 200, 215)), (10, 34))
    gw, gh = len(env.grid[0]), len(env.grid)
    ox, oy = (W - gw * CELL) // 2, HEAD + 4
    for y, row in enumerate(env.grid):
        for x, wall in enumerate(row):
            if wall:
                pygame.draw.rect(surf, (95, 95, 108), (ox + x * CELL, oy + y * CELL, CELL, CELL))
            else:
                pygame.draw.rect(surf, (30, 30, 38), (ox + x * CELL, oy + y * CELL, CELL, CELL), 1)
    b = env.bot
    bc = (ox + int(b.x * CELL), oy + int(b.y * CELL))
    aimed = in_fire_line(env.grid, b.x, b.y, b.angle, env.zx, env.zy, env.zr, b.weapon)
    end = (bc[0] + int(math.cos(b.angle) * CELL * 3), bc[1] + int(math.sin(b.angle) * CELL * 3))
    pygame.draw.line(surf, (255, 90, 90) if aimed else (120, 170, 230), bc, end, 2)
    pygame.draw.circle(surf, (80, 180, 255), bc, int(0.4 * CELL))
    zc = (ox + int(env.zx * CELL), oy + int(env.zy * CELL))
    pygame.draw.circle(surf, TYPE_COLORS[env.ztype], zc, max(7, int(env.zr * CELL * 1.6)))
    if aimed:
        pygame.draw.circle(surf, (255, 255, 255), zc, max(7, int(env.zr * CELL * 1.6)) + 3, 1)
    # thanh máu zombie (và người chơi giả khi là trận boss)
    pygame.draw.rect(surf, (60, 0, 0), (10, H - 28, 150, 8))
    pygame.draw.rect(surf, (220, 60, 60), (10, H - 28, int(150 * max(env.hp, 0) / env.max_hp), 8))
    surf.blit(fs.render(f"{env.ztype}", True, (210, 210, 210)), (168, H - 33))
    if env.duel:
        pygame.draw.rect(surf, (0, 30, 60), (250, H - 28, 150, 8))
        pygame.draw.rect(surf, (80, 180, 255), (250, H - 28, int(150 * max(b.hp, 0) / PLAYER["hp"]), 8))
    if panel.outcome:
        label = {"caught": "ZOMBIE CHẠM ĐƯỢC", "won": "BOSS THẮNG", "killed": "ZOMBIE BỊ HẠ",
                 "timeout": "HẾT GIỜ"}.get(panel.outcome, panel.outcome)
        t = f.render(label, True, (255, 230, 120))
        surf.blit(t, ((W - t.get_width()) // 2, H // 2))
    if panel.done:
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 150))
        surf.blit(veil, (0, HEAD))
        ok = panel.reached >= N_LEVELS
        t = f.render(panel.status, True, (120, 255, 150) if ok else (255, 130, 110))
        surf.blit(t, ((W - t.get_width()) // 2, H // 2 - 40))
        y = H // 2 - 10
        for lv, live, mean_r, path_r, passed in panel.history:
            line = f"Màn {lv}: {mean_r:+.2f} vs path {path_r:+.2f}  {'qua' if passed else 'TRƯỢT'}"
            surf.blit(fs.render(line, True, (230, 230, 230) if passed else (255, 150, 130)), (W // 2 - 110, y))
            y += 18
    return surf


def summary(panels):
    print("\nKết quả (seed %d):" % panels[0].seed)
    for p in sorted(panels, key=lambda p: -p.reached):
        print(f"  {p.algo.upper():5} đạt tới màn {p.reached}/{N_LEVELS}  - {p.status or 'chưa xong'}")


def main():
    ap = argparse.ArgumentParser(description="Xem PPO/TRPO/A2C/DQN chạy qua các màn đến khi dừng")
    ap.add_argument("--algos", nargs="+", default=["ppo", "trpo", "a2c", "dqn"], choices=list(COL))
    ap.add_argument("--seed", type=int, default=0, help="seed huấn luyện của mô hình (0-4)")
    ap.add_argument("--level", type=int, default=1, help="màn bắt đầu")
    ap.add_argument("--episodes", type=int, default=None, help="số episode xem mỗi màn (mặc định 3; 10 nếu --live)")
    ap.add_argument("--live", action="store_true", help="quyết định dừng từ kết quả đang chạy, không đọc CSV")
    ap.add_argument("--speed", type=float, default=4.0, help="1 = thời gian thật (10 bước/giây)")
    ap.add_argument("--fast", action="store_true", help="không đợi khung hình (dùng khi kiểm tra)")
    ap.add_argument("--max-seconds", type=float, default=None, help="tự thoát sau N giây thật")
    args = ap.parse_args()

    if args.episodes is None:
        args.episodes = 10 if args.live else 3
    results = {} if args.live else read_results()
    panels = [Panel(a, args.seed, results, args.level, args.episodes, args.live) for a in args.algos]
    pygame.init()
    cols = 2 if len(panels) > 1 else 1
    rows = math.ceil(len(panels) / cols)
    screen = pygame.display.set_mode((cols * PANEL_W, rows * PANEL_H))
    pygame.display.set_caption("Zombie RL - so sánh thuật toán qua từng màn")
    fonts = (pygame.font.SysFont("dejavusans,arial", 20, bold=True), pygame.font.SysFont("dejavusans,arial", 14))
    clock = pygame.time.Clock()
    speed, paused, acc, t0 = args.speed, False, 0.0, pygame.time.get_ticks()
    announced = False
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                summary(panels)
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed = min(speed * 1.5, 60)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed / 1.5, 0.5)
                elif e.key == pygame.K_n:
                    for p in panels:
                        p.skip_episode()
        dt = 1 / 60 if args.fast else clock.tick(60) / 1000
        if not paused:
            acc += dt * speed / DECISION_DT          # số bước mô phỏng cần chạy trong khung này
            if args.fast:
                acc = max(acc, 20)
            while acc >= 1:
                acc -= 1
                for p in panels:
                    p.step()
        for i, p in enumerate(panels):
            screen.blit(draw_panel(p, fonts), ((i % cols) * PANEL_W, (i // cols) * PANEL_H))
        pygame.display.flip()
        if all(p.done for p in panels) and not announced:
            announced = True
            summary(panels)
            if args.fast or args.max_seconds is not None:
                return
        if args.max_seconds is not None and (pygame.time.get_ticks() - t0) / 1000 > args.max_seconds:
            summary(panels)
            return


if __name__ == "__main__":
    main()
