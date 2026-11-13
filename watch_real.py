"""Xem các thuật toán zombie đi từ màn 1 đến màn 5 trong GAME THẬT (góc nhìn người chơi), mỗi ô một thuật toán.

  python watch_real.py                         # PPO/TRPO/A2C/DQN, từ màn 1, mỗi màn 5 trận
  python watch_real.py --matches 3 --speed 2   # ít trận hơn, chạy nhanh hơn
  python watch_real.py --level 3               # bắt đầu từ màn 3
  python watch_real.py --algos ppo dqn         # chỉ so hai thuật toán
  python watch_real.py --no-stop               # không dừng, cứ chơi hết 5 màn (chỉ để xem)
  python watch_real.py --max-seconds 60 --shot demo.png   # tự thoát sau 60 giây và lưu ảnh (kiểm tra)

Mỗi ô là một World riêng của game thật. Cả các ô dùng CÙNG seed (cùng điểm xuất hiện của zombie) và CÙNG agent
người chơi RL (models/player_agent.zip), chỉ khác model zombie. Mỗi màn chơi `--matches` trận; trận kết thúc khi
người chơi dọn hết zombie (hoặc hạ boss) hoặc bị hạ.

Luật dừng (tính ngay từ các trận vừa xem, không đọc file CSV): sau mỗi màn chương trình chạy luôn zombie viết tay
`path` (đi theo đường ngắn nhất) trên CÙNG seed và cùng agent người chơi, rồi so SÁT THƯƠNG TRUNG BÌNH gây cho người
chơi. Zombie RL gây ít hơn path thì ô đó dừng và ghi "đạt tới màn N"; qua cả 5 màn thì ghi HOÀN THÀNH.
Số trận ít nên nhiễu; kết luận chính thức lấy từ bảng 300 episode trong báo cáo.

Mặc định chạy TOÀN MÀN HÌNH (F hoặc F11 bật/tắt; --windowed để chạy cửa sổ).
Phím: Space tạm dừng · +/- đổi tốc độ · Tab bật/tắt minimap · Esc thoát.
"""
import argparse
import os

import pygame

from config import FPS, RENDER_W, RENDER_H, WINDOW_SCALE, LEVELS, PLAYER
from game import Game
from game_core import World, ModelBank
from player_env import PlayerAgentDriver

COL = {"ppo": (209, 73, 91), "trpo": (80, 190, 120), "a2c": (70, 150, 210), "dqn": (230, 170, 50),
       "dqn_tgt": (240, 120, 40)}
PW, PH, BAR = 640, 400, 34                      # kích thước một ô (hình) + thanh tiêu đề
N_LEVELS = len(LEVELS)
MAX_SIM_S = 300                                 # trần thời gian mô phỏng cho một trận chạy ẩn (giây game)
_path_cache = {}


def play_hidden(level, controller, seed, agent_path, zdir=None):
    """Chạy một trận không vẽ: trả về (người chơi chết?, sát thương zombie gây ra cho người chơi)."""
    w = World(level, controller, seed=seed, model_bank=ModelBank(zdir) if zdir else None)
    agent = PlayerAgentDriver(agent_path)
    for _ in range(int(MAX_SIM_S * FPS)):
        if w.state != "play":
            break
        w.step(1 / FPS, agent.controls(w, 1 / FPS))
    return w.state == "dead", PLAYER["hp"] - max(w.player.hp, 0)


def path_baseline(level, seeds, agent_path):
    key = (level, tuple(seeds))
    if key not in _path_cache:
        rs = [play_hidden(level, "path", s, agent_path) for s in seeds]
        _path_cache[key] = (sum(d for _, d in rs) / len(rs), sum(int(x) for x, _ in rs))
    return _path_cache[key]


class Panel:
    def __init__(self, algo, args):
        self.algo, self.args = algo, args
        self.zdir = os.path.join("models", "algos", algo, f"seed{args.zombie_seed}")
        self.game = Game(args.level, demo=False, controller="ppo", agent_path=args.agent, zombie_dir=self.zdir)
        self.game.screen = pygame.Surface((RENDER_W * WINDOW_SCALE, RENDER_H * WINDOW_SCALE))
        self.bank = ModelBank(self.zdir)
        self.level, self.idx = args.level, 0
        self.reached = args.level - 1
        self.matches = []                         # (người chơi chết?, sát thương) của màn hiện tại
        self.history = []                         # (màn, sát thương RL, sát thương path, thắng RL, thắng path, qua?)
        self.done, self.status = False, ""
        self.wait, self.ended = 0.0, False
        self.start_match()

    def seeds(self):
        return [self.args.world_seed + i for i in range(self.args.matches)]

    def start_match(self):
        g = self.game
        g.world = World(self.level, "ppo", seed=self.args.world_seed + self.idx, model_bank=self.bank)
        g.agent.reset()
        self.wait, self.ended = 0.0, False

    def step(self, dt):
        if self.done:
            return
        g = self.game
        w = g.world
        w.step(dt, g.read_input(dt))
        if w.state != "play":
            if not self.ended:
                self.ended = True
                self.matches.append((w.state == "dead", PLAYER["hp"] - max(w.player.hp, 0)))
            self.wait += dt
            if self.wait > 2.0:
                self.next_match()

    def next_match(self):
        self.idx += 1
        if self.idx < self.args.matches:
            self.start_match()
            return
        mine = sum(d for _, d in self.matches) / len(self.matches)
        wins = sum(int(x) for x, _ in self.matches)
        path_d, path_w = path_baseline(self.level, self.seeds(), self.args.agent)
        passed = mine >= path_d
        self.history.append((self.level, mine, path_d, wins, path_w, passed))
        if passed:
            self.reached = self.level
        if (not passed and not self.args.no_stop) or self.level >= N_LEVELS:
            ok = self.reached >= N_LEVELS
            self.done = True
            self.status = "HOÀN THÀNH 5/5 màn" if ok else f"DỪNG ở màn {self.level}  (đạt tới màn {self.reached})"
            return
        self.level += 1
        self.idx, self.matches = 0, []
        self.start_match()


def draw_panel(p, font, small, W=PW, H=PH):
    g = p.game
    g.draw()
    surf = pygame.transform.smoothscale(g.screen, (W, H))
    if p.done:
        veil = pygame.Surface((W, H), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 170))
        surf.blit(veil, (0, 0))
        ok = p.reached >= N_LEVELS
        t = font.render(p.status, True, (120, 255, 150) if ok else (255, 130, 110))
        surf.blit(t, ((W - t.get_width()) // 2, 20))
        y = 60
        for lv, mine, path_d, wins, path_w, passed in p.history:
            line = (f"Màn {lv}: sát thương {mine:.0f} vs path {path_d:.0f}  "
                    f"({'qua' if passed else 'TRƯỢT'})  zombie thắng {wins} vs {path_w}")
            surf.blit(small.render(line, True, (230, 230, 230) if passed else (255, 150, 130)), (24, y))
            y += 22
    return surf


def main():
    ap = argparse.ArgumentParser(description="Xem nhiều thuật toán zombie đi qua các màn trong game thật")
    ap.add_argument("--algos", nargs="+", default=["ppo", "trpo", "a2c", "dqn"], choices=list(COL))
    ap.add_argument("--level", type=int, default=1, help="màn bắt đầu")
    ap.add_argument("--matches", type=int, default=5, help="số trận xem mỗi màn")
    ap.add_argument("--no-stop", action="store_true", help="không dừng khi không qua màn, chơi hết 5 màn")
    ap.add_argument("--zombie-seed", type=int, default=0, help="seed huấn luyện của model zombie (0-4)")
    ap.add_argument("--world-seed", type=int, default=1, help="seed của thế giới (cùng seed -> cùng điểm xuất hiện)")
    ap.add_argument("--agent", default=os.path.join("models", "player_agent.zip"))
    ap.add_argument("--windowed", action="store_true", help="chạy trong cửa sổ thay vì toàn màn hình")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--shot", default=None)
    args = ap.parse_args()

    pygame.init()
    panels = [Panel(a, args) for a in args.algos]
    cols = 2 if len(panels) > 1 else 1
    rows = (len(panels) + cols - 1) // cols
    size = (cols * PW, rows * (PH + BAR))
    # SCALED: giữ bố cục cố định rồi tự phóng to vừa màn hình (giữ tỉ lệ)
    flags = pygame.SCALED | (0 if args.windowed else pygame.FULLSCREEN)
    try:
        screen = pygame.display.set_mode(size, flags)
    except pygame.error:                       # không có renderer (vd. chạy kiểm tra headless)
        screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Zombie RL - các thuật toán qua từng màn trong game thật")
    font = pygame.font.SysFont("dejavusans,arial", 20, bold=True)
    small = pygame.font.SysFont("dejavusans,arial", 14)
    clock = pygame.time.Clock()
    speed, paused, sim, t0 = args.speed, False, 0.0, pygame.time.get_ticks()
    dt_fix = 1.0 / FPS
    running = True
    while running:
        real = min(clock.tick(30) / 1000.0, 0.1)
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed = min(speed * 1.5, 30)
                elif e.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed / 1.5, 0.25)
                elif e.key in (pygame.K_f, pygame.K_F11):
                    pygame.display.toggle_fullscreen()
                elif e.key == pygame.K_TAB:
                    for p in panels:
                        p.game.show_map = not p.game.show_map
        if not paused:
            sim += real * speed
            while sim >= dt_fix:
                sim -= dt_fix
                for p in panels:
                    p.step(dt_fix)
        screen.fill((10, 10, 14))
        for i, p in enumerate(panels):
            ox, oy = (i % cols) * PW, (i // cols) * (PH + BAR)
            screen.blit(draw_panel(p, font, small), (ox, oy + BAR))
            c = COL[p.algo]
            pygame.draw.rect(screen, c, (ox, oy, PW, 4))
            screen.blit(font.render(p.algo.upper(), True, c), (ox + 10, oy + 8))
            mid = f"màn {p.level}/{N_LEVELS} · trận {min(p.idx + 1, args.matches)}/{args.matches}"
            screen.blit(small.render(mid, True, (200, 205, 215)), (ox + 150, oy + 12))
            s = f"đạt tới màn {p.reached}/{N_LEVELS}"
            screen.blit(font.render(s, True, (235, 235, 235)), (ox + PW - font.size(s)[0] - 10, oy + 8))
        pygame.display.flip()
        if args.max_seconds is not None and (pygame.time.get_ticks() - t0) / 1000 > args.max_seconds:
            running = False
        if all(p.done for p in panels) and args.max_seconds is not None:
            running = False
    if args.shot:
        pygame.image.save(screen, args.shot)
    print("Kết quả (số trận ngắn, chỉ để nhìn):")
    for p in sorted(panels, key=lambda p: -p.reached):
        print(f"  {p.algo.upper():5} đạt tới màn {p.reached}/{N_LEVELS}  - {p.status or 'chưa xong'}")
    pygame.quit()


if __name__ == "__main__":
    main()
