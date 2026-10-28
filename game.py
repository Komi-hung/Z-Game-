"""Game bắn zombie góc nhìn thứ nhất (raycasting kiểu Wolfenstein) với zombie điều khiển bằng PPO.

File này chỉ lo VẼ và NHẬN PHÍM. Toàn bộ luật chơi nằm trong game_core.py (lớp World),
dùng chung với RL agent đóng vai người chơi (player_env.py).

Điều khiển:
  Chuột: quay trái/phải, chuột trái: bắn (đạn bay theo phương ngang, không cần ngắm lên xuống)
  W/S tiến lùi, A/D đi ngang, ←/→ quay bằng phím, Space bắn bằng phím
  R nạp đạn, 1: súng lục, 2: M4A1 (nhận sau khi qua màn 2)
  Tab: bản đồ nhỏ + bảng RL, Enter: sang màn / chơi lại, Esc: thoát
  Hạ được boss ở màn 5 là thắng, game kết thúc.

Chạy: python game.py                    (bắt đầu từ màn 1, zombie PPO)
      python game.py --level 5          (vào thẳng màn boss)
      python game.py --zombies path     (đổi zombie sang bản viết tay để tự cảm nhận khác biệt)
      python game.py --watch-agent      (xem RL agent người chơi tự chơi)
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import pygame

from combat import RECOVER
from config import (RENDER_W, RENDER_H, WINDOW_SCALE, FOV, FPS, MAX_VIEW, MOUSE_SENSITIVITY,
                    PLAYER, BOSS, WEAPONS, LEVELS, SPRITE_WIDTH_FACTOR, LOG_PLAYTEST, PLAYTEST_DIR)
from game_core import World, Controls, SWITCH_TIME, CONTROLLERS
from geometry import cast_ray, line_of_sight, angle_diff
from sprites import ZombieSprites, WeaponSprites
from zombie_brain import ACTION_NAMES, move_of, wants_attack

PROJ = (RENDER_W / 2) / math.tan(FOV / 2)     # khoảng cách tới mặt phẳng chiếu
N_SHADE = 8
TYPE_COLORS = {"heavy": (170, 130, 255), "vest": (90, 200, 90), "skinny": (240, 200, 80), "boss": (255, 70, 70)}
WEAPON_KEYS = {pygame.K_1: "pistol", pygame.K_2: "m4a1"}


class Game:
    def __init__(self, start_level=1, demo=False, controller="ppo", agent_path=None):
        pygame.init()
        self.screen = pygame.display.set_mode((RENDER_W * WINDOW_SCALE, RENDER_H * WINDOW_SCALE))
        pygame.display.set_caption("Zombie RL")
        self.view = pygame.Surface((RENDER_W, RENDER_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 20, bold=True)
        self.font_big = pygame.font.SysFont("arial", 44, bold=True)
        self.font_small = pygame.font.SysFont("consolas,arial", 15)
        self.sprites = ZombieSprites()
        self.guns = WeaponSprites(WEAPONS)
        self.wall_tex = self._make_wall_textures()
        self.background = self._make_background()
        self.zbuf = [MAX_VIEW] * RENDER_W
        self.show_map = True
        self.demo = demo
        self.log_rows = []
        self.quit = False
        self.agent = None
        if agent_path:                                 # RL agent tự chơi thay người
            from player_env import PlayerAgentDriver
            try:
                self.agent = PlayerAgentDriver(agent_path)
            except OSError as e:                       # Windows chặn torch DLL -> chơi bằng tay
                print(f"[!] Không nạp được RL agent ({e}) -> chơi bằng tay")
        if not demo and self.agent is None:            # khóa chuột trong cửa sổ để quay bằng chuột
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            pygame.mouse.get_rel()
        self.world = World(start_level, controller)
        self.last_decision = 0

    # Chuyển tiếp thuộc tính để phần vẽ đọc thẳng trạng thái của World
    player = property(lambda self: self.world.player)
    zombies = property(lambda self: self.world.zombies)
    grid = property(lambda self: self.world.grid)
    level = property(lambda self: self.world.level)
    state = property(lambda self: self.world.state)
    waves = property(lambda self: self.world.waves)
    wave_idx = property(lambda self: self.world.wave_idx)
    message = property(lambda self: self.world.message)
    message_t = property(lambda self: self.world.message_t)
    ai_name = property(lambda self: self.world.ai_name)
    unstuck_events = property(lambda self: self.world.unstuck_events)
    total_time = property(lambda self: self.world.total_time)
    total_kills = property(lambda self: self.world.total_kills)
    play_time = property(lambda self: self.world.play_time)

    # ---------- tài nguyên ----------
    def _make_wall_textures(self):
        """Tường gạch xám tự sinh (64x64) + các mức tối dần theo khoảng cách."""
        path = os.path.join("assets", "wall.png")
        if os.path.exists(path):
            base = pygame.transform.scale(pygame.image.load(path).convert(), (64, 64))
        else:
            rs = np.random.RandomState(1)
            arr = (rs.rand(64, 64) * 40 + 70).astype(np.uint8)
            for y in range(64):
                if y % 16 in (0, 1):
                    arr[y, :] = 25
                off = 0 if (y // 16) % 2 == 0 else 16
                for x in range(64):
                    if (x + off) % 32 in (0, 1):
                        arr[y, x] = 25
            base = pygame.surfarray.make_surface(np.stack([arr.T] * 3, axis=-1))
        textures = {}
        for side in (0, 1):
            for lv in range(N_SHADE):
                s = base.copy()
                c = int(255 * (1.0 - lv / N_SHADE) * (1.0 if side == 0 else 0.75))
                s.fill((c, c, c), special_flags=pygame.BLEND_RGB_MULT)
                textures[side, lv] = s
        return textures

    def _make_background(self):
        bg = pygame.Surface((RENDER_W, RENDER_H))
        half = RENDER_H // 2
        for y in range(half):
            c = int(10 + 18 * y / half)
            pygame.draw.line(bg, (c, c, c + 4), (0, y), (RENDER_W, y))
        for y in range(half, RENDER_H):
            c = int(18 + 50 * (y - half) / half)
            pygame.draw.line(bg, (c, c, c), (0, y), (RENDER_W, y))
        return bg

    def load_level(self, level):
        self.save_log()
        self.world.load_level(level)
        if self.agent is not None:
            self.agent.reset()

    # ---------- vòng lặp ----------
    def run(self, demo_seconds=None, shot_path=None):
        running, t = True, 0.0
        while running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            if self.demo or self.agent is not None:
                dt = 1.0 / FPS                         # agent chơi với bước thời gian cố định như lúc train
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
                elif e.type == pygame.KEYDOWN:
                    self.on_key(e.key)
            self.update(dt)
            self.draw()
            pygame.display.flip()
            t += dt
            if self.quit or (demo_seconds is not None and t >= demo_seconds):
                running = False
        if shot_path:
            pygame.image.save(self.screen, shot_path)
        self.save_log()
        pygame.quit()

    def on_key(self, key):
        if key == pygame.K_TAB:
            self.show_map = not self.show_map
        elif key in WEAPON_KEYS:
            self.world.switch_weapon(WEAPON_KEYS[key])
        elif key == pygame.K_r:
            self.world.start_reload()
        elif key == pygame.K_RETURN:
            if self.state == "clear":
                self.load_level(self.level + 1)
            elif self.state == "dead":
                self.load_level(self.level)
            elif self.state == "win":                          # hạ boss xong: kết thúc game
                self.quit = True

    def update(self, dt):
        self.world.step(dt, self.read_input(dt))
        if self.world.decision_count != self.last_decision:
            self.last_decision = self.world.decision_count
            self.log_step()
        if self.agent is not None and self.state in ("clear", "dead"):
            self.agent.wait_t += dt                            # agent: tự sang màn / chơi lại sau 2 giây
            if self.agent.wait_t > 2.0:
                self.load_level(self.level + 1 if self.state == "clear" else self.level)

    def read_input(self, dt):
        if self.agent is not None:
            return self.agent.controls(self.world, dt)
        if self.demo:
            return self.autopilot(dt)
        k = pygame.key.get_pressed()
        mouse_dx = pygame.mouse.get_rel()[0] if pygame.mouse.get_focused() else 0
        turn = (k[pygame.K_RIGHT] - k[pygame.K_LEFT]) * PLAYER["turn_speed"] * dt + mouse_dx * MOUSE_SENSITIVITY
        return Controls(forward=k[pygame.K_w] - k[pygame.K_s], strafe=k[pygame.K_d] - k[pygame.K_a],
                        turn=turn, fire=bool(k[pygame.K_SPACE] or pygame.mouse.get_pressed()[0]))

    def autopilot(self, dt):
        """Chỉ dùng cho chế độ --demo: tự quay về zombie gần nhất và bắn."""
        p = self.player
        alive = [z for z in self.zombies if z.alive and line_of_sight(self.grid, p.x, p.y, z.x, z.y)]
        if not alive:
            return Controls(turn=PLAYER["turn_speed"] * dt)
        z = min(alive, key=lambda z: math.hypot(z.x - p.x, z.y - p.y))
        diff = angle_diff(math.atan2(z.y - p.y, z.x - p.x), p.angle)
        turn = max(-1.0, min(1.0, diff * 4)) * PLAYER["turn_speed"] * dt
        return Controls(strafe=1 if int(self.play_time) % 2 else -1, turn=turn, fire=abs(diff) < 0.1)

    # ---------- ghi dữ liệu playtest ----------
    def log_step(self):
        if not LOG_PLAYTEST or self.demo or self.agent is not None:
            return
        p = self.player
        self.log_rows.append([f"{self.play_time:.2f}", self.level, f"{p.x:.3f}", f"{p.y:.3f}",
                              f"{p.angle:.3f}", f"{p.vx:.3f}", f"{p.vy:.3f}", p.weapon, int(p.firing), p.hp])

    def save_log(self):
        if not getattr(self, "log_rows", None):
            return
        os.makedirs(PLAYTEST_DIR, exist_ok=True)
        path = os.path.join(PLAYTEST_DIR, time.strftime("session_%Y%m%d_%H%M%S.csv"))
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if f.tell() == 0:
                w.writerow(["time", "level", "x", "y", "angle", "vx", "vy", "weapon", "firing", "hp"])
            w.writerows(self.log_rows)
        self.log_rows = []

    # ======================= Vẽ =======================
    def draw(self):
        self.view.blit(self.background, (0, 0))
        self.draw_walls()
        self.draw_zombies()
        self.draw_gun()
        pygame.transform.scale(self.view, self.screen.get_size(), self.screen)
        self.draw_hud()

    def draw_walls(self):
        p = self.player
        half = RENDER_H // 2
        for col in range(RENDER_W):
            off = math.atan((col - RENDER_W / 2 + 0.5) / PROJ)
            dist, side, tex_u = cast_ray(self.grid, p.x, p.y, p.angle + off, MAX_VIEW)
            perp = max(dist * math.cos(off), 0.05)
            self.zbuf[col] = perp
            if dist >= MAX_VIEW:
                continue
            h = PROJ / perp
            shade = min(N_SHADE - 1, int(perp / MAX_VIEW * N_SHADE * 1.6))
            tex = self.wall_tex[side, shade]
            tx = min(63, int(tex_u * 64))
            if h <= RENDER_H:
                hi = max(1, int(h))
                self.view.blit(pygame.transform.scale(tex.subsurface((tx, 0, 1, 64)), (1, hi)),
                               (col, half - hi // 2))
            else:                                               # tường sát mặt: chỉ lấy phần nhìn thấy
                vis = RENDER_H / h * 64
                column = tex.subsurface((tx, int((64 - vis) / 2), 1, max(1, int(vis))))
                self.view.blit(pygame.transform.scale(column, (1, RENDER_H)), (col, 0))

    def zombie_frame(self, z):
        n = lambda kind: self.sprites.count(z.type, kind)
        if z.state == "dead":
            return "death", min(n("death") - 1, int(z.anim_t * 10))
        if z.attack.active:
            return "attack", min(n("attack") - 1, int(z.attack.t / (z.st["windup"] + RECOVER) * n("attack")))
        mx, my = move_of(z.action)
        moving = bool(mx or my) or z.charging or z.unstuck_t > 0
        fps = 10 * max(1.0, z.st["speed"] / 2.4)                # con chạy nhanh thì bước chân nhanh
        return "walk", (int(z.anim_t * fps) % n("walk")) if moving else 0

    def draw_zombies(self):
        p = self.player
        half = RENDER_H / 2
        order = sorted(self.zombies, key=lambda z: -math.hypot(z.x - p.x, z.y - p.y))
        for z in order:
            dx, dy = z.x - p.x, z.y - p.y
            ang = angle_diff(math.atan2(dy, dx), p.angle)
            if abs(ang) > FOV / 2 + 0.6:
                continue
            perp = math.hypot(dx, dy) * math.cos(ang)
            if perp < 0.2:
                continue
            kind, idx = self.zombie_frame(z)
            tint = (255, 160, 160) if (z.type == "boss" and z.hp < BOSS["phase2_hp"] * z.max_hp) else (255, 255, 255)
            img = self.sprites.frame(z.type, kind, idx, tint)
            ts = self.sprites.types[z.type]
            # Chiều cao thân trong thế giới suy ra từ radius: bề ngang hình vẽ ~ bề ngang va chạm,
            # nên zombie đứng sát tường không bị "lẹm" vào trong tường.
            body_world = 2 * z.radius * SPRITE_WIDTH_FACTOR / ts.aspect
            unit = PROJ / perp
            sh = unit * body_world * img.get_height() / ts.body_h
            sw = sh * img.get_width() / img.get_height()
            if sh < 1 or sh > RENDER_H * 3:
                continue
            img = pygame.transform.scale(img, (max(1, int(sw)), max(1, int(sh))))
            if z.flash > 0:
                img = img.copy()
                img.fill((120, 0, 0, 0), special_flags=pygame.BLEND_RGBA_ADD)
            self.blit_clipped(img, RENDER_W / 2 + math.tan(ang) * PROJ, half + unit * 0.5, perp)

    def blit_clipped(self, img, cx, bottom, perp):
        """Chỉ vẽ những cột của sprite không bị tường che (so với zbuffer)."""
        w, h = img.get_size()
        left = int(cx - w / 2)
        x0, x1 = max(0, left), min(RENDER_W, left + w)
        run = None
        for x in range(x0, x1 + 1):
            visible = x < x1 and perp < self.zbuf[x]
            if visible and run is None:
                run = x
            elif not visible and run is not None:
                self.view.blit(img, (run, int(bottom - h)), pygame.Rect(run - left, 0, x - run, h))
                run = None

    def gun_frame(self):
        """Chọn frame súng: đang nạp đạn -> animation reload, vừa bắn -> animation fire, còn lại -> ô 0."""
        p = self.player
        w = WEAPONS[p.weapon]
        if p.reload_t > 0:
            frames = self.guns.anims[p.weapon]["reload"]
            return frames[min(len(frames) - 1, int((1 - p.reload_t / w["reload"]) * len(frames)))]
        frames = self.guns.anims[p.weapon]["fire"]
        dur = max(w["cooldown"], 0.15)                     # M4A1 bắn nhanh: vẫn kịp thấy lửa đầu nòng
        if p.fire_t < dur:
            return frames[1 + min(len(frames) - 2, int(p.fire_t / dur * (len(frames) - 1)))]
        return frames[0]

    def draw_gun(self):
        p = self.player
        moving = abs(p.vx) + abs(p.vy) > 0.2
        t = self.play_time * 7
        bob_x = int(math.sin(t) * 4) if moving else 0
        bob_y = int(abs(math.cos(t)) * 4) if moving else 0
        drop = int(p.switch_t / SWITCH_TIME * 90)          # đổi súng: hạ súng xuống rồi đưa lên
        if self.guns.has(p.weapon):
            img = self.gun_frame()
            x = RENDER_W // 2 - img.get_width() // 2 + bob_x
            y = RENDER_H - img.get_height() + 6 + bob_y + drop
            self.view.blit(img, (x, y))
            return
        # Không có ảnh súng: vẽ tạm bằng hình khối
        ox, oy = RENDER_W // 2 - 10 + bob_x, RENDER_H - 52 + int(p.recoil * 60) + bob_y + drop
        pygame.draw.rect(self.view, (35, 35, 40), (ox + 6, oy - 14, 8, 30))
        pygame.draw.rect(self.view, (70, 70, 78), (ox, oy, 20, 40))
        pygame.draw.rect(self.view, (190, 130, 80), (ox - 2, oy + 30, 26, 30))
        if p.flash > 0:
            pygame.draw.circle(self.view, (255, 220, 90), (ox + 10, oy - 16), 7)

    def text(self, s, pos, font=None, color=(240, 240, 240), center=False):
        font = font or self.font
        img = font.render(s, True, color)
        x, y = pos
        if center:
            x -= img.get_width() // 2
        self.screen.blit(font.render(s, True, (0, 0, 0)), (x + 2, y + 2))
        self.screen.blit(img, (x, y))

    def draw_hud(self):
        p = self.player
        W, H = self.screen.get_size()
        if p.hurt > 0:
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((200, 0, 0, int(p.hurt / 0.3 * 90)))
            self.screen.blit(overlay, (0, 0))
        c = (255, 60, 60) if p.hitmark > 0 else (255, 255, 255)
        for a, b in (((-10, 0), (-4, 0)), ((4, 0), (10, 0)), ((0, -10), (0, -4)), ((0, 4), (0, 10))):
            pygame.draw.line(self.screen, c, (W // 2 + a[0], H // 2 + a[1]), (W // 2 + b[0], H // 2 + b[1]), 2)

        pygame.draw.rect(self.screen, (60, 0, 0), (20, H - 40, 200, 18))
        pygame.draw.rect(self.screen, (220, 40, 40), (20, H - 40, int(200 * max(p.hp, 0) / PLAYER["hp"]), 18))
        self.text(f"HP {int(p.hp)}", (24, H - 66))
        w = WEAPONS[p.weapon]
        ammo = "ĐANG NẠP..." if p.reload_t > 0 else f"{p.ammo[p.weapon]} / {w['mag']}"
        self.text(f"{w['name']}   {ammo}", (W - 300, H - 40))
        slots = "  ".join(f"{i}:{WEAPONS[k]['name'] if k in p.unlocked else '---'}"
                          for i, k in enumerate(WEAPONS, 1))
        self.text(slots, (W - 300, H - 64), self.font_small, (200, 200, 200))
        left = sum(z.alive for z in self.zombies)
        self.text(f"{LEVELS[self.level - 1]['name']}   Đợt {self.wave_idx}/{len(self.waves)}   "
                  f"Zombie còn: {left}", (20, 12))
        self.text(f"AI: {self.ai_name}", (20, 36), self.font_small, (180, 220, 255))
        boss = next((z for z in self.zombies if z.type == "boss" and z.alive), None)
        if boss:
            pygame.draw.rect(self.screen, (40, 0, 0), (W // 2 - 200, 66, 400, 14))
            pygame.draw.rect(self.screen, (230, 50, 50), (W // 2 - 200, 66, int(400 * boss.hp / boss.max_hp), 14))
            self.text("BOSS" + ("  - PHA 2" if boss.hp < BOSS["phase2_hp"] * boss.max_hp else ""),
                      (W // 2, 40), center=True)
        if self.show_map:
            self.draw_minimap()

        if self.message_t > 0 and self.state == "play":
            self.text(self.message, (W // 2, H // 3), self.font_big if len(self.message) < 30 else self.font,
                      (255, 240, 180), center=True)
        if self.state == "win":                            # màn hình kết thúc game
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 170))
            self.screen.blit(overlay, (0, 0))
            m, s = divmod(int(self.total_time), 60)
            self.text("CHIẾN THẮNG!", (W // 2, H // 2 - 110), self.font_big, (255, 220, 90), center=True)
            self.text("Boss đã bị hạ. Bạn đã sống sót qua cả 5 màn.", (W // 2, H // 2 - 40), center=True)
            self.text(f"Thời gian: {m} phút {s} giây    Zombie đã hạ: {self.total_kills}",
                      (W // 2, H // 2), center=True)
            self.text("Nhấn Enter hoặc Esc để thoát", (W // 2, H // 2 + 60), self.font_small,
                      (200, 200, 200), center=True)
            return
        msg = {"clear": "HOÀN THÀNH! Enter để sang màn tiếp",
               "dead": "BẠN ĐÃ CHẾT - Enter để chơi lại màn này"}.get(self.state)
        if msg:
            self.text(msg, (W // 2, H // 2 - 60), self.font, (255, 230, 120), center=True)

    def draw_minimap(self):
        """Bản đồ nhỏ + bảng RL: zombie chọn hướng nào, có đang bị ngắm hay không."""
        cell = 7
        gw, gh = len(self.grid[0]), len(self.grid)
        ox, oy = self.screen.get_width() - gw * cell - 14, 12
        surf = pygame.Surface((gw * cell, gh * cell), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 150))
        for y, row in enumerate(self.grid):
            for x, wall in enumerate(row):
                if wall:
                    pygame.draw.rect(surf, (120, 120, 130, 220), (x * cell, y * cell, cell, cell))
        p = self.player
        pc = (int(p.x * cell), int(p.y * cell))
        pygame.draw.circle(surf, (80, 180, 255), pc, 3)
        pygame.draw.line(surf, (80, 180, 255), pc,
                         (pc[0] + int(math.cos(p.angle) * 12), pc[1] + int(math.sin(p.angle) * 12)), 1)
        for z in self.zombies:
            if not z.alive:
                continue
            zc = (int(z.x * cell), int(z.y * cell))
            if z.aimed:
                pygame.draw.circle(surf, (255, 255, 255), zc, 5, 1)          # vòng trắng = đang bị ngắm
            pygame.draw.circle(surf, TYPE_COLORS[z.type], zc, 3)
            ax, ay = move_of(z.action)
            if not wants_attack(z.action):                                # đang giữ đòn, chờ thời cơ
                pygame.draw.circle(surf, (255, 140, 0), zc, 6, 1)
            pygame.draw.line(surf, (255, 255, 0), zc, (zc[0] + int(ax * 9), zc[1] + int(ay * 9)), 1)
        self.screen.blit(surf, (ox, oy))
        alive = [z for z in self.zombies if z.alive]
        ox = min(ox, self.screen.get_width() - 235)
        y = oy + gh * cell + 6
        if alive:
            z = min(alive, key=lambda z: math.hypot(z.x - p.x, z.y - p.y))
            self.text(f"Gần nhất: {z.st['name']}", (ox, y), self.font_small, (255, 255, 160))
            self.text(f"Action: {ACTION_NAMES[z.action]}", (ox, y + 18), self.font_small, (255, 255, 160))
            self.text(f"Bị ngắm: {'CÓ' if z.aimed else 'không'}", (ox, y + 36), self.font_small,
                      (255, 120, 120) if z.aimed else (160, 255, 160))
        self.text(f"Gỡ kẹt (lưới an toàn): {self.unstuck_events}", (ox, y + 54), self.font_small, (200, 200, 200))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--zombies", choices=CONTROLLERS, default="ppo",
                    help="ppo = zombie RL (mặc định), path / chase = zombie viết tay để so sánh")
    ap.add_argument("--watch-agent", nargs="?", const=os.path.join("models", "player_agent.zip"), default=None,
                    help="xem RL agent người chơi tự chơi (mặc định models/player_agent.zip)")
    ap.add_argument("--demo", type=float, default=None, help="tự chơi N giây (để kiểm tra)")
    ap.add_argument("--shot", default=None, help="lưu ảnh màn hình cuối khi chạy --demo")
    args = ap.parse_args()
    Game(args.level, demo=args.demo is not None, controller=args.zombies,
         agent_path=args.watch_agent).run(args.demo, args.shot)


if __name__ == "__main__":
    main()
