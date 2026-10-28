"""Luật chơi của game - KHÔNG vẽ, KHÔNG cần pygame.

Cả game.py (người thật chơi) và player_env.py (RL agent đóng vai người chơi) đều chạy trên
lớp World này, nên agent chơi đúng y luật người thật chơi: cùng bản đồ, đợt zombie, súng,
nạp đạn, luật đánh của zombie, model PPO của zombie và nhịp ra quyết định.

Điều khiển zombie (controller):
  "ppo"   : model PPO đã train (models/zombie_lvK.zip) - giống game
  "path"  : viết tay, đi theo đường ngắn nhất (biết vòng qua tường, không né)
  "chase" : viết tay, đi thẳng về phía người chơi
"""
import math
import os
import random

import numpy as np

from combat import AttackState, BossMods
from config import (DECISION_DT, PLAYER, ZOMBIES, STUCK, WEAPONS, LEVELS, MAX_VIEW,
                    WAVE_NEXT_ALIVE, WAVE_MAX_GAP, HEAL_PER_WAVE, MODEL_DIR)
from geometry import (parse_map, cast_ray, move_with_collision, line_of_sight, resolve_shot,
                      in_fire_line, PathFinder)
from zombie_brain import build_obs, baseline_action, move_of, wants_attack, OBS_SIZE

SWITCH_TIME = 0.3
CONTROLLERS = ("ppo", "path", "chase")


class Controls:
    """Lệnh của người chơi trong một khung hình."""

    def __init__(self, forward=0, strafe=0, turn=0.0, fire=False, reload=False, switch=None):
        self.forward = forward      # -1 lùi, 0, 1 tiến
        self.strafe = strafe        # -1 trái, 0, 1 phải
        self.turn = turn            # góc quay (radian) trong khung hình này
        self.fire = fire            # giữ cò
        self.reload = reload        # bấm R
        self.switch = switch        # None hoặc tên súng muốn đổi sang


class Player:
    def __init__(self, x, y, level):
        self.x, self.y, self.angle = x, y, 0.0
        self.hp = float(PLAYER["hp"])
        self.vx = self.vy = 0.0
        self.sample_x, self.sample_y = x, y           # để tính vận tốc theo nhịp DECISION_DT
        self.unlocked = [k for k, w in WEAPONS.items() if w["unlock"] <= level]
        self.weapon = "m4a1" if "m4a1" in self.unlocked else "pistol"
        self.ammo = {k: w["mag"] for k, w in WEAPONS.items()}
        self.cooldown = self.reload_t = 0.0
        self.fire_t = 9.0
        self.switch_t = 0.0
        self.recoil = self.flash = self.hurt = self.hitmark = 0.0
        self.firing = False


class Zombie:
    def __init__(self, x, y, ztype):
        self.x, self.y, self.type = x, y, ztype
        self.st = ZOMBIES[ztype]
        self.hp = self.max_hp = float(self.st["hp"])
        self.radius = self.st["radius"]
        self.state, self.anim_t = "walk", 0.0
        self.action = 0
        self.path = (0.0, 0.0, 0.0)
        self.attack = AttackState(self.st)
        self.boss = BossMods() if ztype == "boss" else None
        self.charging = False
        self.flash = 0.0
        self.aimed = False
        self.track_x, self.track_y, self.track_t = x, y, 0.0
        self.move_t = 0.0               # thời gian zombie MUỐN đi (để phát hiện kẹt)
        self.unstuck_t = 0.0
        self.dmg_dealt = self.dmg_taken = 0.0
        self.whiffs = 0

    @property
    def alive(self):
        return self.state != "dead"


from numpy_policy import load_policy


class ModelBank:
    """Nạp model PPO của zombie một lần rồi dùng lại (nhiều World có thể dùng chung)."""

    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir, self.cache = model_dir, {}

    def get(self, level):
        """Model của màn này; nếu chưa có thì lấy model màn thấp hơn gần nhất."""
        for k in range(level, 0, -1):
            path = os.path.join(self.model_dir, f"zombie_lv{k}.zip")
            if not os.path.exists(path):
                continue
            if path not in self.cache:
                try:
                    m = load_policy(path)
                except OSError as e:                # không có torch và cũng chưa có .npz -> zombie viết tay
                    print(f"[!] {e}")
                    return None, None
                if m.observation_space.shape[0] != OBS_SIZE:
                    print(f"[!] {path} được train với observation cũ -> bỏ qua, hãy chạy train.py")
                    m = None
                self.cache[path] = m
            if self.cache[path] is not None:
                return self.cache[path], os.path.basename(path)
        return None, None


class World:
    def __init__(self, level=1, controller="ppo", seed=None, model_bank=None):
        assert controller in CONTROLLERS
        self.rng = random.Random(seed)
        self.controller = controller
        self.bank = model_bank or ModelBank()
        self.total_time = 0.0
        self.total_kills = 0
        self.decision_count = 0
        self.learner = None             # zombie đang được train (zombie_world_env.py)
        self.learner_action = 0
        self.load_level(level)

    # ---------- màn chơi ----------
    def load_level(self, level):
        self.level = level
        cfg = LEVELS[level - 1]
        self.grid, spawns, self.free = parse_map(cfg["map"])
        self.pf = PathFinder(self.grid)
        self.spawn_points = spawns.get("Z", []) + spawns.get("B", [])
        px, py = spawns["P"][0]
        self.player = Player(px, py, level)
        self.player.angle = max((i * math.pi / 2 for i in range(4)),
                                key=lambda a: cast_ray(self.grid, px, py, a, MAX_VIEW)[0])
        self.zombies = []
        self.waves, self.wave_idx, self.wave_t = cfg["waves"], 0, 0.0
        self.model = None
        if self.controller == "ppo":
            self.model, name = self.bank.get(level)
            self.ai_name = f"PPO {name}" if self.model else "đuổi thẳng (chưa có model - chạy train.py)"
        else:
            self.ai_name = {"path": "viết tay: đi theo đường", "chase": "viết tay: đuổi thẳng"}[self.controller]
        self.state, self.decision_t, self.play_time = "play", 0.0, 0.0
        self.unstuck_events = 0
        self.message, self.message_t = cfg["name"], 2.5
        if level == WEAPONS["m4a1"]["unlock"]:
            self.message = f"{cfg['name']}  -  Nhận được M4A1! (phím 2)"
            self.message_t = 3.5
        self.spawn_wave()

    def spawn_wave(self):
        """Thả đợt zombie tiếp theo ở các điểm xa người chơi, ưu tiên chỗ người chơi không nhìn thấy."""
        p = self.player
        wave = self.waves[self.wave_idx]
        self.wave_idx += 1
        self.wave_t = 0.0
        if self.wave_idx > 1 and HEAL_PER_WAVE:
            p.hp = min(PLAYER["hp"], p.hp + HEAL_PER_WAVE)
        cands = [(x, y) for x, y in self.spawn_points]
        cands += [(x + 0.5, y + 0.5) for x, y in self.free]
        cands = [c for c in cands if math.hypot(c[0] - p.x, c[1] - p.y) >= 6.0]
        self.rng.shuffle(cands)
        cands.sort(key=lambda c: (line_of_sight(self.grid, p.x, p.y, *c), c not in self.spawn_points))
        types = [t for t, n in wave.items() for _ in range(n)]
        used = []
        for t in types:
            pos = next((c for c in cands if all(math.hypot(c[0] - u[0], c[1] - u[1]) > 1.0 for u in used)),
                       cands[0] if cands else (p.x + 3, p.y))
            used.append(pos)
            self.zombies.append(Zombie(pos[0], pos[1], t))
        if self.wave_idx > 1:
            self.message, self.message_t = f"Đợt {self.wave_idx}/{len(self.waves)}", 1.5

    def player_weapon(self):
        return WEAPONS[self.player.weapon]

    # ---------- lệnh của người chơi ----------
    def switch_weapon(self, w):
        p = self.player
        if w in p.unlocked and w != p.weapon:
            p.weapon, p.reload_t = w, 0.0
            p.cooldown = max(p.cooldown, SWITCH_TIME)
            p.switch_t, p.fire_t = SWITCH_TIME, 9.0
            return True
        if w not in p.unlocked:
            self.message, self.message_t = f"{WEAPONS[w]['name']} mở khóa từ màn {WEAPONS[w]['unlock']}", 1.5
        return False

    def start_reload(self):
        p = self.player
        if p.reload_t <= 0 and p.ammo[p.weapon] < WEAPONS[p.weapon]["mag"]:
            p.reload_t = WEAPONS[p.weapon]["reload"]

    # ---------- một khung hình ----------
    def step(self, dt, ctrl):
        """Chạy luật game trong dt giây. Trả về các sự kiện xảy ra trong khung hình này."""
        ev = {"damage_dealt": 0.0, "damage_taken": 0.0, "kills": 0, "boss_killed": False,
              "shots": 0, "hits": 0}
        p = self.player
        for attr in ("recoil", "flash", "hurt", "hitmark", "switch_t"):
            setattr(p, attr, max(0.0, getattr(p, attr) - dt))
        p.fire_t += dt
        self.message_t = max(0.0, self.message_t - dt)
        for z in self.zombies:
            z.anim_t += dt
            z.flash = max(0.0, z.flash - dt)
        if self.state != "play":
            return ev
        self.play_time += dt
        self.total_time += dt

        if ctrl.switch:
            self.switch_weapon(ctrl.switch)
        if ctrl.reload:
            self.start_reload()
        p.angle += ctrl.turn
        ca, sa = math.cos(p.angle), math.sin(p.angle)
        mx, my = ca * ctrl.forward - sa * ctrl.strafe, sa * ctrl.forward + ca * ctrl.strafe
        n = math.hypot(mx, my)
        if n > 0:
            sp = PLAYER["move_speed"] * dt / n
            p.x, p.y = move_with_collision(self.grid, p.x, p.y, mx * sp, my * sp, PLAYER["radius"])

        self._update_weapon(dt, ctrl.fire, ev)

        # Zombie ra quyết định theo đúng nhịp lúc train
        self.decision_t += dt
        if self.decision_t >= DECISION_DT - 1e-6:
            p.vx, p.vy = (p.x - p.sample_x) / self.decision_t, (p.y - p.sample_y) / self.decision_t
            p.sample_x, p.sample_y = p.x, p.y
            self._decide_zombies()
            self.decision_count += 1
            self.decision_t = 0.0

        for z in self.zombies:
            if z.alive:
                self._update_zombie(z, dt, ev)
        self._separate_zombies()

        alive = sum(z.alive for z in self.zombies)
        self.wave_t += dt
        if self.wave_idx < len(self.waves) and (alive <= WAVE_NEXT_ALIVE or self.wave_t >= WAVE_MAX_GAP):
            self.spawn_wave()
            alive = sum(z.alive for z in self.zombies)

        if p.hp <= 0:
            p.hp, self.state = 0.0, "dead"
        elif alive == 0 and self.wave_idx >= len(self.waves):
            self.state = "win" if self.level == len(LEVELS) else "clear"
        return ev

    def _update_weapon(self, dt, fire, ev):
        p = self.player
        w = WEAPONS[p.weapon]
        p.cooldown -= dt
        p.firing = False
        if p.reload_t > 0:
            p.reload_t -= dt
            if p.reload_t <= 0:
                p.ammo[p.weapon] = w["mag"]
            return
        if p.ammo[p.weapon] <= 0:
            p.reload_t = w["reload"]                       # tự nạp khi hết băng
            return
        if fire and p.cooldown <= 0:
            p.ammo[p.weapon] -= 1
            p.cooldown = w["cooldown"]
            p.recoil, p.flash, p.firing, p.fire_t = 0.12, 0.06, True, 0.0
            ev["shots"] += 1
            alive = [z for z in self.zombies if z.alive]
            hit, _, _ = resolve_shot(self.grid, p.x, p.y, p.angle, w,
                                     [(z.x, z.y, z.radius) for z in alive], self.rng)
            if hit is not None:
                z = alive[hit]
                dealt = min(w["damage"], z.hp)
                z.dmg_taken += dealt
                z.hp -= w["damage"]
                z.flash, p.hitmark = 0.12, 0.15
                ev["hits"] += 1
                ev["damage_dealt"] += dealt
                if z.hp <= 0:
                    z.state, z.anim_t = "dead", 0.0
                    self.total_kills += 1
                    ev["kills"] += 1
                    ev["boss_killed"] |= z.type == "boss"

    def _decide_zombies(self):
        p = self.player
        active = [z for z in self.zombies if z.alive]
        if not active:
            return
        w = WEAPONS[p.weapon]
        for z in active:
            z.aimed = in_fire_line(self.grid, p.x, p.y, p.angle, z.x, z.y, z.radius, w)
            z.path = self.pf.step(z.x, z.y, p.x, p.y, z.radius)
        if self.controller in ("path", "chase") or self.model is None:
            kind = self.controller if self.controller != "ppo" else "chase"
            for z in active:
                z.action = baseline_action(kind, z.x, z.y, p.x, p.y, z.path,
                                           z.attack.ready, z.st["attack_range"])
        else:
            obs = np.stack([self.zombie_obs(z) for z in active])
            actions, _ = self.model.predict(obs, deterministic=True)
            for z, a in zip(active, np.atleast_1d(actions)):
                z.action = int(a)
        if self.learner is not None and self.learner.alive:
            self.learner.action = int(self.learner_action)

    def zombie_obs(self, z):
        """Observation của một zombie (giống hệt lúc train ở env_zombie.py)."""
        p = self.player
        w = WEAPONS[p.weapon]
        return build_obs(self.grid, z.path, z.x, z.y, z.radius, z.type, z.hp / z.max_hp,
                         p.x, p.y, p.vx, p.vy, p.angle, p.weapon,
                         p.reload_t > 0, p.ammo[p.weapon] / w["mag"], z.attack.ready)

    def _update_zombie(self, z, dt, ev):
        p = self.player
        st = z.st
        dist = math.hypot(p.x - z.x, p.y - z.y)
        los = line_of_sight(self.grid, z.x, z.y, p.x, p.y)

        # Vung tay - ra đòn - hồi chiêu (combat.py). Zombie PPO có thể "giữ đòn" để chờ thời cơ.
        was_attacking = z.attack.active
        busy, damage = z.attack.update(dt, dist, los, want=wants_attack(z.action))
        if z.attack.whiffed:
            z.whiffs += 1
        if damage > 0:
            taken = min(damage, max(p.hp, 0.0))
            p.hp -= damage
            p.hurt = 0.3
            ev["damage_taken"] += taken
            z.dmg_dealt += taken
        if busy and not was_attacking:
            z.anim_t = 0.0
        elif was_attacking and not busy:
            z.anim_t = 0.0

        speed = st["speed"]
        charging = False
        if z.boss is not None:
            mult, charging = z.boss.update(dt, z.hp / z.max_hp, dist, los)
            speed *= mult
        z.charging = charging and not busy

        if busy:
            # đang vung tay: chỉ nhích theo người chơi (lunge)
            dx, dy = (z.path[0], z.path[1]) if z.attack.winding_up else (0.0, 0.0)
            speed *= st["lunge"]
        else:
            dx, dy = move_of(z.action)
            # Lưới an toàn chống kẹt: zombie MUỐN đi mà gần như không nhúc nhích -> tạm đi theo đường.
            # (zombie cố ý đứng chờ thì không bị can thiệp). Áp dụng cho mọi controller.
            if dx or dy:
                z.move_t += dt
            z.track_t += dt
            if z.track_t >= STUCK["check_time"]:
                moved = math.hypot(z.x - z.track_x, z.y - z.track_y)
                if (z.move_t > 0.5 * z.track_t and moved < STUCK["min_move_frac"] * st["speed"] * z.move_t
                        and dist > st["attack_range"] + 0.5 and z.unstuck_t <= 0):
                    z.unstuck_t = STUCK["unstuck_time"]
                    self.unstuck_events += 1
                z.track_x, z.track_y, z.track_t, z.move_t = z.x, z.y, 0.0, 0.0
            if z.unstuck_t > 0:
                z.unstuck_t -= dt
                dx, dy = z.path[0], z.path[1]
            if z.charging:
                dx, dy = z.path[0], z.path[1]
        z.x, z.y = move_with_collision(self.grid, z.x, z.y, dx * speed * dt, dy * speed * dt, z.radius)

    def _separate_zombies(self):
        alive = [z for z in self.zombies if z.alive]
        for i, a in enumerate(alive):
            for b in alive[i + 1:]:
                dx, dy = b.x - a.x, b.y - a.y
                d = math.hypot(dx, dy)
                overlap = a.radius + b.radius - d
                if overlap > 0 and d > 1e-6:
                    ux, uy = dx / d * overlap / 2, dy / d * overlap / 2
                    a.x, a.y = move_with_collision(self.grid, a.x, a.y, -ux, -uy, a.radius)
                    b.x, b.y = move_with_collision(self.grid, b.x, b.y, ux, uy, b.radius)
