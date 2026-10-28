"""Môi trường RL (Gymnasium) - "sân tập" của zombie. Không vẽ gì để train nhanh.

Mỗi episode: 1 zombie đấu với 1 người chơi giả (PlayerBot) trên bản đồ của màn chơi.
  - Zombie thường: episode kết thúc khi ĐÁNH TRÚNG người chơi lần đầu (hoặc bị bắn chết / hết giờ).
  - Zombie được chọn "giữ đòn" (action 9-17) để chờ thời cơ; vung trượt bị phạt nhẹ.
  - Boss: trận tay đôi đầy đủ. Người chơi giả có máu, boss vung tay - hồi chiêu đúng như trong game
    (combat.py), có pha 2 và lao tới; trận kết thúc khi một bên chết hoặc hết giờ.
Một model PPO được dùng chung cho mọi zombie trong game.
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from combat import AttackState, BossMods
from config import (LEVELS, ZOMBIES, WEAPONS, REWARD, DECISION_DT, MAX_EPISODE_STEPS, DUEL_MAX_STEPS,
                    DUEL_TYPES, REPLAY_PROB)
from geometry import parse_map, move_with_collision, in_fire_line, line_of_sight, PathFinder
from player_bot import PlayerBot
from zombie_brain import N_ACTIONS, OBS_SIZE, build_obs, move_of, wants_attack

MAX_SUBSTEP = 0.1   # mỗi lần dịch chuyển tối đa 0.1 ô, giống game chạy 60 FPS


def move_zombie(grid, x, y, dx, dy, radius):
    """Chia bước đi dài thành nhiều bước nhỏ để va chạm tường giống hệt trong game."""
    n = max(1, math.ceil(math.hypot(dx, dy) / MAX_SUBSTEP))
    for _ in range(n):
        x, y = move_with_collision(grid, x, y, dx / n, dy / n, radius)
    return x, y


def available_weapons(level):
    return [k for k, w in WEAPONS.items() if w["unlock"] <= level]


class ZombieEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level=1, replay_prob=REPLAY_PROB):
        super().__init__()
        self.level, self.replay_prob = level, replay_prob
        self.cfg = LEVELS[level - 1]
        self.maps = {}
        for k in range(1, level + 1):
            grid, _, free = parse_map(LEVELS[k - 1]["map"])
            self.maps[k] = (grid, free, PathFinder(grid))
        self.types = list(self.cfg["train_mix"])
        p = np.array([self.cfg["train_mix"][t] for t in self.types])
        self.type_probs = p / p.sum()
        self.weapons = available_weapons(level)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)

    def _obs(self):
        b = self.bot
        return build_obs(self.grid, self.path, self.zx, self.zy, self.zr, self.ztype, self.hp / self.max_hp,
                         b.x, b.y, b.vx, b.vy, b.angle, b.weapon_name,
                         b.reloading, b.ammo_frac, self.attack.ready)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        lv = self.level
        if self.level > 1 and rng.random() < self.replay_prob:      # ôn lại bản đồ màn trước
            lv = int(rng.integers(1, self.level))
        self.grid, free, self.pf = self.maps[lv]

        px, py = free[int(rng.integers(len(free)))]
        px, py = px + 0.5, py + 0.5
        for _ in range(100):
            zx, zy = free[int(rng.integers(len(free)))]
            zx, zy = zx + 0.5, zy + 0.5
            if math.hypot(zx - px, zy - py) >= 4.0:
                break
        self.zx, self.zy = zx, zy

        self.ztype = str(rng.choice(self.types, p=self.type_probs))
        self.st = ZOMBIES[self.ztype]
        self.zr, self.speed = self.st["radius"], self.st["speed"]
        self.max_hp = self.hp = float(self.st["hp"])
        self.duel = self.ztype in DUEL_TYPES
        self.max_steps = DUEL_MAX_STEPS if self.duel else MAX_EPISODE_STEPS
        self.attack = AttackState(self.st)
        self.boss = BossMods() if self.ztype == "boss" else None

        weapon = str(rng.choice(self.weapons))
        kiting = rng.random() < self.cfg["bot"].get("kite", 0.0)
        self.bot = PlayerBot(self.grid, px, py, rng.uniform(-math.pi, math.pi), weapon, self.cfg["bot"], rng,
                             pathfinder=self.pf, kiting=kiting)
        self.steps = self.aimed_steps = self.stuck_steps = self.hits_dealt = 0
        self.path = self.pf.step(self.zx, self.zy, px, py, self.zr)
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        b = self.bot
        dist = math.hypot(b.x - self.zx, b.y - self.zy)
        reward = REWARD["step"]

        # ----- zombie hành động -----
        action = int(action)
        los = line_of_sight(self.grid, self.zx, self.zy, b.x, b.y)
        busy, dealt = self.attack.update(DECISION_DT, dist, los, want=wants_attack(action))
        if self.attack.whiffed:
            reward += REWARD["whiff"]
        speed = self.speed
        charging = False
        if self.boss is not None:
            mult, charging = self.boss.update(DECISION_DT, self.hp / self.max_hp, dist, los)
            speed *= mult
        if busy:
            # đang vung tay: chỉ nhích theo người chơi (lunge), không tự đi
            ax, ay = (self.path[0], self.path[1]) if self.attack.winding_up else (0.0, 0.0)
            speed *= self.st["lunge"]
        elif charging:
            ax, ay = self.path[0], self.path[1]           # lao tới: đi theo đường tìm được
        else:
            ax, ay = move_of(action)
        dx, dy = ax * speed * DECISION_DT, ay * speed * DECISION_DT
        ox, oy = self.zx, self.zy
        self.zx, self.zy = move_zombie(self.grid, self.zx, self.zy, dx, dy, self.zr)
        if not busy and (ax or ay) and math.hypot(self.zx - ox, self.zy - oy) < 0.3 * math.hypot(dx, dy):
            self.stuck_steps += 1
        if dealt > 0:
            b.hp -= dealt
            self.hits_dealt += 1
            reward += REWARD["hit_dealt"] if self.duel else 0.0

        # ----- người chơi giả hành động -----
        damage = b.step(DECISION_DT, self.zx, self.zy, self.zr)
        self.hp -= damage
        prev_path = self.path[2]
        self.path = self.pf.step(self.zx, self.zy, b.x, b.y, self.zr)
        dist = math.hypot(b.x - self.zx, b.y - self.zy)

        # ----- trọng tài chấm điểm -----
        # Thưởng theo quãng ĐƯỜNG ĐI rút ngắn được: lao vào tường không được điểm.
        reward += REWARD["approach"] * (prev_path - self.path[2])
        if in_fire_line(self.grid, b.x, b.y, b.angle, self.zx, self.zy, self.zr, b.weapon):
            reward += REWARD["aimed"]
            self.aimed_steps += 1
        if damage > 0:
            reward += REWARD["hit_taken"] * damage / self.max_hp

        terminated, outcome = False, None
        if self.hp <= 0:
            reward += REWARD["death"]
            terminated, outcome = True, "killed"
        elif self.duel and b.hp <= 0:
            reward += REWARD["kill_player"]
            terminated, outcome = True, "won"
        elif not self.duel and self.hits_dealt > 0:
            reward += REWARD["catch"]
            terminated, outcome = True, "caught"
        truncated = self.steps >= self.max_steps
        if truncated and not terminated:
            outcome = "timeout"
        info = {"outcome": outcome, "aimed_frac": self.aimed_steps / self.steps,
                "stuck_frac": self.stuck_steps / self.steps,
                "hp_lost": 1.0 - max(self.hp, 0.0) / self.max_hp,
                "hits_dealt": self.hits_dealt}
        return self._obs(), float(reward), terminated, truncated, info
