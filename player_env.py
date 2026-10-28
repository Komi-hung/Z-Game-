"""RL agent đóng vai NGƯỜI CHƠI - dùng để kiểm tra zombie RL mạnh tới đâu.

Agent chơi trên đúng luật game (game_core.World): cùng bản đồ, đợt zombie, súng, nạp đạn,
zombie PPO, 60 khung hình/giây. Nó chỉ được dùng các nút người chơi có:
tiến/lùi, đi ngang, quay (giới hạn tốc độ như dùng chuột), bắn, nạp đạn, đổi súng.

Một episode = một màn chơi trọn vẹn với 100 máu:
  - người chơi thắng: dọn hết các đợt zombie (màn 5: hạ boss)
  - zombie thắng:     người chơi hết máu
  - hòa (timeout):    quá LEVEL_TIME_LIMIT giây

Vai trò: zombie đã luyện xong là "chính sách cần kiểm tra"; agent này là đối thủ được luyện riêng để tìm cách
thắng nó. Đối thủ càng khó thắng thì chính sách của zombie càng tốt.
"""
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import WEAPONS, LEVELS, PLAYER
from game_core import World, Controls, ModelBank, CONTROLLERS
from geometry import cast_ray, line_of_sight, in_fire_line, angle_diff
from zombie_brain import ZOMBIE_TYPES, move_of

FRAME_DT = 1.0 / 60          # bước thời gian của game
AGENT_DT = 0.1               # agent ra quyết định mỗi 0.1 giây (giống phản xạ người)
FRAMES_PER_ACTION = round(AGENT_DT / FRAME_DT)
MAX_TURN_RATE = 6.0          # rad/giây: quay nhanh như chuột nhưng có giới hạn
TURN_BINS = np.array([-1.0, -0.4, -0.12, -0.04, 0.0, 0.04, 0.12, 0.4, 1.0])
LEVEL_TIME_LIMIT = 240.0     # giây
K_ZOMBIES = 6                # agent theo dõi 6 zombie gần nhất
N_RAYS = 16
RAY_MAX = 10.0
WEAPON_LIST = list(WEAPONS)  # pistol, m4a1

PER_ZOMBIE = 14
OBS_SIZE = 10 + 3 + N_RAYS + K_ZOMBIES * PER_ZOMBIE

# Reward của agent người chơi
P_REWARD = dict(
    damage_dealt=0.5,   # x (damage / 100)
    kill=1.0,
    boss_kill=5.0,
    damage_taken=-1.0,  # x (damage / 100)
    death=-3.0,
    clear=3.0,          # dọn xong màn
    miss=-0.005,        # mỗi phát bắn trượt (đỡ bắn bừa)
    aim=0.003,          # mỗi bước có zombie nằm trong đường ngắm (giúp học ngắm nhanh hơn)
)


def _single_thread_torch():
    """Mỗi env (thường chạy trong một tiến trình riêng) chỉ cần 1 luồng để chạy model zombie.
    Để mặc định, PyTorch ở mọi tiến trình cùng giành hết nhân CPU -> chậm hơn nhiều lần."""
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def build_player_obs(world):
    """Observation của agent người chơi, mọi hướng tính THEO HƯỚNG NHÌN của người chơi."""
    p = world.player
    w = WEAPONS[p.weapon]
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    i = 0
    # --- bản thân (10) ---
    obs[i:i + 10] = (p.hp / PLAYER["hp"], p.ammo[p.weapon] / w["mag"], float(p.reload_t > 0),
                     max(p.reload_t, 0.0) / w["reload"], float(p.cooldown <= 0), float(p.switch_t > 0),
                     float(p.weapon == WEAPON_LIST[0]), float(p.weapon == WEAPON_LIST[1]),
                     float(len(p.unlocked) > 1), world.level / len(LEVELS))
    i += 10
    # --- tiến độ màn (3) ---
    alive = [z for z in world.zombies if z.alive]
    obs[i:i + 3] = (world.wave_idx / len(world.waves), min(len(alive) / 10.0, 1.0),
                    (len(world.waves) - world.wave_idx) / 5.0)
    i += 3
    # --- tường xung quanh (16 tia, tia 0 = hướng đang nhìn) ---
    for k in range(N_RAYS):
        obs[i] = cast_ray(world.grid, p.x, p.y, p.angle + k * 2 * math.pi / N_RAYS, RAY_MAX)[0] / RAY_MAX
        i += 1
    # --- 6 zombie gần nhất (14 mỗi con) ---
    alive.sort(key=lambda z: (z.x - p.x) ** 2 + (z.y - p.y) ** 2)
    ca, sa = math.cos(-p.angle), math.sin(-p.angle)
    for z in alive[:K_ZOMBIES]:
        dx, dy = z.x - p.x, z.y - p.y
        d = math.hypot(dx, dy)
        rel = angle_diff(math.atan2(dy, dx), p.angle)       # 0 = ngay trước mặt
        ax, ay = move_of(z.action)
        mvx, mvy = ax * ca - ay * sa, ax * sa + ay * ca     # hướng zombie đang đi, theo khung người chơi
        feat = [1.0, math.sin(rel), math.cos(rel), min(d / 16.0, 1.0),
                float(line_of_sight(world.grid, p.x, p.y, z.x, z.y)),
                float(in_fire_line(world.grid, p.x, p.y, p.angle, z.x, z.y, z.radius, w))]
        feat += [float(z.type == t) for t in ZOMBIE_TYPES]
        feat += [z.hp / z.max_hp, float(z.attack.active), mvx, mvy]
        obs[i:i + PER_ZOMBIE] = feat
        i += PER_ZOMBIE
    return np.clip(obs, -1.0, 1.0)


def decode_action(action, player):
    """MultiDiscrete -> (tiến, ngang, tốc độ quay rad/s, bắn, nạp, súng muốn đổi)."""
    fwd, strafe, turn, fire, reload_, switch = (int(a) for a in action)
    target = None
    if switch:
        others = [w for w in player.unlocked if w != player.weapon]
        target = others[0] if others else None
    return fwd - 1, strafe - 1, TURN_BINS[turn] * MAX_TURN_RATE, bool(fire), bool(reload_), target


def outcome_of(world, timed_out):
    if world.state in ("clear", "win"):
        return "player_win"
    if world.state == "dead":
        return "zombie_win"
    return "timeout" if timed_out else None


class PlayerEnv(gym.Env):
    """Gymnasium env: agent = người chơi, đối thủ = zombie (PPO hoặc viết tay)."""
    metadata = {"render_modes": []}

    def __init__(self, levels=(1, 2, 3, 4, 5), controllers=CONTROLLERS, controller_probs=(1 / 3, 1 / 3, 1 / 3),
                 zombie_dir=None):
        super().__init__()
        _single_thread_torch()
        self.levels = list(levels)
        self.controllers = list(controllers)
        p = np.array(controller_probs[:len(self.controllers)], dtype=float)
        self.controller_probs = p / p.sum()
        self.bank = ModelBank(zombie_dir) if zombie_dir else ModelBank()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 3, len(TURN_BINS), 2, 2, 2])
        self.fixed = None          # (level, controller, seed) khi đánh giá

    def set_match(self, level, controller, seed):
        """Dùng khi đánh giá: lần reset tiếp theo chơi đúng màn / loại zombie / seed này."""
        self.fixed = (level, controller, seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        if self.fixed is not None:
            level, controller, world_seed = self.fixed
        else:
            level = int(rng.choice(self.levels))
            controller = str(rng.choice(self.controllers, p=self.controller_probs))
            world_seed = int(rng.integers(1 << 31))
        self.world = World(level, controller, seed=world_seed, model_bank=self.bank)
        return build_player_obs(self.world), {"level": level, "controller": controller}

    def step(self, action):
        world = self.world
        fwd, strafe, turn_rate, fire, reload_, switch = decode_action(action, world.player)
        total = {"damage_dealt": 0.0, "damage_taken": 0.0, "kills": 0, "boss_killed": False, "shots": 0, "hits": 0}
        for f in range(FRAMES_PER_ACTION):
            ctrl = Controls(fwd, strafe, turn_rate * FRAME_DT, fire,
                            reload=reload_ and f == 0, switch=switch if f == 0 else None)
            ev = world.step(FRAME_DT, ctrl)
            for k in total:
                if k == "boss_killed":
                    total[k] = total[k] or ev[k]
                else:
                    total[k] += ev[k]
            if world.state != "play":
                break

        r = (P_REWARD["damage_dealt"] * total["damage_dealt"] / 100.0
             + P_REWARD["kill"] * total["kills"]
             + P_REWARD["damage_taken"] * total["damage_taken"] / 100.0
             + P_REWARD["miss"] * (total["shots"] - total["hits"]))
        if total["boss_killed"]:
            r += P_REWARD["boss_kill"]
        if any(z.alive and z.aimed for z in world.zombies):
            r += P_REWARD["aim"]
        terminated = world.state != "play"
        if world.state == "dead":
            r += P_REWARD["death"]
        elif world.state in ("clear", "win"):
            r += P_REWARD["clear"]
        truncated = not terminated and world.play_time >= LEVEL_TIME_LIMIT
        info = {"outcome": outcome_of(world, truncated), "level": world.level,
                "controller": world.controller, "kills": world.total_kills,
                "hp": world.player.hp, "time": world.play_time}
        return build_player_obs(world), float(r), terminated, truncated, info


class PlayerAgentDriver:
    """Cho agent đã train điều khiển người chơi trong game.py (xem agent chơi: --watch-agent)."""

    def __init__(self, path):
        from numpy_policy import load_policy
        self.model = load_policy(path)
        self.reset()

    def reset(self):
        self.t = AGENT_DT
        self.wait_t = 0.0
        self.cmd = (0, 0, 0.0, False, False, None)
        self.fresh = False

    def controls(self, world, dt):
        self.t += dt
        if self.t >= AGENT_DT - 1e-6:
            self.t = 0.0
            action, _ = self.model.predict(build_player_obs(world), deterministic=True)
            self.cmd = decode_action(action, world.player)
            self.fresh = True
        fwd, strafe, turn_rate, fire, reload_, switch = self.cmd
        first = self.fresh
        self.fresh = False
        return Controls(fwd, strafe, turn_rate * dt, fire, reload=reload_ and first,
                        switch=switch if first else None)


class ScriptedPlayer:
    """Người chơi viết tay (dùng làm đối chứng khi đánh giá): ngắm zombie gần nhất nhìn thấy được,
    bắn khi đã vào đường ngắm, lùi lại khi bị áp sát. Dùng ĐÚNG không gian hành động như agent RL
    (cùng giới hạn tốc độ quay, cùng nhịp 0.1 giây) để so sánh công bằng."""

    def act(self, world):
        p = world.player
        alive = [z for z in world.zombies if z.alive]
        if not alive:
            return np.array([1, 1, 4, 0, 0, 0])
        visible = [z for z in alive if line_of_sight(world.grid, p.x, p.y, z.x, z.y)]
        target = min(visible or alive, key=lambda z: math.hypot(z.x - p.x, z.y - p.y))
        dist = math.hypot(target.x - p.x, target.y - p.y)
        want = angle_diff(math.atan2(target.y - p.y, target.x - p.x), p.angle)
        turn = int(np.argmin(np.abs(TURN_BINS * MAX_TURN_RATE * AGENT_DT - want)))
        w = WEAPONS[p.weapon]
        fire = int(bool(visible) and in_fire_line(world.grid, p.x, p.y, p.angle, target.x, target.y,
                                                  target.radius, w))
        fwd = 0 if dist < 3.0 else 1                                  # 0 = lùi, 1 = đứng yên
        strafe = 0 if int(world.play_time * 1.5) % 2 else 2           # đi ngang qua lại để né
        reload_ = int(not visible and p.ammo[p.weapon] < w["mag"] * 0.5)
        switch = int("m4a1" in p.unlocked and p.weapon != "m4a1")
        return np.array([fwd, strafe, turn, fire, reload_, switch])
