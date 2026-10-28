"""Phần RL dùng chung: zombie "nhìn thấy" gì (observation) và "làm được" gì (action).

Env lúc train và game lúc chơi đều gọi build_obs() này, nên đầu vào của model luôn giống nhau.
Nếu sửa file này (đổi kích thước observation hoặc số action) thì phải train lại từ đầu.
"""
import math
import numpy as np

from config import WEAPONS, PLAYER
from geometry import cast_ray, in_fire_line

# 9 hành động theo hướng TUYỆT ĐỐI trên bản đồ: đứng yên + 8 hướng.
_D = math.sqrt(0.5)
ACTIONS = np.array([
    (0, 0),
    (0, -1), (0, 1), (-1, 0), (1, 0),
    (-_D, -_D), (_D, -_D), (-_D, _D), (_D, _D),
], dtype=np.float64)
MOVE_NAMES = ["đứng", "lên", "xuống", "trái", "phải", "chéo↖", "chéo↗", "chéo↙", "chéo↘"]
# 18 action = 9 hướng đi x {cho phép đánh, GIỮ ĐÒN}.
#   0-8 : đi theo hướng đó, người chơi vào tầm thì vung tay (như zombie bình thường)
#   9-17: đi theo hướng đó nhưng CHƯA vung tay -> chờ thời cơ, vờn quanh, ép góc rồi mới đánh
N_MOVES = len(ACTIONS)
N_ACTIONS = 2 * N_MOVES
ACTION_NAMES = MOVE_NAMES + [n + " (giữ đòn)" for n in MOVE_NAMES]

ZOMBIE_TYPES = ["heavy", "vest", "skinny", "boss"]
WEAPON_NAMES = list(WEAPONS)          # pistol, m4a1
N_RAYS, RAY_MAX = 8, 8.0
RAY_ANGLES = [i * 2 * math.pi / N_RAYS for i in range(N_RAYS)]
PATH_NORM = 30.0

# 8 tia + hướng thẳng tới người chơi (2) + khoảng cách thẳng (1)
# + HƯỚNG ĐƯỜNG ĐI vòng qua tường (2) + độ dài đường đi (1)
# + vận tốc người chơi (2) + hướng nhìn người chơi (2) + cờ "đang bị ngắm" (1)
# + loại zombie (4) + súng (2) + máu (1)
# + người chơi đang nạp đạn (1) + % đạn còn lại (1) + đòn đánh của mình đã sẵn sàng (1)
OBS_SIZE = N_RAYS + 3 + 3 + 2 + 2 + 1 + len(ZOMBIE_TYPES) + len(WEAPON_NAMES) + 1 + 3


def build_obs(grid, path, zx, zy, z_radius, z_type, hp_frac, px, py, pvx, pvy, p_angle, weapon_name,
              p_reloading, p_ammo_frac, attack_ready):
    """path = (ux, uy, quãng đường) lấy từ PathFinder.step()."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    i = 0
    for a in RAY_ANGLES:
        obs[i] = cast_ray(grid, zx, zy, a, RAY_MAX)[0] / RAY_MAX
        i += 1
    rx, ry = px - zx, py - zy
    d = math.hypot(rx, ry) + 1e-8
    obs[i:i + 3] = (rx / d, ry / d, min(d / 16.0, 1.0)); i += 3
    ux, uy, pd = path
    obs[i:i + 3] = (ux, uy, min(pd / PATH_NORM, 1.0)); i += 3
    v = PLAYER["move_speed"]
    obs[i:i + 2] = (pvx / v, pvy / v); i += 2
    obs[i:i + 2] = (math.cos(p_angle), math.sin(p_angle)); i += 2
    obs[i] = 1.0 if in_fire_line(grid, px, py, p_angle, zx, zy, z_radius, WEAPONS[weapon_name]) else 0.0; i += 1
    obs[i + ZOMBIE_TYPES.index(z_type)] = 1.0; i += len(ZOMBIE_TYPES)
    obs[i + WEAPON_NAMES.index(weapon_name)] = 1.0; i += len(WEAPON_NAMES)
    obs[i] = hp_frac; i += 1
    obs[i:i + 3] = (1.0 if p_reloading else 0.0, p_ammo_frac, 1.0 if attack_ready else 0.0)
    return np.clip(obs, -1.0, 1.0)


def nearest_action(dx, dy):
    """Action rời rạc gần nhất với một hướng (dx, dy)."""
    if dx == 0 and dy == 0:
        return 0
    return int(np.argmax(ACTIONS[1:] @ np.array([dx, dy])) + 1)


def move_of(action):
    """Vector di chuyển của một action."""
    return ACTIONS[action % N_MOVES]


def wants_attack(action):
    """Action 0-8 cho phép vung tay khi người chơi trong tầm, 9-17 là giữ đòn."""
    return action < N_MOVES


def chase_action(zx, zy, px, py):
    """Baseline 1: đi thẳng về phía người chơi (không né, không biết đi vòng)."""
    return nearest_action(px - zx, py - zy)


def baseline_action(kind, zx, zy, px, py, path, attack_ready=True, attack_range=0.0):
    """Zombie viết tay: đi thẳng (chase) hoặc đi theo đường (path), vào tầm là đánh ngay
    (action 0-8 luôn cho phép đánh)."""
    if kind == "path":
        return nearest_action(path[0], path[1])
    return chase_action(zx, zy, px, py)
