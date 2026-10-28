"""Người chơi giả dùng lúc TRAIN (không dùng trong game).

Bot di chuyển như người thật trong game: WASD (tiến/lùi/strafe) + quay nhanh như dùng chuột.
Bot nhìn thấy zombie trễ vài bước (reaction) và ngắm có sai số (aim_noise),
nên zombie có thể học cách né: đổi hướng ngay khi bị ngắm, vì bot bắn vào vị trí cũ.
"""
import math
from collections import deque

from config import PLAYER, WEAPONS
from geometry import angle_diff, line_of_sight, move_with_collision, resolve_shot

# Khoảng cách bot muốn giữ với zombie, theo loại súng
PREFERRED_DIST = {"pistol": 5.0, "m4a1": 5.0}


class PlayerBot:
    @property
    def reloading(self):
        return self.reload_t > 0

    @property
    def ammo_frac(self):
        return self.ammo / self.weapon["mag"]

    def __init__(self, grid, x, y, angle, weapon_name, skill, rng, pathfinder=None, kiting=False):
        self.grid, self.x, self.y, self.angle = grid, x, y, angle
        self.pf, self.kiting = pathfinder, kiting
        self.kite_dir = (0.0, 0.0)
        self.weapon_name, self.weapon = weapon_name, WEAPONS[weapon_name]
        self.skill, self.rng = skill, rng
        self.vx = self.vy = 0.0
        self.ammo, self.cooldown, self.reload_t = self.weapon["mag"], 0.0, 0.0
        self.hp = float(PLAYER["hp"])
        self.seen = deque(maxlen=skill["reaction"] + 1)   # trí nhớ có độ trễ
        self.aim_err, self.aim_timer = 0.0, 0.0
        self.forward = self.strafe = 0
        self.move_timer = 0.0

    def step(self, dt, zx, zy, zr):
        """Cập nhật bot một bước. Trả về damage gây cho zombie (0 nếu trượt)."""
        self.seen.append((zx, zy))
        sx, sy = self.seen[0]                              # vị trí zombie mà bot "thấy" (trễ)
        sees = line_of_sight(self.grid, self.x, self.y, sx, sy)
        dist = math.hypot(sx - self.x, sy - self.y)

        # Quay mượt về phía zombie, có sai số ngắm thay đổi theo thời gian
        self.aim_timer -= dt
        if self.aim_timer <= 0:
            self.aim_err = self.rng.normal(0.0, self.skill["aim_noise"])
            self.aim_timer = self.rng.uniform(0.3, 0.8)
        max_turn = PLAYER["bot_turn_speed"] * self.skill["turn"] * dt
        if sees:
            target = math.atan2(sy - self.y, sx - self.x) + self.aim_err
            self.angle += max(-max_turn, min(max_turn, angle_diff(target, self.angle)))
        else:
            self.angle += max_turn * 0.5                   # không thấy thì quay tìm

        ox, oy = self.x, self.y
        if self.kiting and self.pf is not None and dist < PREFERRED_DIST[self.weapon_name] + 2:
            self._kite(dt, sx, sy)                         # thả diều: vừa chạy vừa bắn
        else:
            self._wander(dt, dist)
        self.vx, self.vy = (self.x - ox) / dt, (self.y - oy) / dt
        return self._shoot(dt, sees, dist, sx, sy, zx, zy, zr)

    def _kite(self, dt, sx, sy):
        """Chọn hướng chạy làm ĐƯỜNG ĐI từ zombie tới mình dài nhất (chạy vòng quanh cột,
        không chạy vào ngõ cụt). Hướng nhìn vẫn giữ về phía zombie để bắn, giống người dùng WASD + chuột."""
        field = self.pf.field(int(sx), int(sy))
        sp = PLAYER["move_speed"] * dt
        best, best_score = None, -math.inf
        for k in range(8):
            a = k * math.pi / 4
            ux, uy = math.cos(a), math.sin(a)
            nx, ny = move_with_collision(self.grid, self.x, self.y, ux * sp, uy * sp, PLAYER["radius"])
            if math.hypot(nx - self.x, ny - self.y) < 0.5 * sp:
                continue
            score = (field[int(ny)][int(nx)] + 0.5 * math.hypot(nx - sx, ny - sy)
                     + 0.3 * (ux * self.kite_dir[0] + uy * self.kite_dir[1]))   # ưu tiên giữ hướng cũ
            if score > best_score:
                best, best_score = (nx, ny, ux, uy), score
        if best is not None:
            self.x, self.y, ux, uy = best
            self.kite_dir = (ux, uy)

    def _wander(self, dt, dist):
        """Di chuyển kiểu WASD: giữ khoảng cách ưa thích + strafe ngẫu nhiên."""
        self.move_timer -= dt
        if self.move_timer <= 0:
            pref = PREFERRED_DIST[self.weapon_name]
            self.forward = 1 if dist > pref + 1 else (-1 if dist < pref - 1 else 0)
            if self.rng.random() < 0.3:
                self.forward = int(self.rng.integers(-1, 2))
            self.strafe = int(self.rng.integers(-1, 2))
            self.move_timer = self.rng.uniform(0.4, 1.2)
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        mx = ca * self.forward - sa * self.strafe          # bên phải hướng nhìn = (-sin, cos)
        my = sa * self.forward + ca * self.strafe
        n = math.hypot(mx, my)
        if n > 0:
            sp = PLAYER["move_speed"] * dt / n
            self.x, self.y = move_with_collision(self.grid, self.x, self.y, mx * sp, my * sp, PLAYER["radius"])

    def _shoot(self, dt, sees, dist, sx, sy, zx, zy, zr):
        """Bắn: nạp đạn tự động khi hết băng. Trả về damage gây cho zombie."""
        self.cooldown -= dt
        if self.reload_t > 0:
            self.reload_t -= dt
            if self.reload_t <= 0:
                self.ammo = self.weapon["mag"]
            return 0.0
        if self.ammo <= 0:
            self.reload_t = self.weapon["reload"]
            return 0.0
        if sees and self.cooldown <= 0 and dist <= self.weapon["range"]:
            tolerance = math.asin(min(1.0, zr / max(dist, 1e-6))) + 0.05
            if abs(angle_diff(math.atan2(sy - self.y, sx - self.x), self.angle)) < tolerance:
                self.ammo -= 1
                self.cooldown = self.weapon["cooldown"]
                hit, _, _ = resolve_shot(self.grid, self.x, self.y, self.angle, self.weapon,
                                         [(zx, zy, zr)], self.rng)   # trúng hay không tính theo vị trí THẬT
                if hit is not None:
                    return float(self.weapon["damage"])
        return 0.0
