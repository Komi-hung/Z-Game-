"""Hình học dùng CHUNG cho env (train) và game (chơi).

Mọi phép tính va chạm, raycast, đường ngắm, trúng đạn đều nằm ở đây,
để zombie học trên đúng "thế giới" mà người chơi nhìn thấy.
Tọa độ: x = cột, y = hàng (y tăng xuống dưới). Góc 0 = hướng +x.
"""
import math
from collections import deque


def parse_map(rows):
    """Chuỗi bản đồ -> (grid, spawns, free_cells).

    grid[y][x] = True nếu là tường.
    spawns: {'P': [(x, y)], 'Z': [...], ...} tọa độ tâm ô.
    free_cells: các ô sàn đi tới được từ vị trí P (dùng để sinh vị trí ngẫu nhiên khi train).
    """
    width = len(rows[0])
    assert all(len(r) == width for r in rows), "Mọi hàng bản đồ phải dài bằng nhau"
    grid = [[c == "#" for c in row] for row in rows]
    spawns = {}
    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            if c not in "#.":
                spawns.setdefault(c, []).append((x + 0.5, y + 0.5))
    assert "P" in spawns, "Bản đồ cần có ký hiệu P"

    # flood fill từ P để chỉ lấy vùng đi tới được
    sx, sy = int(spawns["P"][0][0]), int(spawns["P"][0][1])
    seen, queue = {(sx, sy)}, deque([(sx, sy)])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (nx, ny) not in seen and not grid[ny][nx]:
                seen.add((nx, ny))
                queue.append((nx, ny))
    return grid, spawns, sorted(seen)


def is_wall(grid, x, y):
    ix, iy = int(math.floor(x)), int(math.floor(y))
    if iy < 0 or ix < 0 or iy >= len(grid) or ix >= len(grid[0]):
        return True
    return grid[iy][ix]


def angle_diff(a, b):
    """a - b, chuẩn hóa về [-pi, pi]."""
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def cast_ray(grid, ox, oy, angle, max_dist):
    """Raycast DDA. Trả về (khoảng cách, side, vị trí trúng trên mặt tường 0..1).

    side = 0: trúng mặt tường dọc, 1: mặt tường ngang (dùng để tô đậm nhạt).
    Nếu không trúng tường trong max_dist thì trả về max_dist.
    """
    dx, dy = math.cos(angle), math.sin(angle)
    mx, my = int(math.floor(ox)), int(math.floor(oy))
    ddx = abs(1.0 / dx) if dx != 0 else 1e30
    ddy = abs(1.0 / dy) if dy != 0 else 1e30
    if dx < 0:
        step_x, side_x = -1, (ox - mx) * ddx
    else:
        step_x, side_x = 1, (mx + 1.0 - ox) * ddx
    if dy < 0:
        step_y, side_y = -1, (oy - my) * ddy
    else:
        step_y, side_y = 1, (my + 1.0 - oy) * ddy
    h, w = len(grid), len(grid[0])
    while True:
        if side_x < side_y:
            dist, side_x, mx, side = side_x, side_x + ddx, mx + step_x, 0
        else:
            dist, side_y, my, side = side_y, side_y + ddy, my + step_y, 1
        if dist > max_dist:
            return max_dist, side, 0.0
        if mx < 0 or my < 0 or mx >= w or my >= h or grid[my][mx]:
            hit = oy + dist * dy if side == 0 else ox + dist * dx
            return dist, side, hit - math.floor(hit)


def _blocked(grid, x, y, r):
    return (is_wall(grid, x - r, y - r) or is_wall(grid, x + r, y - r)
            or is_wall(grid, x - r, y + r) or is_wall(grid, x + r, y + r))


def move_with_collision(grid, x, y, dx, dy, radius):
    """Di chuyển từng trục riêng -> khi đi chéo vào tường sẽ trượt dọc tường."""
    if not _blocked(grid, x + dx, y, radius):
        x += dx
    if not _blocked(grid, x, y + dy, radius):
        y += dy
    return x, y


def line_of_sight(grid, ax, ay, bx, by):
    d = math.hypot(bx - ax, by - ay)
    if d < 1e-6:
        return True
    return cast_ray(grid, ax, ay, math.atan2(by - ay, bx - ax), d)[0] >= d - 1e-6


def clear_path(grid, ax, ay, bx, by, radius):
    """Như line_of_sight nhưng cho cả THÂN có bán kính radius: kiểm tra tâm và hai mép."""
    d = math.hypot(bx - ax, by - ay)
    if d < 1e-6:
        return True
    nx, ny = -(by - ay) / d * radius, (bx - ax) / d * radius
    return all(line_of_sight(grid, ax + ox, ay + oy, bx + ox, by + oy)
               for ox, oy in ((0, 0), (nx, ny), (-nx, -ny)))


def in_fire_line(grid, px, py, p_angle, zx, zy, z_radius, weapon):
    """True nếu một phát bắn lúc này CÓ THỂ trúng zombie (trong tầm, trong góc, không bị tường che).

    Cùng logic với resolve_shot(): góc lệch tối đa = góc che của thân zombie + spread của súng.
    Đây là tín hiệu "bị ngắm" mà reward dùng để dạy zombie né.
    """
    d = math.hypot(zx - px, zy - py)
    if d > weapon["range"] + z_radius:
        return False
    if d < 1e-6:
        return True
    off = abs(angle_diff(math.atan2(zy - py, zx - px), p_angle))
    if off > math.asin(min(1.0, z_radius / d)) + weapon["spread"]:
        return False
    return line_of_sight(grid, px, py, zx, zy)


class PathFinder:
    """Tìm đường ngắn nhất trên lưới (Dijkstra 8 hướng, không cắt góc tường).

    Bảng khoảng cách tới mỗi ô đích được tính một lần rồi lưu lại, nên rất rẻ khi gọi nhiều.
    Dùng để: (1) cho zombie biết hướng đi vòng qua tường, (2) tính reward theo đường đi thật.
    """
    SQ2 = math.sqrt(2)

    def __init__(self, grid):
        self.grid = grid
        self.h, self.w = len(grid), len(grid[0])
        self._fields = {}

    def _neighbors(self, x, y):
        g = self.grid
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if not g[y + dy][x + dx]:
                yield x + dx, y + dy, 1.0
        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            if not g[y + dy][x + dx] and not g[y][x + dx] and not g[y + dy][x]:
                yield x + dx, y + dy, self.SQ2

    def field(self, tx, ty):
        """Khoảng cách đường đi từ mọi ô tới ô (tx, ty)."""
        key = (tx, ty)
        if key not in self._fields:
            import heapq
            dist = [[math.inf] * self.w for _ in range(self.h)]
            dist[ty][tx] = 0.0
            heap = [(0.0, tx, ty)]
            while heap:
                d, x, y = heapq.heappop(heap)
                if d > dist[y][x]:
                    continue
                for nx, ny, c in self._neighbors(x, y):
                    if d + c < dist[ny][nx]:
                        dist[ny][nx] = d + c
                        heapq.heappush(heap, (d + c, nx, ny))
            self._fields[key] = dist
        return self._fields[key]

    def step(self, zx, zy, px, py, radius=0.0):
        """Trả về (ux, uy, quãng đường): hướng nên đi tiếp và độ dài đường đi tới người chơi."""
        direct = math.hypot(px - zx, py - zy)
        if direct < 1e-6:
            return 0.0, 0.0, 0.0
        if clear_path(self.grid, zx, zy, px, py, radius):       # cả thân đi lọt: đi thẳng
            return (px - zx) / direct, (py - zy) / direct, direct
        cx, cy = int(zx), int(zy)
        f = self.field(int(px), int(py))
        best = None
        for nx, ny, _ in self._neighbors(cx, cy):
            if best is None or f[ny][nx] < f[best[1]][best[0]]:
                best = (nx, ny)
        if best is None or f[best[1]][best[0]] == math.inf:     # (không xảy ra trên bản đồ liên thông)
            return (px - zx) / direct, (py - zy) / direct, direct
        tx, ty = best[0] + 0.5, best[1] + 0.5
        d = math.hypot(tx - zx, ty - zy)
        if d < 1e-6:
            return 0.0, 0.0, f[best[1]][best[0]]
        return (tx - zx) / d, (ty - zy) / d, d + f[best[1]][best[0]]


def resolve_shot(grid, px, py, p_angle, weapon, targets, rng):
    """Bắn một viên đạn. targets = [(x, y, radius), ...].

    Trả về (index mục tiêu trúng hoặc None, góc viên đạn, độ dài đường đạn).
    rng chỉ cần có hàm uniform(a, b) (random.Random hoặc numpy Generator đều được).
    """
    shot = p_angle + (rng.uniform(-weapon["spread"], weapon["spread"]) if weapon["spread"] > 0 else 0.0)
    wall = cast_ray(grid, px, py, shot, weapon["range"])[0]
    ux, uy = math.cos(shot), math.sin(shot)
    best, best_t = None, wall
    for i, (tx, ty, tr) in enumerate(targets):
        vx, vy = tx - px, ty - py
        t = vx * ux + vy * uy                 # khoảng cách dọc theo đường đạn
        if 0 < t < best_t and abs(vx * uy - vy * ux) <= tr:
            best, best_t = i, t
    return best, shot, best_t
