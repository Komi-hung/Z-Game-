"""Đọc sprite sheet của 4 loại zombie và cắt từng frame.

Cấu trúc thư mục:
  assets/zombie-heavy/  walk.png attack.png death.png   (zombie mập)
  assets/zombie-vest/   ...                             (zombie vừa)
  assets/zombie-skinny/ ...                             (zombie ốm)
  assets/boss/          ...
Mỗi sheet là lưới đều: walk 4 cột x 2 hàng, attack và death 3 cột x 2 hàng.
Thiếu ảnh thì game tự vẽ zombie bằng hình khối để vẫn chạy được.
"""
import os
import numpy as np
import pygame
from PIL import Image

ASSET_DIR = "assets"
TYPE_FOLDERS = {"heavy": "zombie-heavy", "vest": "zombie-vest", "skinny": "zombie-skinny", "boss": "boss"}
LAYOUT = {"walk": (4, 2), "attack": (3, 2), "death": (3, 2)}   # (cột, hàng)
TARGET_H = 64        # chiều cao frame "đứng" sau khi thu nhỏ (px), hợp với độ phân giải 320x200
ALPHA_MIN = 40       # bỏ các pixel gần trong suốt (quầng sáng, nhiễu)


def _bbox(alpha):
    rows, cols = np.flatnonzero(alpha.any(axis=1)), np.flatnonzero(alpha.any(axis=0))
    if len(rows) == 0:
        return None
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def _remove_specks(alpha_mask, min_pixels=40):
    """Xóa các mảnh pixel tách rời: mảnh quá nhỏ (nhiễu) và mảnh chạm mép ô
    (thường là chân/tay của frame ô bên cạnh lấn sang). Luôn giữ mảnh lớn nhất."""
    h, w = alpha_mask.shape
    seen = np.zeros_like(alpha_mask)
    comps = []
    for sy, sx in zip(*np.nonzero(alpha_mask)):
        if seen[sy, sx]:
            continue
        stack, comp = [(sy, sx)], []
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and alpha_mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        comps.append(comp)
    keep = np.zeros_like(alpha_mask)
    if not comps:
        return keep
    largest = max(comps, key=len)
    for comp in comps:
        ys, xs = map(list, zip(*comp))
        touches_edge = min(ys) == 0 or min(xs) == 0 or max(ys) == h - 1 or max(xs) == w - 1
        if comp is largest or (len(comp) >= min_pixels and not touches_edge):
            keep[ys, xs] = True
    return keep


def load_sheet(path, cols, rows):
    """Cắt sheet theo lưới rồi canh mọi frame theo CHÂN.

    Mỗi frame được đặt sao cho tâm bàn chân nằm giữa và đáy chân chạm đáy khung,
    nên nhân vật không bị rung khi đổi frame dù các ô trong sheet vẽ lệch nhau.
    Trả về (list ảnh RGBA numpy cùng kích thước, chiều cao thân của frame đầu).
    """
    img = Image.open(path).convert("RGBA")
    if img.height // rows > 200:                 # sheet độ phân giải cao: thu nhỏ trước cho nhanh
        k = 160 / (img.height // rows)
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    rgba = np.array(img)
    h, w = rgba.shape[:2]
    ch, cw = h // rows, w // cols
    items = []
    for r in range(rows):
        for c in range(cols):
            cell = rgba[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw].copy()
            mask = _remove_specks(cell[..., 3] >= ALPHA_MIN)
            if not mask.any():
                continue
            cell[~mask] = 0
            top, bottom, left, right = _bbox(mask)
            feet = mask[bottom - max(2, (bottom - top) // 5):bottom]      # 20% dưới cùng = chân
            fx = np.flatnonzero(feet.any(axis=0)).mean()
            items.append((cell, top, bottom, left, right, fx))
    half_w = int(np.ceil(max(max(fx - l, r - fx) for _, _, _, l, r, fx in items))) + 1
    height = max(b - t for _, t, b, _, _, _ in items)
    frames = []
    for cell, top, bottom, left, right, fx in items:
        out = np.zeros((height, 2 * half_w, 4), dtype=np.uint8)
        x0 = int(round(half_w - (fx - left)))
        out[height - (bottom - top):, x0:x0 + (right - left)] = cell[top:bottom, left:right]
        frames.append(out)
    body_h = items[0][2] - items[0][1]          # frame 0 là tư thế đứng
    return frames, body_h


def _to_surface(arr):
    h, w = arr.shape[:2]
    return pygame.image.frombuffer(arr.tobytes(), (w, h), "RGBA").copy()


def _fallback(kind):
    """Zombie vẽ bằng hình khối khi thiếu ảnh."""
    frames = []
    for i in range({"walk": 4, "attack": 3, "death": 4}[kind]):
        s = pygame.Surface((32, 64), pygame.SRCALPHA)
        if kind == "death":
            hh = 64 - i * 14
            pygame.draw.rect(s, (60, 150, 60), (4, 64 - hh, 24, hh))
        else:
            pygame.draw.circle(s, (70, 170, 70), (16, 12), 9)
            pygame.draw.rect(s, (40, 110, 50), (7, 21, 18, 22))
            arm_y = 22 if kind == "walk" else 14 - i * 3
            pygame.draw.rect(s, (70, 170, 70), (1, arm_y, 6, 14))
            pygame.draw.rect(s, (70, 170, 70), (25, arm_y, 6, 14))
            off = 3 if (kind == "walk" and i % 2) else 0
            pygame.draw.rect(s, (60, 60, 110), (8 + off, 43, 7, 21))
            pygame.draw.rect(s, (60, 60, 110), (17 - off, 43, 7, 21))
        frames.append(s)
    return frames


class TypeSprites:
    """3 animation của một loại zombie, đã thu nhỏ cùng tỉ lệ."""

    def __init__(self, folder):
        self.anims = {}
        raw = {}
        for kind, (cols, rows) in LAYOUT.items():
            path = os.path.join(ASSET_DIR, folder, f"{kind}.png")
            raw[kind] = load_sheet(path, cols, rows) if os.path.exists(path) else None
        self.from_files = all(v is not None for v in raw.values())
        if not self.from_files:
            for kind in LAYOUT:
                self.anims[kind] = _fallback(kind)
            self.body_h, self.aspect = 64, 0.5
            return
        # Mỗi animation được scale sao cho thân ở frame đầu cao đúng TARGET_H px
        # (các sheet có thể khác độ phân giải, ví dụ walk 1536px nhưng attack 384px).
        for kind, (frames, body_h) in raw.items():
            k = TARGET_H / body_h
            self.anims[kind] = [pygame.transform.smoothscale(
                _to_surface(f), (max(1, round(f.shape[1] * k)), max(1, round(f.shape[0] * k))))
                for f in frames]
        b = _bbox(raw["walk"][0][0][..., 3] > 0)
        self.body_h = TARGET_H
        self.aspect = (b[3] - b[2]) / (b[1] - b[0])     # rộng / cao của thân khi đứng


class WeaponSprites:
    """Animation súng góc nhìn thứ nhất: assets/weapons/<súng>_fire.png và _reload.png.

    Mỗi file là 1 hàng 6 ô 128x128 (đã làm sạch bằng tools/prepare_weapons.py).
    fire: ô 0 = cầm súng bình thường, ô 1 = lửa đầu nòng, ô 2-5 = giật và hạ súng.
    reload: 6 bước thay băng đạn, ô cuối trùng tư thế cầm súng.
    """
    CELL = 128

    def __init__(self, guns):
        self.anims = {}
        for gun in guns:
            anims = {}
            for kind in ("fire", "reload"):
                path = os.path.join(ASSET_DIR, "weapons", f"{gun}_{kind}.png")
                if not os.path.exists(path):
                    break
                strip = pygame.image.load(path)
                if pygame.display.get_surface() is not None:
                    strip = strip.convert_alpha()
                n = strip.get_width() // self.CELL
                anims[kind] = [strip.subsurface((i * self.CELL, 0, self.CELL, self.CELL)).copy() for i in range(n)]
            if len(anims) == 2:
                self.anims[gun] = anims

    def has(self, gun):
        return gun in self.anims


class ZombieSprites:
    def __init__(self):
        self.types = {t: TypeSprites(f) for t, f in TYPE_FOLDERS.items()}
        if pygame.display.get_surface() is not None:
            for ts in self.types.values():
                for kind in ts.anims:
                    ts.anims[kind] = [f.convert_alpha() for f in ts.anims[kind]]
        self._cache = {}

    def frame(self, ztype, kind, index, tint=(255, 255, 255)):
        key = (ztype, kind, index, tint)
        if key not in self._cache:
            surf = self.types[ztype].anims[kind][index]
            if tint != (255, 255, 255):
                surf = surf.copy()
                surf.fill((*tint, 255), special_flags=pygame.BLEND_RGBA_MULT)
            self._cache[key] = surf
        return self._cache[key]

    def count(self, ztype, kind):
        return len(self.types[ztype].anims[kind])
