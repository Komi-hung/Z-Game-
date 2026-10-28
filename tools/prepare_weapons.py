"""Làm sạch sprite sheet súng rồi lưu vào assets/weapons/ để game dùng.

Chạy (từ thư mục project):  python tools/prepare_weapons.py
Ảnh gốc đặt trong tools/weapon_src/:
  pistol_fire.png  pistol_reload.png  m4a1_fire.png  m4a1_reload.png
Mỗi sheet là 1 hàng 6 frame. Frame 0 của sheet "fire" là tư thế cầm súng bình thường.

Các bước cải thiện:
  1. Xóa nền đen đặc (flood fill từ mép ảnh) -> nền trong suốt, kể cả nhiễu nén quanh súng.
  2. Bỏ các mảnh pixel lẻ tẻ (nhiễu).
  3. Đưa mọi sheet về cùng kích thước ô 128x128 (M4A1 gốc 362px, súng lục 128px).
  4. Khớp frame cuối của "reload" với frame đứng yên của "fire" (cùng cỡ, cùng vị trí)
     để lúc nạp đạn xong không bị giật hình.
  5. Viền tối 1px quanh hình cho nét pixel rõ trên nền tối của game.
"""
import os
from collections import deque

import numpy as np
from PIL import Image

SRC = os.path.join(os.path.dirname(__file__), "weapon_src")
DST = os.path.join(os.path.dirname(__file__), "..", "assets", "weapons")
CELL = 128
FRAMES = 6
BG_TOLERANCE = 14        # pixel nền: kênh màu lớn nhất <= giá trị này
ERODE = 3                # khe hẹp hơn ~2*ERODE px không cho nền lan qua
MIN_PIECE = 30           # mảnh nhỏ hơn số pixel này coi là nhiễu
OUTLINE = (12, 12, 16, 255)


def remove_background(rgba):
    """Pixel gần đen nối liền với mép ảnh -> trong suốt. Giữ nguyên ảnh đã có alpha."""
    if rgba[..., 3].min() < 255:                      # ảnh đã trong suốt sẵn
        rgba[rgba[..., 3] < 40] = 0
        rgba[..., 3] = np.where(rgba[..., 3] > 0, 255, 0)
        return rgba
    # Súng đen trên nền đen: nếu loang thẳng từ mép, vùng nền sẽ "chảy" qua viền mảnh
    # vào các mảng tối bên trong thân súng. Nên: co vùng nền lại (bịt các khe hẹp),
    # loang từ mép trên vùng đã co, rồi nở ra lại để ăn nốt viền nhiễu sát hình.
    bg_like = rgba[..., :3].max(axis=-1) <= BG_TOLERANCE
    core = _shrink(bg_like, ERODE)
    bg = _grow(_flood_from_edges(core), ERODE + 1) & bg_like
    rgba[bg] = 0
    return rgba


def _shift_all(mask, op, fill):
    out = mask.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sh = np.full_like(mask, fill)            # ngoài mép ảnh coi như nền
        ys = slice(max(dy, 0), mask.shape[0] + min(dy, 0))
        yd = slice(max(-dy, 0), mask.shape[0] + min(-dy, 0))
        xs = slice(max(dx, 0), mask.shape[1] + min(dx, 0))
        xd = slice(max(-dx, 0), mask.shape[1] + min(-dx, 0))
        sh[ys, xs] = mask[yd, xd]
        out = op(out, sh)
    return out


def _shrink(mask, r):
    for _ in range(r):
        mask = _shift_all(mask, np.logical_and, True)
    return mask


def _grow(mask, r):
    for _ in range(r):
        mask = _shift_all(mask, np.logical_or, False)
    return mask


def _flood_from_edges(mask):
    h, w = mask.shape
    out = np.zeros_like(mask)
    q = deque((y, x) for y in range(h) for x in (0, w - 1) if mask[y, x])
    q.extend((y, x) for x in range(w) for y in (0, h - 1) if mask[y, x])
    for y, x in q:
        out[y, x] = True
    while q:
        y, x = q.popleft()
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                q.append((ny, nx))
    return out


def remove_specks(rgba):
    mask = rgba[..., 3] > 0
    h, w = mask.shape
    seen = np.zeros_like(mask)
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        comp, stack = [], [(sy, sx)]
        seen[sy, sx] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True; stack.append((ny, nx))
        if len(comp) < MIN_PIECE:
            ys, xs = zip(*comp)
            rgba[list(ys), list(xs)] = 0
    return rgba


def to_cells(rgba):
    """Cắt 6 ô và thu về CELL x CELL (lấy trung bình vùng rồi làm sắc lại alpha)."""
    h, w = rgba.shape[:2]
    cw = w // FRAMES
    cells = []
    for i in range(FRAMES):
        cell = Image.fromarray(rgba[:, i * cw:(i + 1) * cw])
        if cell.size != (CELL, CELL):
            # premultiply để viền không bị loang màu nền khi thu nhỏ
            arr = np.array(cell).astype(np.float32)
            arr[..., :3] *= arr[..., 3:4] / 255.0
            small = np.array(Image.fromarray(arr.astype(np.uint8)).resize((CELL, CELL), Image.BOX)).astype(np.float32)
            a = small[..., 3:4]
            small[..., :3] = np.where(a > 0, small[..., :3] * 255.0 / np.maximum(a, 1), 0)
            small[..., 3] = np.where(a[..., 0] >= 110, 255, 0)
            cell = Image.fromarray(np.clip(small, 0, 255).astype(np.uint8))
        cells.append(np.array(cell))
    return cells


def bbox(cell):
    ys, xs = np.nonzero(cell[..., 3] > 0)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def match_pose(cells, ref):
    """Scale + dịch cả sheet reload để frame cuối khớp frame đứng yên của sheet fire."""
    t0, b0, l0, r0 = bbox(ref)
    t1, b1, l1, r1 = bbox(cells[-1])
    k = (b0 - t0) / (b1 - t1)
    if abs(k - 1) < 0.03 and abs(b0 - b1) <= 2 and abs((l0 + r0) - (l1 + r1)) <= 4:
        return cells                                     # đã khớp sẵn
    out = []
    for c in cells:
        img = Image.fromarray(c)
        size = max(1, round(CELL * k))
        img = img.resize((size, size), Image.NEAREST)
        canvas = Image.new("RGBA", (CELL, CELL), (0, 0, 0, 0))
        # đặt sao cho đáy & tâm ngang của frame cuối trùng với frame tham chiếu
        dx = round((l0 + r0) / 2 - (l1 + r1) / 2 * k)
        dy = CELL - size + (b0 - CELL)          # giữ tay chạm đúng mép dưới
        canvas.paste(img, (dx, dy), img)
        out.append(np.array(canvas))
    return out


def add_outline(cell):
    a = cell[..., 3] > 0
    grown = a.copy()
    grown[1:] |= a[:-1]; grown[:-1] |= a[1:]; grown[:, 1:] |= a[:, :-1]; grown[:, :-1] |= a[:, 1:]
    edge = grown & ~a
    edge[-1, :] = False                                  # không viền ở đáy (tay đi ra khỏi màn hình)
    cell[edge] = OUTLINE
    return cell


def process(name):
    rgba = np.array(Image.open(os.path.join(SRC, f"{name}.png")).convert("RGBA"))
    rgba = remove_specks(remove_background(rgba))
    return to_cells(rgba)


def main():
    os.makedirs(DST, exist_ok=True)
    for gun in ("pistol", "m4a1"):
        fire = process(f"{gun}_fire")
        reload_ = match_pose(process(f"{gun}_reload"), fire[0])
        for kind, cells in (("fire", fire), ("reload", reload_)):
            cells = [add_outline(c.copy()) for c in cells]
            strip = np.concatenate(cells, axis=1)
            Image.fromarray(strip).save(os.path.join(DST, f"{gun}_{kind}.png"))
            print("đã lưu", os.path.join(DST, f"{gun}_{kind}.png"))


if __name__ == "__main__":
    main()
