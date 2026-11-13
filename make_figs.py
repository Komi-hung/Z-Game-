"""Hình bổ sung cho báo cáo: vòng lặp RL, bản đồ 5 màn, đường học PPO, ảnh chụp game."""
import os, sys, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from PIL import Image
sys.path.insert(0, "/mnt/user-data/outputs/zombie-rl")
import config
OUT = os.path.dirname(os.path.abspath(__file__)); SH = os.path.join(OUT, "..", "figs"); ROOT = "/mnt/user-data/outputs/zombie-rl/models"
plt.rcParams.update({"mathtext.fontset": "dejavuserif", "font.family": "Liberation Serif", "font.size": 10, "figure.dpi": 170})
INK, RED, GREY, LINE = "#140a0b", "#9b1c2c", "#5a4a4c", "#d9cdce"

# ---- 1. vòng lặp RL
fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
def box(x, y, w, h, title, sub, ec, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", ec=ec, fc=fc, lw=1.6))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.6, color=GREY)
box(0.4, 3.35, 4.4, 1.3, "AGENT (zombie / boss)", "policy $\\pi_\\theta(a\\,|\\,s)$: MLP 64×64", RED, "#f7f2f2")
box(0.4, 0.35, 4.4, 1.3, "ENVIRONMENT", "game 5 màn · bước quyết định 0,1 giây", INK)
def arr(p0, p1, txt, tx, ty, col=INK, ha="left"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, lw=1.6, color=col))
    ax.text(tx, ty, txt, fontsize=9, color=col, ha=ha, va="center")
arr((1.2, 1.7), (1.2, 3.3), "observation $s_t$\n(29 số)", 1.05, 2.5, ha="right")
arr((2.6, 1.7), (2.6, 3.3), "reward $r_t$", 2.75, 2.5, RED)
arr((4.0, 3.3), (4.0, 1.7), "action $a_t$\n(18 lựa chọn)", 4.15, 2.5)
ax.add_patch(FancyArrowPatch((2.6, 4.7), (2.6, 4.7), arrowstyle="-"))
# bên phải: cập nhật policy
ax.add_patch(FancyBboxPatch((6.1, 2.35), 4.7, 2.3, boxstyle="round,pad=0.02,rounding_size=0.12", ec=LINE, fc="white", lw=1.2))
ax.text(8.45, 4.3, "Cập nhật policy", ha="center", fontsize=11, fontweight="bold", color=INK)
for i, t in enumerate(["PPO: clip tỉ lệ xác suất ε = 0,2", "TRPO: ràng buộc KL ≤ 0,01", "A2C: policy gradient, không giới hạn bước", "DQN: học Q(s,a) từ replay buffer"]):
    ax.text(6.3, 3.8 - i * 0.42, "• " + t, fontsize=8.8, color=INK, va="center")
ax.add_patch(FancyArrowPatch((4.85, 4.0), (6.05, 4.0), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=GREY))
ax.text(5.45, 4.25, "(s, a, r)", fontsize=8.2, color=GREY, ha="center")
ax.add_patch(FancyArrowPatch((6.05, 2.7), (4.85, 3.55), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=RED, connectionstyle="arc3,rad=0.2"))
ax.text(5.72, 3.3, "θ mới", fontsize=8.8, color=RED, ha="center")
ax.text(8.45, 1.2, "Mục tiêu: max  $J(\\theta) = E[\\sum_t \\gamma^t r_t]$,  γ = 0,99", ha="center", fontsize=9.4, color=INK)
ax.text(8.45, 0.7, "lặp lại hàng triệu bước trên 5 màn", ha="center", fontsize=8.6, color=GREY, style="italic")
fig.savefig(f"{OUT}/fig_rl_loop.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- 2. bản đồ 5 màn
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
for i, lv in enumerate(config.LEVELS):
    ax = axs.flat[i]; m = lv["map"]; H, W = len(m), len(m[0]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.axis("off")
    for y, row in enumerate(m):
        for x, ch in enumerate(row):
            if ch == "#": ax.add_patch(Rectangle((x, y), 1, 1, fc="#3a2e30", ec="#3a2e30", lw=0.4))
            else: ax.add_patch(Rectangle((x, y), 1, 1, fc="#f7f2f2", ec="#e6dfe0", lw=0.3))
    for y, row in enumerate(m):
        for x, ch in enumerate(row):
            if ch == "P": ax.add_patch(Circle((x + .5, y + .5), .38, fc="#2f6fb0", ec="none")); ax.text(x + .5, y + .52, "P", color="white", fontsize=6, ha="center", va="center", fontweight="bold")
            if ch == "Z": ax.add_patch(Circle((x + .5, y + .5), .38, fc=RED, ec="none")); ax.text(x + .5, y + .52, "Z", color="white", fontsize=6, ha="center", va="center", fontweight="bold")
            if ch == "B": ax.add_patch(Circle((x + .5, y + .5), .6, fc="#4a0d16", ec="none")); ax.text(x + .5, y + .52, "B", color="white", fontsize=7, ha="center", va="center", fontweight="bold")
    ax.set_title(lv["name"], fontsize=9, color=INK, pad=3)
ax = axs.flat[5]; ax.axis("off")
for j, (c, t) in enumerate([("#3a2e30", "tường"), ("#2f6fb0", "P: người chơi xuất phát"), (RED, "Z: điểm sinh zombie"), ("#4a0d16", "B: boss")]):
    ax.add_patch(Rectangle((0.05, 0.75 - j * 0.2), 0.08, 0.1, fc=c, transform=ax.transAxes)); ax.text(0.2, 0.8 - j * 0.2, t, transform=ax.transAxes, fontsize=9, va="center", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_maps.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- 3. đường học PPO (train từ đầu một màn)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False)
for ax, lv in zip(axs, [1, 2, 4]):
    for s, c in zip([0, 1, 2], [RED, "#c9707b", "#e3b0b6"]):
        f = f"{ROOT}/learning_curve_lv{lv}_seed{s}.csv"
        if not os.path.exists(f): continue
        d = pd.read_csv(f); ax.plot(d.steps / 1000, d.eval_reward, color=c, lw=1.5, marker="o", ms=2.5, label=f"PPO seed {s}")
    d = pd.read_csv(f"{ROOT}/learning_curve_lv{lv}_seed0.csv"); ax.axhline(d.path_reward.iloc[0], color=GREY, ls="--", lw=1, label="path")
    if "chase_reward" in d: ax.axhline(d.chase_reward.iloc[0], color="#b9aeb0", ls=":", lw=1, label="chase")
    ax.set_title(f"Màn {lv}", fontsize=9.5); ax.set_xlabel("bước huấn luyện (nghìn)", fontsize=8.5); ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
axs[0].set_ylabel("return đánh giá (100 episode)", fontsize=8.5); axs[0].legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_curve.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---- 4. ảnh chụp game
def S(n): return Image.open(f"{SH}/{n}.png")
lv3, ai = S("lv3"), S("ai_panel")
ai = ai.resize((int(ai.width * lv3.height / ai.height * 0.78), int(lv3.height * 0.78)), Image.NEAREST)
W = lv3.width + ai.width + 12; c = Image.new("RGB", (W, lv3.height), "white"); c.paste(lv3, (0, 0)); c.paste(ai, (lv3.width + 12, (lv3.height - ai.height) // 2)); c.save(f"{OUT}/fig_ai_view.png")
b1, b2 = S("boss"), S("boss2"); c = Image.new("RGB", (b1.width * 2 + 12, b1.height), "white"); c.paste(b1, (0, 0)); c.paste(b2, (b1.width + 12, 0)); c.save(f"{OUT}/fig_boss_shots.png")
S("montage5").save(f"{OUT}/fig_levels.png")
print("ok")
