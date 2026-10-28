"""Toàn bộ số liệu thiết kế game nằm ở đây. Muốn cân bằng game thì chỉ sửa file này.

Đơn vị thống nhất (dùng chung cho env lúc train và game lúc chơi):
  - khoảng cách: ô (cell) của bản đồ
  - tốc độ: ô / giây, tốc độ quay: radian / giây
  - thời gian: giây
Sửa ZOMBIES / WEAPONS / REWARD / bản đồ thì nên train lại (python train.py).
"""
import math

# ---------------- Màn hình ----------------
RENDER_W, RENDER_H = 320, 200      # độ phân giải "pixel" bên trong
WINDOW_SCALE = 3                   # cửa sổ = 960 x 600
FOV = math.radians(66)
FPS = 60
MAX_VIEW = 24.0
MOUSE_SENSITIVITY = 0.003          # radian cho mỗi pixel chuột di chuyển ngang

# ---------------- RL: nhịp ra quyết định ----------------
# Zombie chọn hành động mỗi DECISION_DT giây, cả lúc train lẫn lúc chơi.
DECISION_DT = 0.1

# ---------------- Người chơi ----------------
# bot_turn_speed: tốc độ quay của người chơi giả lúc train. Người thật dùng chuột quay rất nhanh,
# nên bot cũng quay nhanh hơn phím mũi tên để zombie quen với đối thủ dùng chuột.
PLAYER = dict(hp=100, move_speed=2.2, turn_speed=2.5, bot_turn_speed=5.0, radius=0.25)

# ---------------- Zombie ----------------
# Tốc độ: mập 1.2, vừa 2.4, ốm 4.3 (nhanh nhất). Súng lục 25 damage -> 2 viên hạ con ốm (50 máu).
# radius nhỏ lại để không kẹt góc tường; hình vẽ được co theo radius nên không "lẹm" vào tường.
ZOMBIES = {
    "heavy":  dict(name="Zombie mập", hp=200, damage=25, speed=1.2, radius=0.30, attack_range=0.75,
                   attack_cooldown=1.4, windup=0.55, lunge=0.3),
    "vest":   dict(name="Zombie vừa", hp=100, damage=10, speed=2.4, radius=0.22, attack_range=0.65,
                   attack_cooldown=1.0, windup=0.40, lunge=0.4),
    "skinny": dict(name="Zombie ốm",  hp=50,  damage=10, speed=4.3, radius=0.18, attack_range=0.60,
                   attack_cooldown=0.8, windup=0.30, lunge=0.4),
    "boss":   dict(name="Boss",       hp=1500, damage=35, speed=3.0, radius=0.35, attack_range=0.90,
                   attack_cooldown=0.9, windup=0.40, lunge=0.5),
}
# lunge: khi đang vung tay, zombie vẫn nhích theo người chơi với (lunge x tốc độ) -> khó né hơn
SPRITE_WIDTH_FACTOR = 1.15   # bề ngang hình vẽ = 2 x radius x hệ số này

# Boss = não PPO + các pha viết tay
BOSS = dict(phase2_hp=0.5, phase2_speed_mult=1.3,
            charge_min_dist=4.0, charge_time=1.0, charge_cooldown=4.0, charge_speed_mult=2.0)

# Gỡ kẹt (lưới an toàn trong game): nếu zombie gần như đứng im quá lâu thì đi theo đường tìm được
STUCK = dict(check_time=1.0, min_move_frac=0.2, unstuck_time=0.8)

# ---------------- Súng ----------------
# Chỉ còn 2 súng. unlock = màn bắt đầu có súng này: M4A1 nhận sau khi qua màn 2.
WEAPONS = {
    "pistol": dict(name="Súng lục", damage=25, cooldown=0.28, mag=12, reload=1.2, range=10, spread=0.02, unlock=1),
    "m4a1":   dict(name="M4A1",     damage=22, cooldown=0.12, mag=35, reload=1.6, range=12, spread=0.03, unlock=3),
}

# ---------------- Reward (trọng tài) ----------------
REWARD = dict(
    approach=0.3,     # mỗi ô rút ngắn ĐƯỜNG ĐI tới người chơi (không phải đường chim bay)
    step=-0.001,
    aimed=-0.01,      # mỗi bước đứng trong đường ngắm
    hit_taken=-0.5,   # nhân với damage / máu tối đa
    death=-1.0,
    catch=1.0,        # zombie thường: đánh trúng người chơi lần đầu (episode kết thúc tại đây)
    hit_dealt=0.5,    # mỗi đòn đánh trúng người chơi
    whiff=-0.05,      # vung tay trượt (đánh bừa thì bị phạt nhẹ -> học chọn thời điểm)
    kill_player=2.0,  # boss: hạ được người chơi
)
# Boss được train bằng trận tay đôi đầy đủ: người chơi giả có máu, boss vung tay - hồi chiêu như trong game,
# trận chỉ kết thúc khi một bên chết. Zombie thường kết thúc episode khi áp sát được người chơi.
DUEL_TYPES = {"boss"}
MAX_EPISODE_STEPS = 400
DUEL_MAX_STEPS = 600  # trận tay đôi với boss dài tối đa 60 giây

# ---------------- 5 màn chơi ----------------
# Bản đồ: # tường, . sàn, P người chơi, Z điểm xuất hiện zombie, B boss
# waves: các đợt zombie. Đợt sau xuất hiện khi còn <= WAVE_NEXT_ALIVE con hoặc sau WAVE_MAX_GAP giây.
# bot: độ giỏi người chơi giả khi train màn đó (kite = tỉ lệ episode bot biết thả diều quanh cột). train_mix: tỉ lệ loại zombie khi train.
WAVE_NEXT_ALIVE = 1
WAVE_MAX_GAP = 30.0
HEAL_PER_WAVE = 20          # hồi máu khi đợt mới bắt đầu (0 = tắt)

LEVELS = [
    dict(name="Màn 1 - Kho chứa", train_steps=250_000,
         bot=dict(turn=0.45, reaction=5, aim_noise=0.20, kite=0.0),
         train_mix={"vest": 0.8, "heavy": 0.2},
         waves=[{"vest": 3}, {"vest": 4}, {"vest": 3, "heavy": 1}],
         map=[
             "################",
             "#P.....#.......#",
             "#......#.......#",
             "#..##..#..Z....#",
             "#..##..........#",
             "#..............#",
             "#......#..##...#",
             "####.###..##...#",
             "#..............#",
             "#..Z.....#..Z..#",
             "#........#.....#",
             "################",
         ]),
    dict(name="Màn 2 - Hành lang", train_steps=250_000,
         bot=dict(turn=0.60, reaction=4, aim_noise=0.14, kite=0.25),
         train_mix={"vest": 0.5, "skinny": 0.3, "heavy": 0.2},
         waves=[{"vest": 3, "skinny": 1}, {"vest": 3, "skinny": 2}, {"vest": 3, "heavy": 1, "skinny": 2},
                {"skinny": 4}],
         map=[
             "##################",
             "#P...#...........#",
             "#....#..Z........#",
             "#....#.....##....#",
             "#..........##....#",
             "#....#...........#",
             "###.####..####.###",
             "#........#.......#",
             "#..Z.....#...Z...#",
             "#....##..........#",
             "#....##..#....Z..#",
             "#........#.......#",
             "#...Z....#.......#",
             "##################",
         ]),
    dict(name="Màn 3 - Tầng hầm", train_steps=300_000,
         bot=dict(turn=0.75, reaction=3, aim_noise=0.09, kite=0.5),
         train_mix={"vest": 0.34, "skinny": 0.33, "heavy": 0.33},
         waves=[{"vest": 2, "heavy": 1, "skinny": 1}, {"vest": 3, "heavy": 2}, {"skinny": 3, "vest": 2},
                {"heavy": 2, "skinny": 2, "vest": 2}],
         map=[
             "####################",
             "#P.......#.........#",
             "#........#....Z....#",
             "#..##....#.........#",
             "#..##.......##..Z..#",
             "#...........##.....#",
             "#....#.............#",
             "######..####..######",
             "#.........#........#",
             "#..Z......#....Z...#",
             "#.....##.......##..#",
             "#.....##...........#",
             "#..........#.......#",
             "#..Z.......#...Z...#",
             "#..........#.......#",
             "####################",
         ]),
    dict(name="Màn 4 - Phòng thí nghiệm", train_steps=300_000,
         bot=dict(turn=0.90, reaction=2, aim_noise=0.05, kite=0.75),
         train_mix={"vest": 0.34, "skinny": 0.33, "heavy": 0.33},
         waves=[{"vest": 3, "skinny": 2}, {"heavy": 3, "vest": 2}, {"skinny": 4, "vest": 2},
                {"heavy": 2, "skinny": 3, "vest": 3}, {"heavy": 3, "skinny": 3, "vest": 3}],
         map=[
             "####################",
             "#P....#............#",
             "#.....#...Z....Z...#",
             "#.....#............#",
             "#...........##.....#",
             "#..##.......##..Z..#",
             "#..##..............#",
             "#.......#####......#",
             "###..####...##..####",
             "#........#.........#",
             "#..Z.....#....Z....#",
             "#....##.........##.#",
             "#....##..Z......##.#",
             "#..................#",
             "#.......####.......#",
             "#..Z....#..#....Z..#",
             "#..................#",
             "####################",
         ]),
    # Màn 5: người chơi đấu tay đôi với boss. Boss được train lâu nhất, tiếp nối từ model màn 4
    # và ôn lại bản đồ của cả 4 màn trước.
    dict(name="Màn 5 - Boss", train_steps=800_000,
         bot=dict(turn=1.00, reaction=1, aim_noise=0.03, kite=1.0),
         train_mix={"boss": 1.0},
         waves=[{"boss": 1}],
         map=[
             "##################",
             "#P...............#",
             "#................#",
             "#..##........##..#",
             "#..##........##..#",
             "#................#",
             "#................#",
             "#.......##.......#",
             "#.......##...B...#",
             "#................#",
             "#................#",
             "#..##........##..#",
             "#..##........##..#",
             "#................#",
             "#................#",
             "#................#",
             "#................#",
             "##################",
         ]),
]
REPLAY_PROB = 0.5   # khi train màn k, 50% episode dùng bản đồ các màn trước (chống quên)

MODEL_DIR = "models"
LOG_PLAYTEST = True
PLAYTEST_DIR = "playtest_logs"
