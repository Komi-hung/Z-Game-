# Zombie RL

Game bắn zombie góc nhìn thứ nhất (pixel, raycasting kiểu Wolfenstein). Zombie được huấn luyện bằng PPO và khôn dần qua 5 màn. Màn 5 là trận tay đôi với boss.

## Cấu trúc

| File | Vai trò | RL? |
|---|---|---|
| `config.py` | Mọi số liệu: zombie, súng, reward, 5 bản đồ, các đợt zombie, độ giỏi bot | Reward nằm ở đây |
| `geometry.py` | Raycast, va chạm, đường ngắm, trúng đạn, tìm đường (PathFinder). Dùng CHUNG cho train và game | |
| `combat.py` | Luật đánh của zombie (vung tay, ra đòn, hồi chiêu) và các pha của boss. Dùng CHUNG cho train và game | |
| `zombie_brain.py` | Observation (29 số) và 9 action của zombie. Dùng CHUNG cho train và game | **Có** |
| `player_bot.py` | Người chơi giả để zombie tập (biết thả diều quanh cột ở các màn sau) | **Có** |
| `env_zombie.py` | Môi trường Gymnasium: sân tập của zombie | **Có** |
| `train.py` | Train PPO theo curriculum màn 1 → 5, so sánh với 3 baseline | **Có** |
| `sprites.py` | Cắt sprite sheet zombie trong `assets/<loại zombie>/` và súng trong `assets/weapons/` | |
| `tools/prepare_weapons.py` | Làm sạch ảnh súng gốc trong `tools/weapon_src/` rồi lưu vào `assets/weapons/` | |
| `game_core.py` | **Toàn bộ luật chơi** (màn, đợt zombie, súng, zombie AI) - không vẽ. Game và agent người chơi cùng chạy trên file này | |
| `game.py` | Vẽ + nhận phím, chạy trên `game_core.py` | Chỉ dùng model |
| `player_env.py` | Môi trường cho RL agent **đóng vai người chơi** + người chơi viết tay để đối chứng | **Có** |
| `train_player.py` | Train agent người chơi (PPO) | **Có** |
| `evaluate_match.py` | Cho người chơi đấu N trận với zombie PPO và zombie viết tay, đếm ai thắng | Đánh giá |
| `zombie_world_env.py` | Zombie luyện **ngay trong game thật** với agent người chơi (đối kháng) | **Có** |
| `league.py` | Self-play theo vòng: zombie học -> giữ bản tốt nhất -> người chơi học lại -> đo | **Có** |
| `tools/learning_curve.py` | Huấn luyện lại từ đầu một màn với nhiều seed, ghi đường học (return theo số bước) và mốc baseline | Đánh giá |

## Cài đặt (Windows, VS Code)

```powershell
cd "D:\Zombie RL"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy

```powershell
python game.py               # chơi từ màn 1
python game.py --level 5     # vào thẳng màn boss
python game.py --zombies path    # chơi với zombie viết tay để tự so sánh với zombie RL
python game.py --watch-agent     # xem RL agent người chơi tự chơi
python train.py              # train lại cả 5 màn (số bước mỗi màn nằm trong config.py)
python train.py --level 5    # chỉ train lại boss
python train.py --scale 0.1  # train thử nhanh với 10% số bước
python train.py --finetune --scale 0.4   # chỉnh nhẹ config (tốc độ, damage): học tiếp từ model hiện có
```

## Điều khiển

Chuột quay trái/phải · chuột trái bắn · W/S tiến lùi · A/D đi ngang · ←/→ quay bằng phím · Space bắn bằng phím · R nạp đạn · 1 súng lục · 2 M4A1 (nhận sau khi qua màn 2) · Tab bảng RL · Enter sang màn / chơi lại màn · Esc thoát

Hạ boss ở màn 5 là chiến thắng: game hiện màn hình tổng kết rồi kết thúc.

## Súng

| Súng | Damage | Tốc độ bắn | Băng đạn | Nạp đạn | Có từ |
|---|---|---|---|---|---|
| Súng lục | 25 | 0.28 s/phát | 12 | 1.2 s | Màn 1 |
| M4A1 | 22 | 0.12 s/phát | 35 | 1.6 s | Màn 3 |

Muốn thay hình súng: đặt ảnh mới vào `tools/weapon_src/` (1 hàng 6 frame, nền trong suốt hoặc nền đen) rồi chạy `python tools/prepare_weapons.py`.

## Zombie

| Loại | Thư mục sprite | Máu | Damage | Tốc độ (ô/s) |
|---|---|---|---|---|
| Mập | `assets/zombie-heavy` | 200 | 25 | 1.2 |
| Vừa | `assets/zombie-vest` | 100 | 10 | 2.4 |
| Ốm | `assets/zombie-skinny` | 50 | 10 | 4.3 |
| Boss | `assets/boss` | 1500 | 35 | 3.0 (pha 2 nhanh hơn, có lao tới) |

## Phần RL làm gì

- **Action (18)**: 9 hướng đi x {cho phép đánh, giữ đòn}. Zombie tự quyết định có vung tay khi người chơi vào tầm hay chờ thời cơ (người chơi bị dồn vào tường, đang nạp đạn...). Khi vung tay, zombie vẫn nhích theo người chơi nên không thể né chỉ bằng cách lùi đúng nhịp.
- **Observation**: 8 tia khoảng cách tới tường, vị trí người chơi, **hướng đường đi vòng qua tường** (từ PathFinder), vận tốc và hướng nhìn của người chơi, cờ "đang bị ngắm", loại zombie, súng người chơi cầm, máu, **người chơi đang nạp đạn / còn bao nhiêu đạn**, đòn đánh của mình đã hồi chiêu chưa.
- **Boss train bằng trận tay đôi**: người chơi giả có máu, biết thả diều quanh cột; boss vung tay - hồi chiêu đúng luật trong game; trận chỉ kết thúc khi một bên chết. Boss được thưởng mỗi đòn trúng và khi hạ được người chơi.
- **Reward**: thưởng khi rút ngắn **quãng đường đi thật** (không phải đường chim bay, nên lao vào tường không được điểm), phạt khi bị ngắm, bị bắn, bị giết; thưởng lớn khi áp sát người chơi.
- **Curriculum**: màn sau học tiếp từ model màn trước và ôn lại bản đồ cũ. Người chơi giả mạnh dần qua các màn. Boss được train lâu nhất.

Bảng RL (Tab) trong game: vạch vàng là hướng PPO vừa chọn, vòng trắng là zombie đang bị ngắm, vòng cam là zombie đang **giữ đòn** (chờ thời cơ). Dòng "Gỡ kẹt" đếm số lần lưới an toàn phải can thiệp khi zombie đứng im quá lâu.

## Kết quả train

`models/eval.csv`: so sánh PPO với 3 baseline (random, chase = đi thẳng, path = đi theo đường ngắn nhất) ở từng màn: tỉ lệ bắt được, bị giết, hết giờ, % thời gian bị ngắm, % bước bị tường chặn, % máu mất. Đây là số liệu cho báo cáo.

`playtest_logs/`: ghi lại cách bạn di chuyển và bắn mỗi lần chơi.

## Đánh giá zombie RL bằng agent người chơi

```powershell
python train_player.py           # train agent người chơi (mặc định 3 triệu bước)
python evaluate_match.py         # 100 trận cho mỗi màn + 100 lượt chơi cả game
```

Logic đánh giá:

1. **Cùng luật**: agent người chơi chạy trên `game_core.py`, đúng luật người thật chơi (60 khung hình/giây, cùng súng, cùng đợt zombie). Nó chỉ dùng các nút người chơi có, tốc độ quay giới hạn 6 rad/s, quyết định mỗi 0.1 giây.
2. **So sánh, không chỉ đếm**: cùng một người chơi đấu với zombie PPO và 2 bản viết tay (`path`, `chase`). Zombie RL tốt khi nó thắng nhiều hơn bản viết tay.
3. **Không thiên vị**: agent người chơi được tập với cả 3 loại zombie, nên nó không chỉ giỏi khai thác riêng zombie PPO.
4. **So theo cặp**: trận thứ i với mọi loại zombie dùng cùng seed, nên khác biệt đến từ cách zombie chơi chứ không do may rủi. Kiểm định McNemar cho biết khác biệt có ý nghĩa thống kê không (p < 0.05).
5. **Khoảng tin cậy**: 100 trận chỉ chính xác khoảng ±10% (cột "95% CI"). Muốn chắc hơn thì tăng `--matches`.
6. **Hai chế độ**: `level` (mỗi màn một trận, 100 máu) và `campaign` (màn 1 → 5 với một mạng như game thật).

Kết quả từng trận lưu ở `models/match_eval.csv`.

## Self-play: làm zombie "khôn thật"

```powershell
python league.py              # chạy 1 vòng (khoảng 1,5 - 2 tiếng trên máy 2 nhân)
python league.py --rounds 3   # chạy nhiều vòng liên tiếp
```

Mỗi vòng:

1. **Zombie học trong game thật** (`zombie_world_env.py`) với tất cả các phiên bản agent người chơi đã có, cộng người chơi viết tay. Boss học nhiều nhất.
2. **Giữ bản tốt nhất**: bản zombie mới đấu với bản đang dùng trên cùng các trận (cùng seed, cùng đối thủ). Màn nào bản mới thắng nhiều hơn mới được thay vào `models/zombie_lvK.zip` (game dùng file này). Mọi bản cũ lưu ở `models/league/`.
3. **Người chơi học lại** để khắc chế bản zombie tốt nhất, rồi vào nhóm đối thủ của vòng sau.
4. **Đo**: zombie tốt nhất đấu với người chơi vừa luyện để khắc chế nó. Đây là thước đo "khôn thật": zombie chỉ biết đi theo thuật toán sẽ bị khai thác ngay.

Đường học (learning curve) của PPO huấn luyện từ đầu, nhiều seed:

```powershell
python tools/learning_curve.py --level 1 --steps 250000 --seed 0   # ghi models/learning_curve_lv1_seed0.csv
```

Theo dõi tiến bộ trong `models/league.csv`. Sau khi tự train bằng `train.py`, có thể so với bản đang dùng: `python league.py --promote-from <thư mục model mới>`.
