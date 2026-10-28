"""Luyện zombie ĐỐI KHÁNG ngay trong game thật (game_core.World) với người chơi RL.

Khác env_zombie.py (sân tập đơn giản 1 zombie vs người chơi giả viết tay):
  - Bản đồ, đợt zombie, đồng đội, súng, 60 khung hình/giây: y như khi người thật chơi.
  - Người chơi là agent RL (models/player_agent.zip và các bản cũ trong league),
    thỉnh thoảng là người chơi viết tay -> zombie phải học né / ép góc / chọn lúc đánh
    trước một đối thủ BIẾT HỌC, chứ không chỉ đi theo đường.
  - Mỗi episode có 1 zombie "người học" (ở màn boss: chính boss). Các zombie khác dùng
    model tốt nhất hiện tại (đồng đội, giữ nguyên trong vòng này).

Episode kết thúc khi: zombie người học chết / người chơi chết / màn kết thúc / hết giờ.
"""
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import ZOMBIES, REWARD, DECISION_DT, MODEL_DIR
from game_core import World, Controls, ModelBank
from geometry import in_fire_line
from player_env import (build_player_obs, decode_action, ScriptedPlayer, FRAME_DT, FRAMES_PER_ACTION,
                        _single_thread_torch)
from zombie_brain import N_ACTIONS, OBS_SIZE

EPISODE_TIME = 90.0      # giây


class ZombieWorldEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level, opponents=(), scripted_prob=0.15, teammate_dir=MODEL_DIR):
        """opponents: danh sách đường dẫn model người chơi (PPO). scripted_prob: tỉ lệ gặp người chơi viết tay."""
        super().__init__()
        _single_thread_torch()
        from stable_baselines3 import PPO
        self.level = level
        self.opponents = [PPO.load(p, device="cpu") for p in opponents]
        self.scripted = ScriptedPlayer()
        self.scripted_prob = scripted_prob if self.opponents else 1.0
        self.bank = ModelBank(teammate_dir)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)

    def _obs(self):
        z = self.learner
        p = self.world.player
        z.path = self.world.pf.step(z.x, z.y, p.x, p.y, z.radius)
        return self.world.zombie_obs(z)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.world = World(self.level, "ppo", seed=int(rng.integers(1 << 31)), model_bank=self.bank)
        alive = [z for z in self.world.zombies if z.alive]
        bosses = [z for z in alive if z.type == "boss"]
        self.learner = bosses[0] if bosses else alive[int(rng.integers(len(alive)))]
        self.world.learner = self.learner
        if rng.random() < self.scripted_prob:
            self.opponent, self.stochastic = None, False
        else:
            self.opponent = self.opponents[int(rng.integers(len(self.opponents)))]
            self.stochastic = rng.random() < 0.5          # một nửa số trận người chơi hành động có ngẫu nhiên
        obs = self._obs()
        self.prev_path = self.learner.path[2]
        self.prev_dealt = self.prev_taken = 0.0
        self.prev_whiffs = 0
        self.steps = self.hits = 0
        return obs, {"type": self.learner.type}

    def _player_controls(self):
        w = self.world
        if self.opponent is None:
            action = self.scripted.act(w)
        else:
            action, _ = self.opponent.predict(build_player_obs(w), deterministic=not self.stochastic)
        return decode_action(action, w.player)

    def step(self, action):
        w, z = self.world, self.learner
        self.steps += 1
        w.learner_action = int(action)
        w.decision_t = DECISION_DT            # zombie quyết định ngay ở khung hình đầu (khớp nhịp 0.1 s)
        fwd, strafe, turn_rate, fire, reload_, switch = self._player_controls()
        for f in range(FRAMES_PER_ACTION):
            ctrl = Controls(fwd, strafe, turn_rate * FRAME_DT, fire,
                            reload=reload_ and f == 0, switch=switch if f == 0 else None)
            w.step(FRAME_DT, ctrl)
            if w.state != "play" or not z.alive:
                break

        p = w.player
        obs = self._obs()
        st = ZOMBIES[z.type]
        reward = REWARD["step"]
        # Thưởng theo quãng ĐƯỜNG ĐI rút ngắn (không tính lúc đang lao tới do kịch bản boss)
        reward += REWARD["approach"] * (self.prev_path - z.path[2])
        self.prev_path = z.path[2]
        if z.alive and in_fire_line(w.grid, p.x, p.y, p.angle, z.x, z.y, z.radius, w.player_weapon()):
            reward += REWARD["aimed"]
        d_taken = z.dmg_taken - self.prev_taken
        d_dealt = z.dmg_dealt - self.prev_dealt
        reward += REWARD["hit_taken"] * d_taken / z.max_hp
        if d_dealt > 0:
            self.hits += 1
            reward += REWARD["hit_dealt"] * d_dealt / st["damage"]
        reward += REWARD["whiff"] * (z.whiffs - self.prev_whiffs)
        self.prev_taken, self.prev_dealt, self.prev_whiffs = z.dmg_taken, z.dmg_dealt, z.whiffs

        terminated, outcome = False, None
        if w.state == "dead":
            reward += REWARD["kill_player"]
            terminated, outcome = True, "won"
        elif not z.alive:
            reward += REWARD["death"]
            terminated, outcome = True, "killed"
        elif w.state != "play":
            terminated, outcome = True, "level_end"
        truncated = not terminated and w.play_time >= EPISODE_TIME
        if truncated:
            outcome = "timeout"
        info = {"outcome": outcome, "type": z.type, "hits": self.hits, "hp_lost": z.dmg_taken / z.max_hp}
        return obs, float(reward), terminated, truncated, info
