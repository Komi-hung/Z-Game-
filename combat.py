"""Luật đánh nhau dùng CHUNG cho env (train) và game (chơi).

AttackState: zombie vung tay (windup) rồi mới gây damage, sau đó phải hồi chiêu.
  - Zombie PPO tự QUYẾT ĐỊNH lúc ra đòn (action "đánh"), nên có thể chờ thời cơ:
    người chơi bị dồn vào tường, đang nạp đạn, vừa lùi hết đà...
  - Zombie viết tay (baseline) thì cứ vào tầm là đánh (want=True mọi lúc).
  - Trong lúc vung tay, zombie vẫn nhích theo người chơi (lunge) như đòn cận chiến thật,
    nên người chơi không thể né chỉ bằng cách lùi đúng nhịp.
BossMods: các pha viết tay của boss (pha 2 nhanh hơn, lao tới khi ở xa).
"""
from config import BOSS

RECOVER = 0.35     # thời gian thu tay sau khi ra đòn


class AttackState:
    def __init__(self, stats):
        self.st = stats
        self.cooldown = 0.0
        self.t = 0.0
        self.active = False
        self.struck = False
        self.whiffed = False        # True đúng bước vung tay trượt

    @property
    def ready(self):
        return not self.active and self.cooldown <= 0

    @property
    def winding_up(self):
        return self.active and not self.struck

    def update(self, dt, dist, los, want=True):
        """want: zombie có muốn ra đòn ở bước này không.
        Trả về (đang bận đánh -> không tự di chuyển, damage gây ra bước này)."""
        self.cooldown -= dt
        self.whiffed = False
        if not self.active:
            if want and dist < self.st["attack_range"] and self.cooldown <= 0:
                self.active, self.t, self.struck = True, 0.0, False
            else:
                return False, 0.0
        self.t += dt
        damage = 0.0
        if not self.struck and self.t >= self.st["windup"]:          # damage đúng lúc ra đòn
            self.struck = True
            if dist < self.st["attack_range"] + 0.3 and los:
                damage = float(self.st["damage"])
            else:
                self.whiffed = True
        if self.t >= self.st["windup"] + RECOVER:
            self.active, self.cooldown = False, self.st["attack_cooldown"]
        return True, damage


class BossMods:
    def __init__(self):
        self.charge_t = 0.0
        self.charge_cd = 0.0

    def update(self, dt, hp_frac, dist, los):
        """Trả về (hệ số tốc độ, đang lao tới hay không)."""
        mult = 1.0
        phase2 = hp_frac < BOSS["phase2_hp"]
        if phase2:
            mult *= BOSS["phase2_speed_mult"]
        self.charge_cd -= dt
        if (phase2 and self.charge_t <= 0 and self.charge_cd <= 0
                and dist > BOSS["charge_min_dist"] and los):
            self.charge_t, self.charge_cd = BOSS["charge_time"], BOSS["charge_cooldown"]
        charging = self.charge_t > 0
        if charging:
            self.charge_t -= dt
            mult *= BOSS["charge_speed_mult"]
        return mult, charging
