"""Chạy policy PPO mà KHÔNG cần PyTorch (dùng khi Windows chặn torch DLL).

Mạng của SB3 MlpPolicy chỉ là MLP tanh nên suy luận bằng numpy cho kết quả giống hệt
predict(deterministic=True). Trọng số được xuất sang file .npz cạnh file .zip bằng
tools/export_numpy.py (chạy một lần trên máy có torch)."""
import os
import numpy as np


class NumpyPolicy:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.layers = [(d[f"w{i}"], d[f"b{i}"]) for i in range(int(d["n_hidden"]))]
        self.out_w, self.out_b = d["out_w"], d["out_b"]
        self.nvec = [int(x) for x in d["nvec"]]          # [18] = Discrete, [3,3,9,2,2,2] = MultiDiscrete
        self.obs_size = self.layers[0][0].shape[1]
        self.observation_space = type("S", (), {"shape": (self.obs_size,)})()

    def predict(self, obs, deterministic=True):
        x = np.asarray(obs, dtype=np.float32)
        single = x.ndim == 1
        x = np.atleast_2d(x)
        for w, b in self.layers:
            x = np.tanh(x @ w.T + b)
        logits = x @ self.out_w.T + self.out_b
        if len(self.nvec) == 1:
            act = logits.argmax(axis=1)
        else:
            parts, i = [], 0
            for n in self.nvec:
                parts.append(logits[:, i:i + n].argmax(axis=1))
                i += n
            act = np.stack(parts, axis=1)
        return (act[0] if single else act), None


def load_policy(zip_path):
    """PPO.load nếu có torch; nếu Windows chặn torch thì dùng file .npz cùng tên."""
    try:
        from stable_baselines3 import PPO
        return PPO.load(zip_path, device="cpu")
    except (ImportError, OSError) as e:
        npz = os.path.splitext(zip_path)[0] + ".npz"
        if not os.path.exists(npz):
            raise OSError(f"Không nạp được torch ({e}) và chưa có {npz}. "
                          f"Chạy `python tools/export_numpy.py` trên máy có torch chạy được.") from e
        return NumpyPolicy(npz)
