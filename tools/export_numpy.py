"""Xuất mọi model models/*.zip sang .npz để chạy không cần torch (xem numpy_policy.py)."""
import glob, io, os, sys, zipfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from stable_baselines3 import PPO

for path in sorted(glob.glob(os.path.join("models", "*.zip")) + glob.glob(os.path.join("models", "league", "*", "*.zip"))):
    try:
        m = PPO.load(path, device="cpu")
        sd = {k: v.cpu().numpy() for k, v in m.policy.state_dict().items()}
    except Exception as e:
        print("bỏ qua", path, e); continue
    hid = sorted(int(k.split(".")[2]) for k in sd if k.startswith("mlp_extractor.policy_net.") and k.endswith("weight"))
    out = {"n_hidden": len(hid), "out_w": sd["action_net.weight"], "out_b": sd["action_net.bias"]}
    for i, h in enumerate(hid):
        out[f"w{i}"], out[f"b{i}"] = sd[f"mlp_extractor.policy_net.{h}.weight"], sd[f"mlp_extractor.policy_net.{h}.bias"]
    sp = m.action_space
    out["nvec"] = np.array(sp.nvec if hasattr(sp, "nvec") else [sp.n])
    np.savez(os.path.splitext(path)[0] + ".npz", **out)
    print("đã xuất", os.path.splitext(path)[0] + ".npz")
