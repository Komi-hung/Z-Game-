"""Xuất mọi model models/*.zip sang .npz để chạy không cần torch (xem numpy_policy.py)."""
import glob, io, os, sys, zipfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from numpy_policy import algo_class

for path in sorted(glob.glob(os.path.join("models", "*.zip")) + glob.glob(os.path.join("models", "league", "*", "*.zip"))
                   + glob.glob(os.path.join("models", "algos", "*", "*", "*.zip"))):
    try:
        m = algo_class(path).load(path, device="cpu")
        sd = {k: v.cpu().numpy() for k, v in m.policy.state_dict().items()}
    except Exception as e:
        print("bỏ qua", path, e); continue
    if "q_net.q_net.0.weight" in sd:          # DQN: lớp cuối của Q-network là "logits", chọn argmax Q
        pre = "q_net.q_net."
        idx = sorted({int(k.split(".")[2]) for k in sd if k.startswith(pre)})
        hid, last = idx[:-1], idx[-1]
        out = {"n_hidden": len(hid), "out_w": sd[f"{pre}{last}.weight"], "out_b": sd[f"{pre}{last}.bias"],
               "act": "relu"}
    else:                                     # PPO / A2C / TRPO: actor MLP + action_net
        pre = "mlp_extractor.policy_net."
        hid = sorted(int(k.split(".")[2]) for k in sd if k.startswith(pre) and k.endswith("weight"))
        out = {"n_hidden": len(hid), "out_w": sd["action_net.weight"], "out_b": sd["action_net.bias"],
               "act": "tanh"}
    for i, h in enumerate(hid):
        out[f"w{i}"], out[f"b{i}"] = sd[f"{pre}{h}.weight"], sd[f"{pre}{h}.bias"]
    sp = m.action_space
    out["nvec"] = np.array(sp.nvec if hasattr(sp, "nvec") else [sp.n])
    np.savez(os.path.splitext(path)[0] + ".npz", **out)
    print("đã xuất", os.path.splitext(path)[0] + ".npz")
