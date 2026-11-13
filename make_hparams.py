"""Đọc siêu tham số THẬT từ các file model đã lưu (không gõ tay) -> hparams.json."""
import sys, json, os
os.chdir("/mnt/user-data/outputs/zombie-rl"); sys.path.insert(0, ".")
import stable_baselines3 as sb3
from sb3_contrib import TRPO
P = {"ppo": (sb3.PPO, "models/algos/ppo/seed0/zombie_lv1.zip"), "trpo": (TRPO, "models/algos/trpo/seed0/zombie_lv1.zip"),
     "a2c": (sb3.A2C, "models/algos/a2c/seed0/zombie_lv1.zip"), "dqn": (sb3.DQN, "models/algos/dqn/seed0/zombie_lv1.zip"),
     "dqn_tgt": (sb3.DQN, "models/algos/dqn_tgt/seed0/zombie_lv2.zip"), "player": (sb3.PPO, "models/player_agent.zip")}
KEYS = ["n_envs", "gamma", "n_steps", "batch_size", "n_epochs", "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm", "target_kl", "cg_max_steps",
        "cg_damping", "line_search_shrinking_factor", "line_search_max_iter", "n_critic_updates", "buffer_size", "learning_starts", "tau",
        "gradient_steps", "target_update_interval", "exploration_fraction", "exploration_initial_eps", "exploration_final_eps", "normalize_advantage"]
out = {}
for n, (cls, path) in P.items():
    m = cls.load(path, device="cpu"); d = {"file": path}
    for k in KEYS:
        if hasattr(m, k): d[k] = getattr(m, k)
    lr = m.learning_rate; d["learning_rate"] = float(lr(1.0)) if callable(lr) else float(lr)
    if hasattr(m, "clip_range"): d["clip_range"] = float(m.clip_range(1.0))
    if hasattr(m, "train_freq"): d["train_freq"] = int(m.train_freq.frequency)
    pol = m.policy
    d["params_total"] = sum(p.numel() for p in pol.parameters())
    if hasattr(pol, "q_net"): d["params_qnet"] = sum(p.numel() for p in pol.q_net.parameters())
    d["optimizer"] = type(pol.optimizer).__name__
    net = pol.q_net.q_net if hasattr(pol, "q_net") else pol.mlp_extractor.policy_net
    d["activation"] = type(net[1]).__name__
    d["net_arch"] = str(m.policy_kwargs.get("net_arch", "mặc định SB3: 64, 64"))
    d["obs_dim"] = int(m.observation_space.shape[0]); sp = m.action_space
    d["action_space"] = str(sp)
    out[n] = d
json.dump(out, open("/tmp/claude-0/-home-claude/da93a6da-0ef4-501e-8ab2-24ed2e6e4c4a/scratchpad/report/hparams.json", "w"), indent=1, ensure_ascii=False, default=str)
for n, d in out.items(): print(n, d["optimizer"], d["activation"], d["params_total"], d.get("params_qnet"), d["learning_rate"])
