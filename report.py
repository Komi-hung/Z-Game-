"""Dựng báo cáo PDF từ stats.json + hình do analysis.py tạo."""
import json, os, datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether, NextPageTemplate)

import math, sys, re as _re
import pandas as _pd0
D = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{D}/stats.json"))
HP = json.load(open(f"{D}/hparams.json"))            # siêu tham số đọc từ model đã lưu (make_hparams.py)
sys.path.insert(0, "/mnt/user-data/outputs/zombie-rl"); import config as CFG
OUT = f"{D}/Bao_cao_so_sanh_thuat_toan_RL.pdf"

FD = "/usr/share/fonts/truetype/liberation/"
for n, f in [("TN", "LiberationSerif-Regular.ttf"), ("TNB", "LiberationSerif-Bold.ttf"), ("TNI", "LiberationSerif-Italic.ttf"),
             ("TNBI", "LiberationSerif-BoldItalic.ttf"), ("TNM", "LiberationMono-Regular.ttf")]:
    pdfmetrics.registerFont(TTFont(n, FD + f))
pdfmetrics.registerFontFamily("TN", normal="TN", bold="TNB", italic="TNI", boldItalic="TNBI")

INK = colors.HexColor("#140a0b"); GREY = colors.HexColor("#5a4a4c"); RED = colors.HexColor("#9b1c2c")
LIGHT = colors.HexColor("#f7f2f2"); LINE = colors.HexColor("#d9cdce"); GOOD = colors.HexColor("#2f7d4f")

def ps(name, **kw):
    base = dict(fontName="TN", fontSize=11.5, leading=16, textColor=INK, spaceAfter=5)
    base.update(kw); return ParagraphStyle(name, **base)
BODY = ps("body", alignment=4)
H1 = ps("h1", keepWithNext=1, fontName="TNB", fontSize=15, leading=19, spaceBefore=12, spaceAfter=7, textColor=INK)
H2 = ps("h2", keepWithNext=1, fontName="TNB", fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4, textColor=RED)
CAP = ps("cap", fontName="TNI", fontSize=9.6, leading=12, textColor=GREY, alignment=1, spaceAfter=9)
CELL = ps("cell", fontName="TN", fontSize=9.4, leading=11.6, spaceAfter=0)
CELLB = ps("cellb", fontName="TNB", fontSize=9.4, leading=11.6, spaceAfter=0, textColor=colors.white)
BUL = ps("bul", leftIndent=14, bulletIndent=3, spaceAfter=2.5, alignment=4)
NOTE = ps("note", fontName="TN", fontSize=10.3, leading=13.8, backColor=LIGHT, borderPadding=(6, 8, 6, 8),
          spaceBefore=4, spaceAfter=10, borderColor=LINE, borderWidth=0.5)
EQ = ps("eq", fontName="TN", fontSize=11.5, leading=16, alignment=1, spaceBefore=2, spaceAfter=8)

story = []
P = lambda t, s=BODY: story.append(Paragraph(t, s))
def B(items):
    for t in items: story.append(Paragraph(t, BUL, bulletText="•"))
FIGN = [0]
def fig(name, width, cap):
    from reportlab.lib.utils import ImageReader
    FIGN[0] += 1; cap = f"Hình {FIGN[0]}. " + _re.sub(r"^Hình \d+\.\s*", "", cap)
    iw, ih = ImageReader(f"{D}/{name}").getSize()
    story.append(KeepTogether([Image(f"{D}/{name}", width=width, height=width * ih / iw), Paragraph(cap, CAP)]))
def gv(v):                                    # số -> chuỗi kiểu Việt (dấu phẩy thập phân)
    if v is None: return "–"
    if isinstance(v, bool): return "có" if v else "không"
    return (f"{v:g}" if isinstance(v, float) else str(v)).replace(".", ",")
def lrf(v):                                   # learning rate dạng m×10^e
    e = math.floor(math.log10(v)); m = v / 10 ** e
    return f"{m:g}×10<super>{str(e).replace('-', '−')}</super>".replace(".", ",")
def T(data, widths, hl_rows=(), align_right_from=1):
    rows = [[Paragraph(str(v), CELLB if i == 0 else CELL) for v in r] for i, r in enumerate(data)]
    t = Table(rows, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), INK), ("GRID", (0, 0), (-1, -1), 0.35, LINE),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.6),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for i in range(1, len(rows)):
        if i % 2 == 0: st.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    for i in hl_rows: st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f3dcdf")))
    t.setStyle(TableStyle(st)); story.append(t); story.append(Spacer(1, 8))

L = S["levels"]; BS = S["base"]; R = S["reach"]
A4W = A4[0] - 40 * mm
NAME = {"ppo": "PPO", "trpo": "TRPO", "a2c": "A2C", "dqn": "DQN"}
ALG = ["ppo", "trpo", "a2c", "dqn"]
pct = lambda v: f"{v * 100:.0f}%"
f2 = lambda v: f"{v:.2f}"
def reach_str(a): return " / ".join(str(r["reach"]) for r in R[a])
def reach_mean(a): return sum(r["reach"] for r in R[a]) / len(R[a])
def full(a): return sum(r["reach"] == 5 for r in R[a])
NS = {a: len(R[a]) for a in ALG}
NEP = S["episodes"]
import pandas as _pd
_ME = _pd.read_csv("/mnt/user-data/outputs/zombie-rl/models/match_eval.csv")
_MB = _pd.read_csv("/mnt/user-data/outputs/zombie-rl/models/match_eval_before_retrain.csv")
def _mz(df, player, z, lv):
    """số trận zombie thắng; lv=0 là cả game (campaign: tính theo lượt, một lượt thua khi có dòng zombie_win)."""
    if lv == 0:
        d = df[(df["mode"] == "campaign") & (df.player == player) & (df.zombies == z)]
        return int(d[d.outcome == "zombie_win"].seed.nunique())
    d = df[(df["mode"] == "level") & (df.player == player) & (df.zombies == z) & (df.level == lv)]
    return int((d.outcome == "zombie_win").sum())
mz = lambda p, z, l: _mz(_ME, p, z, l)
mzb = lambda p, z, l: _mz(_MB, p, z, l)
MN = int(len(_ME[(_ME["mode"] == "level") & (_ME.player == "agent") & (_ME.zombies == "ppo") & (_ME.level == 5)]))
RETRAIN = "1,5 triệu"
_TU = _pd.read_csv("/mnt/user-data/outputs/zombie-rl/models/algo_compare_tune.csv").drop_duplicates(["algo", "seed", "level"], keep="last")
_TD = _pd.read_csv("/mnt/user-data/outputs/zombie-rl/models/algo_compare_tuned.csv").drop_duplicates(["algo", "seed", "level"], keep="last")
_DEF = _pd.read_csv("/mnt/user-data/outputs/zombie-rl/models/algo_compare.csv").drop_duplicates(["algo", "seed", "level"], keep="last")
_DEF = _DEF[_DEF.algo == "dqn"]
def _reach(df):
    out = {}
    for s, h in df.groupby("seed"):
        h = h.set_index("level"); k = 0
        while k + 1 in h.index and h.loc[k + 1, "passed"] == 1:
            k += 1
        out[int(s)] = k
    return out
import math as _m
from scipy import stats as _st
def _tci(x):
    x = list(x); m = sum(x) / len(x)
    if len(x) < 2: return m, m, m
    sd = (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5
    h = _st.t.ppf(0.975, len(x) - 1) * sd / _m.sqrt(len(x)); return m, m - h, m + h
best = max(ALG, key=lambda a: (reach_mean(a), L[f"{a}_5"]["win"]))

# ---------------- Trang bìa ----------------
story.append(Spacer(1, 52 * mm))
story.append(Paragraph("BÁO CÁO THÍ NGHIỆM", ps("k", fontName="TNB", fontSize=10, textColor=RED, leading=13)))
story.append(Paragraph("So sánh PPO, TRPO, A2C và DQN cho kẻ địch trong game Zombie RL", ps("t", fontName="TNB", fontSize=22, leading=28, spaceAfter=10)))
story.append(Paragraph("kèm thiết kế lại trận boss để chiến thắng đến từ kỹ năng học được thay vì chỉ số", ps("st", fontName="TNI", fontSize=12.5, leading=17, textColor=GREY, spaceAfter=28)))
story.append(Table([[""]], colWidths=[60 * mm], rowHeights=[2], style=[("BACKGROUND", (0, 0), (-1, -1), RED)]))
story.append(Spacer(1, 14))
for t in ["Môn học: REL (Reinforcement Learning)",
          "Nhóm: Nguyễn Võ Trung Hưng (SE193635) · Ngô Phúc Nguyên (SE192516) · Lê Như Đạt (SE192807)",
          f"Ngày: {datetime.date.today().strftime('%d/%m/%Y')}",
          f"Quy mô thí nghiệm: 4 thuật toán × {max(NS.values())} seed × 5 màn, {NEP} episode đánh giá mỗi ô"]:
    story.append(Paragraph(t, ps("c", fontName="TN", fontSize=10, leading=15, textColor=GREY)))
story.append(NextPageTemplate("body")); story.append(PageBreak())

# ---------------- Tóm tắt ----------------
P("Tóm tắt", H1)
P(f"Báo cáo trả lời hai câu hỏi. <b>(1)</b> Với cùng môi trường, cùng reward, cùng mạng 64×64 và cùng số bước huấn luyện, "
  f"thuật toán RL nào đưa zombie đi xa nhất trong curriculum 5 màn? <b>(2)</b> Trận boss trước đây thắng nhờ kỹ năng hay nhờ chỉ số, "
  f"và làm sao để mô hình phải học kỹ năng né đạn?")
B(["<b>Học tăng cường</b>: mỗi zombie là một agent với observation 29 số, 18 action và reward có định hình (mục 1); học qua curriculum 5 màn, "
   "tổng 1,9 triệu bước mỗi lượt. Thông số chi tiết của từng thuật toán ở mục 2.2.",
   f"<b>PPO</b> qua đủ 5 màn ở {full('ppo')}/{NS['ppo']} seed, <b>TRPO</b> {full('trpo')}/{NS['trpo']}, "
   f"<b>A2C</b> {full('a2c')}/{NS['a2c']}, <b>DQN</b> {full('dqn')}/{NS['dqn']}. "
   f"Số màn đạt được trung bình: PPO {reach_mean('ppo'):.1f}, TRPO {reach_mean('trpo'):.1f}, A2C {reach_mean('a2c'):.1f}, DQN {reach_mean('dqn'):.1f}.",
   f"Ở màn boss, tỉ lệ thắng: PPO {pct(L['ppo_5']['win'])}, TRPO {pct(L['trpo_5']['win'])}, A2C {pct(L['a2c_5']['win'])}, DQN {pct(L['dqn_5']['win'])}, "
   f"so với boss đi thẳng (path) {pct(BS['path_5']['win'])} và boss zig-zag viết tay {pct(BS['zigzag_5']['win'])}.",
   "Boss cũ (1500 máu, tự lao tới) thắng ~90% ngay cả khi chỉ đi thẳng: sức mạnh nằm ở chỉ số. Bản chỉnh đầu tiên (700 máu) cân bằng trong sân tập "
   "nhưng chưa cân bằng trong game thật (boss chưa thắng trận nào trong 40 trận), vì người chơi giả trong sân tập bắn chậm bằng nửa game. Sau khi đồng bộ người chơi giả với game và đặt lại máu boss 1300, "
   f"boss RL né được {pct(L['ppo_5']['dodge'])} số phát (PPO) so với {pct(BS['path_5']['dodge'])} của boss đi thẳng.",
   f"Kết quả của DQN ở bảng chính phụ thuộc nhiều vào cách học tiếp giữa các màn: khi giữ replay buffer, không cho ε chạy lại từ 1,0 và cập nhật target thường hơn, "
   f"DQN đạt tới màn {sum(_reach(_TD).values()) / 5:.1f} thay vì {sum(_reach(_DEF).values()) / 5:.1f} trung bình (mục 5.5; chỉ DQN được chỉnh).",
   f"Trong game thật, đấu với agent người chơi RL ({MN} trận mỗi cặp): boss PPO thắng {mz('agent','ppo',5)}/{MN} trận ở màn boss, "
   f"boss đi thẳng {mz('agent','path',5)}/{MN}. Tính cả game 5 màn, zombie PPO thắng {mz('agent','ppo',0)}/{MN} lượt so với {mz('agent','path',0)}/{MN} của zombie đi theo đường."])

# ---------------- 1. Bài toán ----------------
P("1. Học tăng cường trong dự án: bài toán và môi trường", H1)
P("1.1 Mô hình hóa bài toán", H2)
P("Học tăng cường (reinforcement learning, RL) dạy một <b>tác tử</b> (agent) hành động bằng thử và sai. Ở mỗi bước, agent nhận <b>observation</b> "
  "s<sub>t</sub> từ môi trường, chọn <b>action</b> a<sub>t</sub> theo <b>policy</b> π<sub>θ</sub>(a|s), rồi nhận <b>reward</b> r<sub>t</sub>. "
  "Mục tiêu là tìm tham số θ làm tổng reward chiết khấu kỳ vọng J(θ) = E[ Σ γ<super>t</super> r<sub>t</sub> ] lớn nhất, với hệ số chiết khấu γ = 0,99 "
  "(γ gần 1 nghĩa là agent quan tâm cả phần thưởng ở xa, không chỉ phần thưởng trước mắt).")
fig("fig_rl_loop.png", 135 * mm, "Vòng lặp học tăng cường trong dự án. Mỗi 0,1 giây agent nhận observation (29 số) và reward, chọn một trong 18 action; "
    "các bộ (s, a, r) thu được dùng để cập nhật θ theo thuật toán đang thử.")
P("Trong dự án, <b>mỗi zombie là một agent</b>: policy là mạng nơ-ron MLP 64×64 nhận 29 số và trả về xác suất của 18 action. Không hành vi nào được lập trình sẵn; "
  "cách áp sát, vòng qua tường hay né đạn đều đến từ reward và quá trình thử và sai. Bài toán là <b>POMDP</b> (quan sát một phần) vì zombie chỉ thấy vector 29 số, "
  "không thấy toàn bản đồ và không biết người chơi định làm gì tiếp. Ngoài zombie còn một agent RL thứ hai là <b>agent người chơi</b> (mục 2.3), "
  "được huấn luyện riêng để làm đối thủ thử: nếu zombie vẫn thắng được một người chơi biết học thì chính sách của zombie mới thật sự đáng tin.")
P("1.2 Observation và action", H2)
P("Môi trường huấn luyện là <font name='TNM'>env_zombie.py</font> (Gymnasium): một zombie đấu với một người chơi giả viết tay (PlayerBot) có độ trễ phản xạ và sai số ngắm. "
  "Observation gồm 29 giá trị, đều được chuẩn hóa về khoảng [−1, 1]:")
T([["Nhóm", "Số giá trị", "Ý nghĩa"],
   ["Tia tới tường", "8", "Khoảng cách từ zombie tới tường theo 8 hướng đều nhau (chia cho 8 ô), cho biết địa hình xung quanh"],
   ["Người chơi (đường thẳng)", "3", "Vector đơn vị hướng tới người chơi và khoảng cách"],
   ["Đường đi thật", "3", "Hướng bước kế tiếp trên đường đi ngắn nhất (Dijkstra, vòng qua tường) và độ dài đường đi"],
   ["Chuyển động người chơi", "2", "Vận tốc theo hai trục"],
   ["Hướng nhìn người chơi", "2", "(cos, sin) của góc nhìn"],
   ["Đang bị ngắm", "1", "Cờ 0/1: zombie nằm trong đường đạn của súng người chơi đang cầm"],
   ["Loại zombie", "4", "One-hot: mập, vừa, ốm, boss"],
   ["Súng người chơi", "2", "One-hot: súng lục, M4A1"],
   ["Máu zombie", "1", "Tỉ lệ máu còn lại"],
   ["Trạng thái chiến đấu", "3", "Người chơi đang nạp đạn; tỉ lệ đạn còn lại; đòn của zombie đã sẵn sàng chưa"],
   ["<b>Tổng</b>", "<b>29</b>", ""]],
  [40 * mm, 20 * mm, A4W - 60 * mm])
P("<b>Action</b> gồm 18 lựa chọn rời rạc: 9 hướng đi (đứng yên, 4 hướng thẳng, 4 hướng chéo) × 2 trạng thái đòn (cho phép đánh khi người chơi trong tầm, "
  "hoặc giữ đòn để chờ thời cơ). Một quyết định kéo dài 0,1 giây, cả lúc huấn luyện lẫn lúc chơi.")
fig("fig_ai_view.png", A4W, "Màn 3 trong game thật. Phóng to bên phải: minimap kèm thông tin quyết định của zombie RL (zombie gần nhất, action đang chọn, có đang bị ngắm hay không). "
    "Chấm vàng trên minimap là hướng mỗi zombie đang đi.")
P("1.3 Reward", H2)
P("Nếu chỉ thưởng khi bắt được người chơi thì reward rất thưa và agent học rất chậm, nên nhóm dùng <b>reward shaping</b>: thêm các thưởng và phạt nhỏ ở mỗi bước "
  "để dẫn dắt hành vi. Các giá trị lấy từ <font name='TNM'>config.py</font>:")
RW = CFG.REWARD
T([["Thành phần", "Giá trị", "Hành vi được khuyến khích"],
   ["approach", f"+{gv(RW['approach'])} × số ô đường đi rút ngắn", "Đi theo đường thật tới người chơi (không phải đường chim bay), nên học được cách vòng qua tường"],
   ["catch", f"+{gv(RW['catch'])}", "Zombie thường đánh trúng người chơi lần đầu (kết thúc episode)"],
   ["hit_dealt", f"+{gv(RW['hit_dealt'])} mỗi đòn", "Trận boss: mỗi đòn trúng người chơi"],
   ["kill_player", f"+{gv(RW['kill_player'])}", "Trận boss: hạ được người chơi"],
   ["aimed", f"{gv(RW['aimed'])} mỗi bước", "Phạt khi đứng trong đường ngắm: nguồn gốc của hành vi né đạn"],
   ["hit_taken", f"{gv(RW['hit_taken'])} × damage / máu tối đa", "Phạt khi bị bắn trúng, tỉ lệ với sát thương"],
   ["whiff", f"{gv(RW['whiff'])}", "Vung tay trượt: học chọn đúng thời điểm đánh thay vì đánh bừa"],
   ["death", f"{gv(RW['death'])}", "Zombie bị hạ"],
   ["step", f"{gv(RW['step'])} mỗi bước", "Phạt nhẹ theo thời gian để đi nhanh hơn"]],
  [28 * mm, 46 * mm, A4W - 74 * mm])
P("Reward định nghĩa điều đáng học. Ở trận boss ban đầu, người chơi giả ngắm gần như hoàn hảo nên né đạn không giúp giảm sát thương; "
  "chính sách tối ưu theo reward là lao vào và dựa vào chỉ số máu (mục 3). Đây là bài học trung tâm của dự án: muốn agent học kỹ năng, môi trường và reward phải khiến kỹ năng có lợi.", NOTE)
P("1.4 Curriculum: 5 màn khó dần", H2)
P("Nếu đặt agent vào màn khó ngay từ đầu, nó hầu như không gặp reward dương nên không học được. <b>Curriculum learning</b> chia bài toán thành 5 màn: "
  "màn sau học tiếp từ model của màn trước (warm-start) và 50% episode ôn lại bản đồ các màn cũ để không quên. Một lượt huấn luyện đầy đủ là "
  f"{sum(l['train_steps'] for l in CFG.LEVELS) / 1e6:.1f} triệu bước.".replace(".", ","))
fig("fig_maps.png", 140 * mm, "Bản đồ 5 màn (P: điểm xuất phát của người chơi, Z: điểm sinh zombie, B: boss). Màn sau có nhiều tường, cột và đợt zombie hơn.")
_VN = {"vest": "vừa", "heavy": "mập", "skinny": "ốm", "boss": "boss"}
_WP = {1: "súng lục", 2: "súng lục", 3: "súng lục hoặc M4A1", 4: "súng lục hoặc M4A1", 5: "M4A1"}
_rows = [["Màn", "Bước train", "Zombie khi train", "Số đợt", "Người chơi giả: phản xạ · nhiễu ngắm · thả diều", "Súng"]]
for i, lv in enumerate(CFG.LEVELS, 1):
    bt = lv["bot"]
    _rows.append([lv["name"].split(" - ")[1], f"{lv['train_steps'] // 1000}k",
                  ", ".join(f"{_VN[k]} {v * 100:.0f}%" for k, v in lv["train_mix"].items()), str(len(lv["waves"])),
                  f"{bt['reaction']} bước ({bt['reaction'] / 10:.1f} s) · {gv(bt['aim_noise'])} · {bt['kite'] * 100:.0f}%", _WP[i]])
T(_rows, [26 * mm, 17 * mm, 38 * mm, 14 * mm, A4W - 123 * mm, 28 * mm])
fig("fig_levels.png", A4W, "Ảnh chụp từ game thật ở 5 màn (người chơi tự động, zombie do model RL điều khiển): kho chứa, hành lang, tầng hầm, phòng thí nghiệm và boss.")
_z = [["Loại", "Máu", "Sát thương", "Tốc độ (ô/s)", "Tầm đánh", "Hồi chiêu (s)"]]
for k in ("vest", "heavy", "skinny", "boss"):
    z = CFG.ZOMBIES[k]; _z.append([z["name"], str(z["hp"]), str(z["damage"]), gv(z["speed"]), gv(z["attack_range"]), gv(z["attack_cooldown"])])
T(_z, [34 * mm] + [(A4W - 34 * mm) / 5] * 5)
_w = [["Súng người chơi", "Sát thương", "Hồi chiêu (s)", "Băng đạn", "Nạp đạn (s)", "Tầm bắn (ô)"]]
for k in ("pistol", "m4a1"):
    w = CFG.WEAPONS[k]; _w.append([w["name"], str(w["damage"]), gv(w["cooldown"]), str(w["mag"]), gv(w["reload"]), str(w["range"])])
T(_w, [34 * mm] + [(A4W - 34 * mm) / 5] * 5)
P(f"Boss có hai pha: khi máu còn dưới {CFG.BOSS['phase2_hp'] * 100:.0f}%, tốc độ tăng {gv(CFG.BOSS['phase2_speed_mult'])} lần. Mọi bước di chuyển của boss do model quyết định.", CAP)
P("1.5 Đường học thực tế", H2)
_cv = {}
for _lv in (1, 2, 4):
    _fin, _pp = [], None
    for _sd in (0, 1, 2):
        _d = _pd0.read_csv(f"/mnt/user-data/outputs/zombie-rl/models/learning_curve_lv{_lv}_seed{_sd}.csv"); _pp = _d.path_reward.iloc[0]; _fin.append(_d.eval_reward.tail(3).mean())
    _cv[_lv] = (sum(_fin) / 3, _pp)
fig("fig_curve.png", A4W, "Đường học của zombie PPO khi train từ đầu ở một màn (không qua curriculum), 3 seed. Mỗi điểm là return đánh giá trên 100 episode cố định; vạch là baseline path và chase.")
P(f"Return tăng rất nhanh trong khoảng 20 nghìn bước đầu rồi dao động quanh mốc baseline. Trung bình 3 mốc đánh giá cuối của 3 seed: "
  f"màn 1 đạt {f2(_cv[1][0])} so với path {f2(_cv[1][1])}; màn 2 đạt {f2(_cv[2][0])} so với {f2(_cv[2][1])}; màn 4 đạt {f2(_cv[4][0])} so với {f2(_cv[4][1])}. "
  "Màn 4 chỉ nhỉnh hơn path một chút và các đường học dao động mạnh, khớp với việc màn này là chỗ nhiều thuật toán dừng lại ở mục 5. "
  "Đây là số đo 100 episode của một lần chạy riêng, nên khác nhẹ so với bảng chính (300 episode).")

# ---------------- 2. Thuật toán ----------------
P("2. Bốn thuật toán được so sánh và thông số mô hình", H1)
P("2.1 Cách mỗi thuật toán cập nhật policy", H2)
P("Bốn thuật toán đại diện cho ba cách cập nhật chính sách khác nhau. Ba thuật toán đầu là <i>on-policy actor-critic</i>: dùng dữ liệu vừa thu "
  "bằng chính policy hiện tại, có mạng actor π<sub>θ</sub>(a|s) và critic V(s). DQN là <i>off-policy value-based</i>: học Q(s,a) từ bộ nhớ "
  "replay và chọn action có Q lớn nhất. Hàm lợi thế (advantage) Â<sub>t</sub> ước lượng mức một action tốt hơn trung bình, tính bằng GAE.")
P("<b>A2C</b> tối đa hóa trực tiếp:  L(θ) = Ê<sub>t</sub>[ log π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) · Â<sub>t</sub> ]  — không giới hạn bước nhảy của policy.", EQ)
P("<b>TRPO</b>:  max Ê<sub>t</sub>[ r<sub>t</sub>(θ) Â<sub>t</sub> ]  với ràng buộc  KL(π<sub>cũ</sub> ‖ π<sub>θ</sub>) ≤ δ = 0,01,  r<sub>t</sub> = π<sub>θ</sub>/π<sub>cũ</sub>", EQ)
P("<b>PPO</b>:  L = Ê<sub>t</sub>[ min( r<sub>t</sub>Â<sub>t</sub>, clip(r<sub>t</sub>, 1−ε, 1+ε) Â<sub>t</sub> ) ],  ε = 0,2", EQ)
P("<b>DQN</b>:  L = ( r + γ max<sub>a′</sub> Q<sub>θ⁻</sub>(s′, a′) − Q<sub>θ</sub>(s, a) )<super>2</super>,  θ⁻ là mạng target cập nhật mỗi 5000 bước", EQ)
T([["", "PPO", "TRPO", "A2C", "DQN"],
   ["Loại", "On-policy, clip", "On-policy, trust region cứng", "On-policy, không giới hạn", "Off-policy, Q-learning"],
   ["Số env song song", "8", "8", "8", "1"],
   ["Rollout / batch", "512 × 8 bước, batch 256, 10 epoch", "512 × 8 bước, CG + line search", "5 × 8 bước, 1 cập nhật", "replay 200k, batch 64, train mỗi 4 bước"],
   ["Khám phá", "policy ngẫu nhiên + entropy 0,01", "policy ngẫu nhiên", "policy ngẫu nhiên + entropy 0,01", "ε-greedy 1→0,05 trong 20% bước"],
   ["Ưu điểm", "ổn định, tận dụng dữ liệu nhiều epoch", "bảo đảm mỗi bước không phá policy", "đơn giản, nhanh", "tái dùng dữ liệu cũ, không cần nhiều env"],
   ["Nhược điểm", "nhiều siêu tham số", "chậm hơn mỗi cập nhật", "dao động lớn giữa các seed", "Q dễ bị ước lượng quá cao, cần chỉnh lịch khám phá"]],
  [26 * mm] + [(A4W - 26 * mm) / 4] * 4)
P("Mọi thuật toán dùng mạng MLP 64×64 mặc định của Stable-Baselines3 (sb3-contrib cho TRPO), γ = 0,99, không tinh chỉnh siêu tham số riêng (riêng DQN có thêm một lần chỉnh ở mục 5.5). "
  "Đây là so sánh <i>với cấu hình mặc định hợp lý</i>, không phải so sánh mỗi thuật toán ở cấu hình tốt nhất của nó.", NOTE)


P("2.2 Thông số chi tiết của từng mô hình", H2)
P("Mọi giá trị dưới đây được đọc trực tiếp từ các file model đã lưu bằng <font name='TNM'>make_hparams.py</font> (không gõ tay), "
  "nên khớp với mô hình thật đã dùng để đo kết quả. Dấu “–” nghĩa là thuật toán không có tham số đó.")
_p, _t, _a = HP["ppo"], HP["trpo"], HP["a2c"]
P("<b>Nhóm on-policy actor-critic</b> (huấn luyện cùng lúc 8 môi trường song song):", ps("lbl", fontName="TNB", spaceAfter=3))
T([["Tham số", "PPO", "TRPO", "A2C"],
   ["Thư viện", "Stable-Baselines3 2.9.0", "sb3-contrib 2.9.0", "Stable-Baselines3 2.9.0"],
   ["Mạng", "actor 29→64→64→18; critic 29→64→64→1", "như PPO", "như PPO"],
   ["Hàm kích hoạt", _p["activation"], _t["activation"], _a["activation"]],
   ["Tổng tham số học được", f"{_p['params_total']:,}".replace(",", " "), f"{_t['params_total']:,}".replace(",", " "), f"{_a['params_total']:,}".replace(",", " ")],
   ["Optimizer", _p["optimizer"], _t["optimizer"] + " (critic); actor cập nhật bằng CG + line search", _a["optimizer"]],
   ["Learning rate", lrf(_p["learning_rate"]), lrf(_t["learning_rate"]) + " (chỉ critic)", lrf(_a["learning_rate"])],
   ["γ (chiết khấu)", gv(_p["gamma"]), gv(_t["gamma"]), gv(_a["gamma"])],
   ["GAE λ", gv(_p["gae_lambda"]), gv(_t["gae_lambda"]), gv(_a["gae_lambda"])],
   ["Số env song song", gv(_p["n_envs"]), gv(_t["n_envs"]), gv(_a["n_envs"])],
   ["Bước mỗi env / rollout", gv(_p["n_steps"]), gv(_t["n_steps"]), gv(_a["n_steps"])],
   ["Batch size", gv(_p["batch_size"]), gv(_t["batch_size"]), f"{_a['n_steps'] * _a['n_envs']} (cả rollout)"],
   ["Epoch mỗi rollout", gv(_p["n_epochs"]), "–", "1"],
   ["Giới hạn cập nhật", f"clip ε = {gv(_p['clip_range'])}", f"KL ≤ {gv(_t['target_kl'])}", "không có"],
   ["Hệ số entropy", gv(_p["ent_coef"]), "–", gv(_a["ent_coef"])],
   ["Hệ số value loss", gv(_p["vf_coef"]), "–", gv(_a["vf_coef"])],
   ["Giới hạn norm gradient", gv(_p["max_grad_norm"]), "–", gv(_a["max_grad_norm"])],
   ["Chuẩn hóa advantage", gv(_p["normalize_advantage"]), gv(_t["normalize_advantage"]), gv(_a["normalize_advantage"])],
   ["Tham số riêng TRPO", "–", f"CG tối đa {_t['cg_max_steps']} vòng, damping {gv(_t['cg_damping'])}; line search co {gv(_t['line_search_shrinking_factor'])}, tối đa {_t['line_search_max_iter']} lần; critic cập nhật {_t['n_critic_updates']} lần/rollout", "–"]],
  [34 * mm] + [(A4W - 34 * mm) / 3] * 3)
_d, _g = HP["dqn"], HP["dqn_tgt"]
P("<b>DQN</b> (1 môi trường, off-policy) ở cấu hình mặc định và cấu hình đã chỉnh ở mục 5.5:", ps("lbl2", fontName="TNB", spaceAfter=3))
T([["Tham số", "DQN mặc định", "DQN đã chỉnh (từ màn 2 trở đi)"],
   ["Mạng Q", f"29→64→64→18, {_d['activation']}, {_d['params_qnet']:,} tham số".replace(",", " ") + " mỗi mạng (Q và target)", "như mặc định"],
   ["Optimizer · learning rate", f"{_d['optimizer']} · {lrf(_d['learning_rate'])}", "như mặc định"],
   ["γ", gv(_d["gamma"]), gv(_g["gamma"])],
   ["Replay buffer", f"{_d['buffer_size'] // 1000}k mẫu, rỗng mỗi khi sang màn mới", "giữ buffer của màn trước"],
   ["Bắt đầu học sau", f"{_d['learning_starts']} bước", f"{_g['learning_starts']} bước (buffer đã có dữ liệu)"],
   ["Batch · chu kỳ train", f"{_d['batch_size']} · mỗi {_d['train_freq']} bước, {_d['gradient_steps']} gradient step", "như mặc định"],
   ["Cập nhật mạng target", f"mỗi {_d['target_update_interval']} bước (τ = {gv(_d['tau'])}, chép cứng)", f"mỗi {_g['target_update_interval']} bước"],
   ["ε-greedy", f"{gv(_d['exploration_initial_eps'])} → {gv(_d['exploration_final_eps'])} trong {_d['exploration_fraction'] * 100:.0f}% số bước, chạy lại ở mỗi màn",
    f"{gv(_g['exploration_initial_eps'])} → {gv(_g['exploration_final_eps'])} trong {_g['exploration_fraction'] * 100:.0f}% số bước, khi sang màn mới"],
   ["Giới hạn norm gradient", gv(_d["max_grad_norm"]), gv(_g["max_grad_norm"])]],
  [38 * mm, (A4W - 38 * mm) / 2, (A4W - 38 * mm) / 2])
P(f"<b>Chung cho cả bốn thuật toán</b>: observation 29 chiều, 18 action rời rạc, reward như mục 1.3, curriculum 5 màn "
  f"({' / '.join(str(l['train_steps'] // 1000) + 'k' for l in CFG.LEVELS)} bước, tổng {sum(l['train_steps'] for l in CFG.LEVELS) / 1e6:.1f} triệu bước mỗi lượt), "
  f"{max(NS.values())} seed mỗi thuật toán, đánh giá {NEP} episode sau mỗi màn. Mạng của PPO, TRPO và A2C đều có {_p['params_total']:,} tham số học được (actor và critic riêng); "
  "DQN chỉ có một mạng Q nên nhỏ hơn.".replace(",", " "), NOTE)
P("2.3 Agent người chơi (RL)", H2)
_pl = HP["player"]
P("Agent người chơi chơi trên đúng luật game (cùng bản đồ, đợt zombie, súng, nạp đạn, 60 khung hình/giây) và chỉ được dùng các nút người chơi có. "
  "Nó được tập đều với ba loại zombie (PPO, path, chase mỗi loại một phần ba) trên cả 5 màn để không chỉ học cách khai thác riêng zombie PPO; "
  "zombie không học thêm trong lúc này.")
T([["Thành phần", "Giá trị"],
   ["Thuật toán", f"PPO (Stable-Baselines3), {_pl['optimizer']}, hàm kích hoạt {_pl['activation']}"],
   ["Mạng", f"actor và critic riêng, mỗi mạng 113→256→256→đầu ra; tổng {_pl['params_total']:,} tham số".replace(",", " ")],
   ["Observation", f"{_pl['obs_dim']} chiều"],
   ["Action", "MultiDiscrete [3, 3, 9, 2, 2, 2]: tiến/lùi/đứng, trái/phải/đứng, 9 mức tốc độ quay, bắn, nạp đạn, đổi súng"],
   ["Siêu tham số", f"lr {lrf(_pl['learning_rate'])}, γ {gv(_pl['gamma'])}, λ {gv(_pl['gae_lambda'])}, clip {gv(_pl['clip_range'])}, entropy {gv(_pl['ent_coef'])}, "
                    f"{_pl['n_envs']} env, {_pl['n_steps']} bước/env, batch {_pl['batch_size']}, {_pl['n_epochs']} epoch"],
   ["Nhịp quyết định", "0,1 giây (như phản xạ người), quay có giới hạn tốc độ như dùng chuột"],
   ["Huấn luyện thêm", f"{RETRAIN} bước với boss 1300 máu (mục 5.4)"]],
  [32 * mm, A4W - 32 * mm])

# ---------------- 3. Boss ----------------
P("3. Thiết kế lại trận boss", H1)
P("3.1 Hiện trạng ban đầu: boss thắng nhờ chỉ số", H2)
P("Đo sơ bộ (60 episode) với boss cũ: 1500 máu, damage 35, có cú lao tới do code viết sẵn. Boss <b>chỉ đi thẳng</b> (chase) thắng 90%, "
  "boss đi theo đường (path) thắng 92%, boss PPO cũ thắng 95%, và không lần nào boss bị hạ. Boss đứng trong đường ngắm 75–83% thời gian mà vẫn thắng, "
  "nghĩa là mô hình không cần né: chênh lệch giữa có RL và không có RL chỉ 3–5 điểm phần trăm.")
P("3.2 Nguyên nhân sâu: đối thủ ngắm như aimbot", H2)
P("Chỉ giảm máu boss là chưa đủ. Người chơi giả ở màn boss có độ trễ phản xạ 1 bước (0,1 giây): nó bắn vào vị trí gần như hiện tại, nên né không làm nó trượt. "
  "Để kiểm chứng, nhóm viết một boss <b>zig-zag</b>: đi theo đường nhưng lệch ±30° xen kẽ mỗi 0,4 giây. Nếu né có giá trị, boss zig-zag phải thắng boss đi thẳng.")
T([["Phản xạ của bot (màn boss)", "Boss đi thẳng (path)", "Boss zig-zag ±30°", "Kết luận"],
   ["0,1 giây (cũ)", "thắng 40%", "thắng 33%, né 12% số phát", "Né <b>thiệt hơn</b> lao thẳng"],
   ["0,3 giây (mới, gần người thật)", "thắng 42%", "thắng 52%, né 24% số phát", "Né <b>có lợi</b>"]],
  [44 * mm, 36 * mm, 46 * mm, A4W - 126 * mm])
P("Đo sơ bộ 60 episode với boss 700 máu và người chơi giả bản cũ (xem 3.4). Kết quả cho thấy với bot phản xạ 0,1 giây, chính sách tối ưu là lao vào và dựa vào damage. "
  "Thuật toán RL đã học đúng điều đó, đây là lý do người chơi cảm thấy boss \"mạnh nhờ damage\".", CAP)
P("3.3 Lần chỉnh thứ nhất", H2)
B(["Máu boss 1500 → 700, damage 35 → 30 (người chơi 100 máu chết sau 4 đòn thay vì 3).",
   "Bỏ cú lao tới viết tay (<font name='TNM'>scripted_charge=False</font>): mọi bước di chuyển của boss do mô hình quyết định.",
   "Bot ở màn boss phản xạ 3 bước (0,3 giây) thay vì 1 bước.",
   "Thêm chỉ số <b>dodge</b>: tỉ lệ phát bắn của bot bị trượt, và baseline <b>zigzag</b> làm mốc \"né viết tay\"."])
P("3.4 Kiểm tra trong game thật: khoảng cách giữa sân tập và game", H2)
P("Boss 700 máu cho số liệu đẹp trong sân tập, nhưng khi đưa vào game thật (<font name='TNM'>evaluate_match.py</font>, 40 trận mỗi cặp) "
  "thì <b>boss chưa thắng được trận nào trong 40 trận</b>, kể cả boss PPO, trước cả người chơi viết tay lẫn agent người chơi RL; boss bị hạ sau 6–10 giây. "
  "Đây là hiện tượng <i>sim-to-game gap</i>: môi trường huấn luyện khác môi trường thật. Hai điểm khác được tìm ra:")
B(["<b>Tốc độ bắn.</b> Trong sân tập, người chơi giả bắn tối đa một phát mỗi bước quyết định (0,1 giây), tức M4A1 chỉ đạt khoảng 4 phát/giây. "
   "Trong game M4A1 bắn mỗi 0,12 giây, khoảng 8,3 phát/giây; một băng 35 viên gây 770 damage, nhiều hơn 700 máu của boss.",
   "<b>Loại súng.</b> Sân tập chọn ngẫu nhiên giữa súng lục và M4A1, còn ở màn boss của game người chơi luôn có M4A1. "
   "Đo 200 episode với boss 700 máu đi thẳng: gặp bot cầm súng lục boss thắng 73%, gặp bot cầm M4A1 chỉ thắng 3%. Con số ~40% ở bảng trên là trung bình của hai trường hợp rất khác nhau."])
P("<b>Lần chỉnh thứ hai.</b> Ở màn boss, người chơi giả luôn cầm M4A1 và bắn đúng tốc độ của game. Máu boss được chọn lại bằng cách quét trong sân tập (100 episode mỗi mức):")
T([["Máu boss", "700", "1000", "1200", "<b>1300</b>", "1400", "≥1600"],
   ["Boss đi thẳng (path) thắng", "0%", "6%", "12%", "<b>30%</b>", "58%", "≥88%"],
   ["Boss zig-zag thắng", "0%", "9%", "46%", "<b>70%</b>", "84%", "–"]],
  [50 * mm] + [(A4W - 50 * mm) / 6] * 6)
P(f"Chọn 1300 máu vì đây là mức né có giá trị lớn nhất (zig-zag thắng gấp hơn hai lần đi thẳng) và boss không biết né chỉ thắng phần nhỏ số trận. "
  f"Toàn bộ 4 thuật toán × {max(NS.values())} seed được train lại màn 5 với cấu hình này; số liệu màn 5 trong báo cáo là của cấu hình mới. "
  f"Đo lại bằng {NEP} episode: boss đi thẳng thắng {pct(BS['path_5']['win'])}, boss zig-zag thắng {pct(BS['zigzag_5']['win'])}.", CAP)

# ---------------- 4. Giao thức ----------------
P("4. Giao thức thí nghiệm", H1)
B([f"<b>Huấn luyện</b>: mỗi thuật toán × {max(NS.values())} seed (0–{max(NS.values()) - 1}), đi qua 5 màn; màn sau học tiếp từ mô hình màn trước của chính thuật toán đó.",
   f"<b>Đánh giá</b>: sau mỗi màn, {NEP} episode với chính sách tất định (chọn action xác suất cao nhất), không ôn màn cũ. "
   f"Episode thứ i luôn dùng seed môi trường 10000+i cho mọi chính sách, nên so sánh được <i>theo cặp</i>: cùng vị trí xuất phát, cùng loại súng, cùng hành vi ngẫu nhiên của bot.",
   "<b>Qua màn</b>: return trung bình ≥ return của baseline path trên cùng các episode. <b>Đạt tới màn</b> = số màn qua liên tiếp từ màn 1 (dừng ở màn đầu tiên chưa qua, như người chơi). Thuật toán vẫn được train tiếp các màn sau để có đủ số liệu.",
   f"<b>Thống kê</b>: khoảng tin cậy 95% của return tính theo phân phối t trên {max(NS.values())} giá trị trung bình của từng seed; tỉ lệ thắng dùng khoảng Wilson "
   f"trên {max(NS.values()) * NEP} episode gộp; chênh lệch so với path tính theo cặp episode."])

# ---------------- 5. Kết quả ----------------
P("5. Kết quả", H1)
P("5.1 Thuật toán đi được bao xa", H2)
rows = [["Thuật toán"] + [f"seed {r['seed']}" for r in R["ppo"]] + ["Qua đủ 5 màn", "Đạt tới màn (TB)"]]
for a in ALG:
    rows.append([NAME[a]] + [str(r["reach"]) for r in R[a]] + [f"{full(a)}/{NS[a]}", f"{reach_mean(a):.1f}"])
T(rows, [24 * mm] + [16 * mm] * len(R["ppo"]) + [26 * mm, 28 * mm], hl_rows=[1 + ALG.index(best)])
fig("fig_pass.png", 135 * mm, f"Hình 1. Số seed qua từng màn. Ô càng nhạt là màn càng ít seed vượt được baseline path.")

def weakest(a):
    return min(range(1, 6), key=lambda l: L[f"{a}_{l}"]["pass_rate"])
wa, wd = weakest("a2c"), weakest("dqn")
npass = lambda a, l: round(L[f"{a}_{l}"]["pass_rate"] * NS[a])
P(f"PPO và TRPO đạt tới màn 5 ở mọi seed. A2C dao động mạnh: số màn đạt được theo từng seed là {reach_str('a2c')}, "
  f"tức cùng một thuật toán và cùng cấu hình nhưng có seed qua hết, có seed trượt ngay màn 1. "
  f"DQN dừng sớm nhất ở cấu hình mặc định ({reach_str('dqn')}; mục 5.5 cho thấy có thể cải thiện); màn thử thách nhất là màn {wd}, chưa seed nào vượt path.")

P("5.2 Return từng màn", H2)
fig("fig_return.png", A4W, f"Hình 2. Return trung bình (vạch: CI 95% theo seed). Cột xám là baseline path. {NEP} episode mỗi seed.")
rows = [["Màn", "path"] + [NAME[a] for a in ALG]]
for l in range(1, 6):
    rows.append([str(l), f2(BS[f"path_{l}"]["ret"])] +
                [f"{f2(L[f'{a}_{l}']['ret'])}  <font color='#5a4a4c' size='7'>[{f2(L[f'{a}_{l}']['ret_lo'])}; {f2(L[f'{a}_{l}']['ret_hi'])}]</font>" for a in ALG])
T(rows, [14 * mm, 18 * mm] + [(A4W - 32 * mm) / 4] * 4)
fig("fig_diff.png", A4W, "Hình 3. Return của thuật toán trừ return của path trên cùng episode. Khoảng tin cậy nằm hẳn trên 0 nghĩa là vượt path có ý nghĩa.")
sig = [(a, l) for a in ALG for l in range(1, 6) if L[f"{a}_{l}"]["diff_lo"] > 0]
nonsig = [(a, l) for a in ALG for l in range(1, 6) if L[f"{a}_{l}"]["diff_lo"] <= 0 <= L[f"{a}_{l}"]["diff_hi"]]
worse = [(a, l) for a in ALG for l in range(1, 6) if L[f"{a}_{l}"]["diff_hi"] < 0]
P(f"Có {len(sig)}/20 cặp (thuật toán, màn) vượt path <b>có ý nghĩa</b> (CI 95% của chênh lệch nằm trên 0). "
  + (f"Chưa phân biệt được với path: {', '.join(f'{NAME[a]} màn {l}' for a, l in nonsig)}. " if nonsig else "")
  + (f"<b>Thấp hơn path có ý nghĩa</b>: {', '.join(f'{NAME[a]} màn {l}' for a, l in worse)}. " if worse else "")
  + "PPO và TRPO vượt path có ý nghĩa ở cả 5 màn.")

P("5.3 Màn boss: chiến thắng có đến từ kỹ năng không?", H2)
fig("fig_boss.png", A4W, f"Hình 4. Màn boss. Bên trái vạch đứt: boss viết tay. Bên phải: boss do RL điều khiển. Né = tỉ lệ phát bắn của bot bị trượt.")
rows = [["Boss", "Thắng (CI 95%)", "Né đạn", "Thời gian bị ngắm", "Máu boss mất"]]
for p in ["chase", "path", "zigzag"]:
    b = BS[f"{p}_5"]; rows.append([p + " (viết tay)", f"{pct(b['win'])} [{pct(b['win_lo'])}–{pct(b['win_hi'])}]", pct(b["dodge"]), pct(b["aimed"]), pct(b["hp_lost"])])
for a in ALG:
    b = L[f"{a}_5"]; rows.append([NAME[a], f"{pct(b['win'])} [{pct(b['win_lo'])}–{pct(b['win_hi'])}]", pct(b["dodge"]), pct(b["aimed"]), pct(b["hp_lost"])])
T(rows, [34 * mm, 40 * mm, 26 * mm, 34 * mm, A4W - 134 * mm])
P(f"Mọi boss RL đều thắng nhiều hơn boss đi thẳng ({pct(BS['path_5']['win'])}) và né nhiều hơn hẳn ({pct(BS['path_5']['dodge'])}). "
  f"Vì chỉ số boss giống nhau cho mọi chính sách, phần chênh lệch về tỉ lệ thắng đến từ cách di chuyển: né đạn và chọn thời điểm ra đòn. "
  f"Boss zig-zag viết tay né {pct(BS['zigzag_5']['dodge'])} và thắng {pct(BS['zigzag_5']['win'])}; boss PPO né ít hơn ({pct(L['ppo_5']['dodge'])}) "
  f"nhưng thắng nhiều hơn ({pct(L['ppo_5']['win'])}). Điều này cho thấy mô hình RL không chỉ né nhiều mà còn học được lúc nào nên né và lúc nào nên lao vào.")

P("5.4 Kiểm tra trong game thật", H2)
P(f"Các mục trên đo trong sân tập. Mục này cho zombie vào <b>game thật</b> (cùng mã nguồn với bản người chơi dùng), đấu với hai người chơi tự động: "
  f"agent người chơi RL (PPO, mạng 256×256, đã train tiếp {RETRAIN} bước với boss mới) và người chơi viết tay. Mỗi cặp {MN} trận; "
  f"zombie PPO là mô hình seed 0. Số trong bảng là số trận <b>zombie thắng</b> trên {MN}.")
rows = [["Màn", "Agent RL: PPO", "path", "chase", "Viết tay: PPO", "path", "chase"]]
for l in [1, 2, 3, 4, 5, 0]:
    rows.append(["Cả game (1→5)" if l == 0 else str(l)] + [str(mz(p, z, l)) for p in ("agent", "scripted") for z in ("ppo", "path", "chase")])
T(rows, [30 * mm] + [(A4W - 30 * mm) / 6] * 6, hl_rows=[5])
P(f"Hàng \"Cả game\": người chơi đi từ màn 1 đến màn 5 với một mạng. Ở màn boss, trước agent RL: boss PPO thắng {mz('agent','ppo',5)}/{MN}, "
  f"boss đi thẳng {mz('agent','path',5)}/{MN}, boss chase {mz('agent','chase',5)}/{MN}. "
  f"Trước khi agent được train tiếp với boss mới, các con số này là {mzb('agent','ppo',5)}, {mzb('agent','path',5)} và {mzb('agent','chase',5)}: "
  f"sau {RETRAIN} bước train tiếp, boss PPO vẫn giữ được thế cân bằng với agent (khác biệt nằm trong sai số của {MN} trận).", CAP)
P(f"Hai điều rút ra. <b>Thứ nhất</b>, trước agent RL, khác biệt giữa zombie PPO và zombie viết tay xuất hiện cả trong game thật: tính cả game, "
  f"zombie PPO thắng {mz('agent','ppo',0)}/{MN} lượt, path {mz('agent','path',0)}/{MN}, chase {mz('agent','chase',0)}/{MN}. "
  f"Tuy vậy phần hơn này chủ yếu đến từ màn boss: ở màn 1–4 zombie PPO và zombie path thắng số trận gần như nhau "
  f"({sum(mz('agent','ppo',l) for l in range(1,5))} so với {sum(mz('agent','path',l) for l in range(1,5))} trên {4*MN} trận), "
  f"tức ở các màn này vẫn còn dư địa để chuyển lợi thế trong sân tập thành lợi thế trong game. "
  f"<b>Thứ hai</b>, trước người chơi viết tay, cả ba loại boss có kết quả {mz('scripted','ppo',5)}, {mz('scripted','path',5)} và {mz('scripted','chase',5)} trên {MN}. "
  f"Con số này nên được đọc thận trọng: ở màn boss vị trí xuất phát cố định và người chơi viết tay không có yếu tố ngẫu nhiên, nên {MN} trận gần như là một trận lặp lại "
  f"(ví dụ các trận với boss path đều kết thúc ở cùng một thời điểm). Người chơi viết tay cũng không có độ trễ phản xạ như người chơi giả trong sân tập.")

fig("fig_boss_shots.png", A4W, "Trận boss trong game thật do agent người chơi RL điều khiển. Trái: boss pha 1 khi người chơi đang nạp đạn. Phải: pha 2 khi boss còn dưới 50% máu, nhanh hơn 1,3 lần (chữ BOSS - PHA 2 trên thanh máu).")
P("5.5 Chỉnh DQN: replay buffer, ε và target update", H2)
P("Ở cấu hình mặc định, mỗi khi sang màn mới DQN gần như học lại từ đầu, dù trọng số mạng được giữ. Có ba lý do nằm trong cách Stable-Baselines3 nạp lại mô hình: "
  "<b>(1)</b> lịch ε-greedy chạy lại từ 1,0, nên 20% số bước đầu của màn (50–60 nghìn bước) zombie đi gần như ngẫu nhiên; "
  "<b>(2)</b> replay buffer bị bỏ trống, phải gom lại 5000 bước mới bắt đầu học; "
  "<b>(3)</b> mạng target chỉ được chép lại sau mỗi 5000 bước, tức khoảng 50 lần trong một màn 250 nghìn bước, nên giá trị Q lan truyền chậm. "
  "Nhóm thử bốn cấu hình, mỗi cấu hình thêm <b>một</b> thay đổi so với cấu hình trước để thấy tác dụng riêng của từng thay đổi.")
TUNE = [("dqn", "Mặc định (như bảng chính)"), ("dqn_eps", "Sang màn mới, ε bắt đầu từ 0,3, giảm về 0,05 trong 10% số bước"),
        ("dqn_buf", "Như trên + mang replay buffer của màn trước sang (học ngay, không chờ 5000 bước)"),
        ("dqn_tgt", "Như trên + chép mạng target mỗi 1000 bước thay vì 5000")]
NT = _TU.seed.nunique()
rows = [["Cấu hình", "Thay đổi", "Màn 1", "Màn 2", "Màn 3", "Màn 4", "Đạt tới màn (TB)"]]
for a, d in TUNE:
    g = _TU[_TU.algo == a]
    cells = []
    for l in range(1, 5):
        h = g[g.level == l]
        cells.append(f"{h.mean_reward.mean():.2f} <font color='#5a4a4c' size='7'>({int(h.passed.sum())}/{len(h)})</font>")
    rr = _reach(g)
    rows.append([f"<font name='TNM'>{a}</font>", d] + cells + [f"{sum(rr.values()) / len(rr):.1f}"])
rows.append(["path", "baseline"] + [f2(BS[f"path_{l}"]["ret"]) for l in range(1, 5)] + ["–"])
T(rows, [20 * mm, 52 * mm] + [(A4W - 102 * mm) / 4] * 4 + [30 * mm], hl_rows=[4])
P(f"Thử trên {NT} seed mới ({', '.join(str(s) for s in sorted(_TU.seed.unique()))}), màn 1–4, {NEP} episode mỗi ô. Số trong ngoặc là số seed qua màn. "
  "Ba cấu hình đầu giống nhau ở màn 1 (chỉ khác cách học tiếp) nên dùng chung một mô hình màn 1. "
  "Seed để thử tách khỏi seed 0–4 của bảng chính, để việc chọn cấu hình không dựa trên chính các seed dùng để báo cáo.", CAP)
P("Mỗi thay đổi đều làm return ở màn 4, màn khó nhất của DQN, tăng lên. Mang replay buffer sang có tác dụng lớn nhất ở màn 2 và 4; "
  "cập nhật target thường hơn giúp ở cả màn 1. Vì chỉ có 3 seed, mức tăng của từng thay đổi riêng lẻ chỉ nên đọc là xu hướng. "
  "Cấu hình cuối (<font name='TNM'>dqn_tgt</font>) được train đủ 5 màn trên seed 0–4 để so với DQN mặc định theo cặp seed:")
rD, rT = _reach(_DEF), _reach(_TD)
SE = sorted(rT)
rows = [["", ] + [f"seed {s}" for s in SE] + ["Qua đủ 5 màn", "Đạt tới màn (TB)"]]
rows.append(["DQN mặc định"] + [str(rD[s]) for s in SE] + [f"{sum(v == 5 for v in rD.values())}/{len(rD)}", f"{sum(rD.values()) / len(rD):.1f}"])
rows.append(["DQN đã chỉnh"] + [str(rT[s]) for s in SE] + [f"{sum(v == 5 for v in rT.values())}/{len(rT)}", f"{sum(rT.values()) / len(rT):.1f}"])
T(rows, [30 * mm] + [16 * mm] * len(SE) + [26 * mm, A4W - 56 * mm - 16 * mm * len(SE)], hl_rows=[2])
rows = [["Màn", "path", "DQN mặc định", "DQN đã chỉnh", "Chênh lệch (CI 95%)", "Seed qua màn"]]
for l in range(1, 6):
    d = _DEF[_DEF.level == l].set_index("seed").mean_reward; t = _TD[_TD.level == l].set_index("seed").mean_reward
    m, lo, hi = _tci([t[s] - d[s] for s in SE])
    rows.append([str(l), f2(BS[f"path_{l}"]["ret"]), f2(d.mean()), f2(t.mean()), f"{m:+.2f} [{lo:+.2f}; {hi:+.2f}]",
                 f"{int(_DEF[_DEF.level == l].passed.sum())} → {int(_TD[_TD.level == l].passed.sum())}"])
T(rows, [14 * mm, 18 * mm, 30 * mm, 30 * mm, 44 * mm, A4W - 136 * mm])
_w5 = _TD[_TD.level == 5]
P(f"Return trung bình trên 5 seed. Chênh lệch = DQN đã chỉnh trừ DQN mặc định trên cùng seed; CI theo phân phối t với 5 cặp. "
  f"Ở màn boss, DQN đã chỉnh thắng {pct(_w5.catch.mean())} (mặc định {pct(L['dqn_5']['win'])}).", CAP)
P("<b>Lưu ý về tính công bằng.</b> Chỉ DQN được chỉnh: nhóm thử 4 cấu hình DQN (12 lượt chạy màn 1–4), còn PPO, TRPO và A2C giữ cấu hình mặc định. "
  "Vì vậy bảng ở mục 5.1–5.3 vẫn là phép so sánh chính và DQN đã chỉnh không được đưa vào xếp hạng. "
  "Kết quả ở mục này cho thấy DQN có tiềm năng tốt hơn bảng chính phản ánh: thứ hạng của DQN phần lớn phụ thuộc vào cách học tiếp giữa các màn, chứ không chỉ vào bản thân thuật toán. "
  "Nhóm không chỉnh A2C, PPO và TRPO vì giới hạn thời gian và máy chỉ có 2 nhân CPU; đây là hướng mở rộng tự nhiên cho các lần thử tiếp theo.", NOTE)

P("5.6 Chi phí huấn luyện", H2)
rows = [["", "PPO", "TRPO", "A2C", "DQN"],
        ["Thời gian train cả curriculum (phút, 1 nhân CPU)"] + [f"{S['time'][a]:.0f}" if S["time"][a] else "-" for a in ALG],
        ["Bước mỗi giây (đo lúc chọn thuật toán)", "≈2650", "≈2620", "≈3180", "≈1190"]]
T(rows, [70 * mm] + [(A4W - 70 * mm) / 4] * 4)
P("Hai thuật toán khác đã được cân nhắc và để dành cho phần mở rộng vì tốn nhiều thời gian trên máy 2 nhân: RecurrentPPO (có LSTM, ≈75 bước/giây, ước tính 7 giờ cho một lượt curriculum) "
  "và QR-DQN (≈110 bước/giây).", NOTE)

# ---------------- 6. Thảo luận ----------------
P("6. Thảo luận", H1)
P("<b>Vì sao PPO và TRPO ổn định?</b> Cả hai giới hạn mức thay đổi của policy sau mỗi lần cập nhật (PPO bằng clip, TRPO bằng ràng buộc KL). "
  "Trong curriculum, mỗi màn bắt đầu từ policy của màn trước; nếu bước cập nhật đầu tiên quá lớn trên bản đồ mới, kỹ năng cũ có thể bị phá. "
  "PPO còn dùng mỗi rollout 10 lần (10 epoch), có thể là lý do return của PPO nhỉnh hơn TRPO ở màn 3–4; tuy vậy khoảng tin cậy của hai thuật toán chồng nhau nên chưa thể nói PPO tốt hơn TRPO.")
P("<b>Vì sao A2C dao động?</b> A2C cập nhật sau mỗi 5 bước × 8 env, gradient có phương sai lớn và không có cơ chế chặn bước nhảy. "
  "Kết quả vì vậy phụ thuộc nhiều vào seed: có seed qua đủ 5 màn, có seed dừng ở các màn đầu.")
P(f"<b>Vì sao DQN dừng sớm nhưng boss lại mạnh?</b> Giả thuyết (i) đã được kiểm chứng ở mục 5.5; (ii) và (iii) chưa được kiểm chứng riêng: "
  "(i) mỗi màn DQN bắt đầu với bộ nhớ replay rỗng và ε-greedy khám phá lại từ 1,0, nên phần đầu mỗi màn gần như đi ngẫu nhiên; "
  "(ii) chỉ dùng 1 env nên dữ liệu ít đa dạng hơn; (iii) ở các màn nhiều loại zombie và bản đồ phức tạp, ước lượng max Q dễ bị lệch. "
  f"Trận boss có cấu trúc đơn giản hơn (một đối thủ, một mục tiêu) và có 800k bước, nên DQN có đủ thời gian để học (thắng {pct(L['dqn_5']['win'])}, cao nhất trong bốn thuật toán ở màn này).")
FL = S["flips"]
if FL:
    P(f"<b>Tiêu chí qua màn nhạy với nhiễu ở gần ngưỡng.</b> Ở màn 1–4, seed 0–1 được đánh giá hai lần: lần đầu 100 episode, lần này {NEP} episode "
      f"(cùng mô hình, khác số episode). Có {len(FL)}/{S['n_compared_100']} ô đổi kết luận: "
      + "; ".join(f"{NAME[f['algo']]} seed {f['seed']} màn {f['level']}: {f['r100']:.2f} so với {f['p100']:.2f} (100 ep) → "
                  f"{f['rnew']:.2f} so với {f['pnew']:.2f} ({NEP} ep, {'qua' if f['passed_new'] else 'trượt'})" for f in FL)
      + ". Cả hai lần chênh lệch đều nhỏ hơn độ dao động thống kê, nên với các ô này nên đọc là \"ngang path\" thay vì qua hoặc trượt. "
        "Đây là lý do báo cáo dùng nhiều seed và khoảng tin cậy thay vì một lần chạy.")
P("<b>Bài học từ boss.</b> Khi chính sách tối ưu của môi trường là \"lao vào ăn damage\", thuật toán RL tốt sẽ học đúng điều đó. "
  "Muốn kẻ địch có hành vi thông minh thì môi trường phải khiến hành vi thông minh có lợi: ở đây là cho đối thủ một độ trễ phản xạ hợp lý. "
  "Thiết kế môi trường và đối thủ ảnh hưởng tới hành vi học được nhiều hơn việc chọn thuật toán.")

# ---------------- 7. Hạn chế ----------------
P("7. Hạn chế và hướng phát triển", H1)
B([f"Chỉ {max(NS.values())} seed mỗi thuật toán. Kết luận về PPO so với TRPO (cùng qua đủ 5 màn) cần thêm seed để xếp hạng hai thuật toán này; đây là bước mở rộng đầu tiên.",
   "Chỉ DQN được chỉnh (4 cấu hình, 3 seed thử). PPO, TRPO và A2C dùng cấu hình mặc định, chưa chỉnh vì giới hạn thời gian và phần cứng; A2C có thể ổn định hơn với rollout dài hơn hoặc learning rate nhỏ hơn. "
   "Bảng xếp hạng chính vì vậy là so sánh <i>ở cấu hình mặc định</i>.",
   "Tiêu chí qua màn dựa trên return, một con số có cả phần reward shaping. Tỉ lệ thắng được báo cáo kèm để đối chiếu.",
   "Sân tập vẫn còn khác game: ở màn 3–4 người chơi giả cầm M4A1 vẫn bắn bằng nửa tốc độ game (không sửa để giữ nguyên kết quả màn 1–4), nên bước tiếp theo là đồng bộ phần này và train lại màn 3–4 để số liệu sân tập sát với game hơn.",
   f"Kiểm tra trong game thật (5.4) chỉ dùng zombie PPO seed 0 và hai người chơi tự động; đo với người chơi thật là bước tiếp theo để xác nhận boss 1300 máu vừa sức người chơi. "
   f"Agent người chơi có thể được train thêm (hiện là {RETRAIN} bước) để khai thác boss mới tốt hơn.",
   "Trận boss với người chơi viết tay gần như tất định, nên phép đo này dùng để kiểm tra hành vi, còn so sánh các loại boss dựa trên agent người chơi RL.",
   "Khoảng tin cậy Wilson của tỉ lệ thắng gộp mọi episode của mọi seed, nên bỏ qua dao động giữa các seed và hẹp hơn thực tế. Khoảng tin cậy của return (tính theo seed) đáng tin hơn.",
   "Số liệu boss cũ và thí nghiệm zig-zag ở mục 3.1–3.2 là đo sơ bộ 60 episode; bảng quét máu ở 3.4 là 100 episode mỗi mức; các phép thử 40 trận trong game ở 3.4 là đo nhanh."])

# ---------------- Phụ lục ----------------
story.append(KeepTogether([Paragraph("Phụ lục: tái lập thí nghiệm", H1), Paragraph("pip install sb3-contrib<br/>python compare_algos.py --baselines<br/>python compare_algos.py --algos ppo,a2c,trpo,dqn --seeds 0,1,2,3,4<br/>python compare_algos.py --report<br/>python compare_algos.py --tag tune --algos dqn,dqn_eps,dqn_buf,dqn_tgt --seeds 10,11,12 --max-level 4<br/>python compare_algos.py --tag tuned --algos dqn_tgt --seeds 0,1,2,3,4<br/>python game.py --level 3          # chơi và xem AI zombie (TAB: minimap)<br/>python game.py --level 5 --watch-agent   # xem agent người chơi RL đấu boss<br/>python make_hparams.py ; python make_figs.py   # thông số và hình trong báo cáo<br/>python game.py --zombie-algo dqn",
                       ps("code", fontName="TNM", fontSize=9.4, leading=12.5, backColor=LIGHT, borderPadding=6, spaceAfter=8))]))
P("Lệnh cuối: chơi thử với zombie của thuật toán khác (a2c, trpo, dqn). Số liệu nằm trong <font name='TNM'>models/</font>: "
  "<font name='TNM'>algo_compare.csv</font>, <font name='TNM'>algo_episodes.csv</font> (từng episode), <font name='TNM'>algo_baselines.csv</font>; mô hình trong <font name='TNM'>models/algos/</font>.", ps("tail", alignment=0))


def on_page(c, doc):
    c.saveState(); c.setFont("TN", 9); c.setFillColor(GREY)
    c.drawString(20 * mm, 12 * mm, "Zombie RL · So sánh thuật toán RL · REL")
    c.drawRightString(A4[0] - 20 * mm, 12 * mm, str(doc.page))
    c.setStrokeColor(LINE); c.setLineWidth(0.4); c.line(20 * mm, 16 * mm, A4[0] - 20 * mm, 16 * mm); c.restoreState()

def on_cover(c, doc):
    c.saveState(); c.setFillColor(INK); c.rect(0, A4[1] - 18 * mm, A4[0], 18 * mm, fill=1, stroke=0)
    c.setFillColor(RED); c.rect(0, A4[1] - 18 * mm, 6 * mm, 18 * mm, fill=1, stroke=0); c.restoreState()

doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=22 * mm,
                      title="So sánh PPO, TRPO, A2C, DQN - Zombie RL", author="Nguyễn Võ Trung Hưng, Ngô Phúc Nguyên, Lê Như Đạt")
fr = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate("cover", [fr], onPage=on_cover), PageTemplate("body", [fr], onPage=on_page)])
doc.build(story)
print("ok", OUT)
