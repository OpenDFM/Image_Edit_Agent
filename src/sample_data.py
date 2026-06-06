# -*- coding: utf-8 -*-
"""
指令生成器（第二个函数）——按分数段批量生成“参数组合 → 自然语言修图指令”

核心流程：
1) 抽样“10分标准模板”参数（偏好 3–5 个常用工具；每个工具取一个均匀随机值，值域由档位决定）
2) 以模板为基准生成若干“加噪”版本，并按评分规则打分（越偏离越扣分，替代关系减免）
3) 调用 LLM 将每个参数组合转换为自然语言修图指令
4) 对 -10 分的样本，从给定 JSONL 随机抽取负样本指令

输入：
- num_groups: 需要生成的“组”数量（每组都会先采一个 10 分模板，再派生出各分数段若干条）
- per_group_counts: dict[int, int]，键为目标分数（包含 10、8、...、-2、-10 等），值为该分数要生成的指令条数
- negative_jsonl_path: 包含负例指令（文本字段名 "instruction" 或整行即文本）的 JSONL 文件路径；用于填充 -10 分
- seed: 随机种子
- use_llm: 是否用 LLM 生成指令文本；False 时走规则模板（便于离线调试）
- model_fn: 文本生成函数，签名 fn(system_prompt:str, user_prompt:str) -> str
           若你已有 call_vlm_api，可封一层不传 image_list；或直接换成 call_llm_api

输出：
- List[Dict]，每组一个 dict，含 "template_params" 与 "items"（按分数段的样本列表）
- 同时落盘一个 JSONL（可自行添加保存逻辑）
"""

import json, random, math, threading
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

try:
    from llm_caller import generate_amateur_instruction
    import config, prompts
except Exception:
    config = None

# 你的编辑器支持的一组参数（与第一阶段一致）
ALLOWED = [
    "exposure","brightness","contrast","natural_contrast",
    "highlights","shadows","whites","blacks",
    "saturation","vibrance","temperature","tint",
    "sharpness","vignette","fade","grain"
]

# 数值档位（你给出的四档）
ADJUSTMENT_LEVELS = {
    'light':   (-30, -15,  15,  30),
    'medium':  (-60, -45,  45,  60),
    'heavy':   (-90, -75,  75,  90),
    'extreme': (-100,-100, 100, 100),
}
LEVEL_ORDER = ["light", "medium", "heavy", "extreme"]

# 工具出现的经验先验（和常见工作流一致：先光线对比，再色彩，再风格化）：
# 归一化后用作无放回加权采样权重
TOOL_PRIOR = {
    "exposure": 0.9,  # 全局明暗（常用）  Adobe: Light & Contrast 基础滑块之一
    "brightness": 0.9,  # 与曝光有重叠，但偏中间调
    "contrast": 0.7,
    "natural_contrast": 0.5,
    "highlights": 0.6,
    "shadows": 0.6,
    "whites": 0.3,
    "blacks": 0.3,
    "saturation":       0.7,
    "vibrance":         0.7,  # 与 saturation 互补
    "temperature":      0.6,
    "tint":             0.4,
    "sharpness":        0.5,
    "vignette":         0.25,
    "fade": 0.15,
    "grain": 0.1,
}

# 若某些工具更常见单向偏置（用于随机值“正/负”方向的先验；1.0=强烈偏正，-1.0=强烈偏负，0=无偏）
DIRECTION_BIAS = {
    "vibrance":  +0.6,    # 常见提升色彩活力
    "saturation":+0.2,
    "highlights":-0.3,    # 恢复细节时常见“压高光”
    "shadows":  +0.3,     # 拉起暗部
    "whites":   +0.2,
    "blacks":   -0.2,
    "sharpness": +0.2,
    "vignette": -0.5,     # 常见暗角
    "fade": +0.5,  # 常见褪色
    "grain": +0.5,  # 常见加颗粒
}

# 工具“近似替代/高度相关”集合：用于扣分减免
# 依据 Adobe/摄影教材对功能重叠的解释（见回答中的引用）
SUBSTITUTES = [
    ("exposure", "brightness"),
    ("saturation", "vibrance"),
    ("highlights", "whites"),
    ("shadows",   "blacks"),
]

# 目标分数的容差（例如算出 7 分时也可归入 8 分桶）
SCORE_BIN_TOL = 1

def _rng(seed: int) -> random.Random:
    return random.Random()

def _norm_int(rng: random.Random, mean: float, sd: float, lo: int, hi: int) -> int:
    # 截断正态，直到落入区间
    for _ in range(32):
        v = int(round(rng.gauss(mean, sd)))
        if lo <= v <= hi:
            return v
    return min(hi, max(lo, int(round(mean))))  # 兜底

def _weighted_choice_k_without_replacement(rng: random.Random, items: List[str], weights: List[float], k: int) -> List[str]:
    # 简单实现：归一化后按权重迭代挑选
    selected = []
    pool = list(zip(items, weights))
    for _ in range(min(k, len(pool))):
        total = sum(w for _, w in pool)
        r = rng.random() * total
        acc = 0.0
        for i, (it, w) in enumerate(pool):
            acc += w
            if r <= acc:
                selected.append(it)
                pool.pop(i)
                break
    return selected

def _pick_level(rng: random.Random) -> str:
    # 轻度更常见，其次中度，重度与极限较少
    probs = [0.5, 0.3, 0.15, 0.05]
    r = rng.random()
    acc = 0.0
    for level, p in zip(LEVEL_ORDER, probs):
        acc += p
        if r <= acc:
            return level
    return LEVEL_ORDER[-1]

def _sample_value_in_level(rng: random.Random, level: str, bias: float = 0.0) -> int:
    a,b,c,d = ADJUSTMENT_LEVELS[level]
    # 方向：根据 bias 调整正负概率（把 [-1,1] 映射到 [0,1] 概率）
    p_pos = (bias + 1.0) / 2.0
    if rng.random() < p_pos:
        lo, hi = c, d
    else:
        lo, hi = a, b
    return rng.randint(lo, hi)

def _value_level(abs_v: int) -> str:
    v = abs(abs_v)
    for lv in LEVEL_ORDER:
        a,b,c,d = ADJUSTMENT_LEVELS[lv]
        # 以正向区间为准判断级别
        if c <= v <= d:
            return lv
    return "extreme"

def _level_index(level: str) -> int:
    return LEVEL_ORDER.index(level)

def _score_pair(template: Dict[str,int], variant: Dict[str,int]) -> float:
    """
    评分（越接近模板越高）。从 10 起，累计扣分，最后四舍五入并夹到 [-10,10]。
    规则：
      - 档位差 |Δ|<=1 不扣；>1 每超一档扣 2 分
      - 符号翻转额外扣 3 分
      - 少了参数扣 1.5，多了参数扣 1.0
      - 若“少了”但被近似替代项覆盖，或“多了”的是近似替代，则相应扣分 × 0.5
    """
    base = 10.0
    penalty = 0.0

    # 映射：找出替代对
    subs = {frozenset(p) for p in SUBSTITUTES}

    # 处理共同参数
    for k, v0 in template.items():
        if k in variant:
            v1 = variant[k]
            lv0 = _value_level(v0)
            lv1 = _value_level(v1)
            d = abs(_level_index(lv1) - _level_index(lv0))
            if d > 1:
                penalty += 2.0 * (d - 1)
            # 符号翻转
            if (v0 == 0) != (v1 == 0):
                pass
            if v0 * v1 < 0:
                penalty += 3.0
        else:
            # 缺失参数：检查是否被替代
            p = 1.5
            for a,b in SUBSTITUTES:
                s = frozenset((a,b))
                if k in s:
                    the_other = list(s - {k})[0]
                    if the_other in variant:
                        p *= 0.5
            penalty += p

    # 处理新增参数
    for k in variant.keys():
        if k not in template:
            p = 1.0
            # 若是某个模板键的替代项，减半
            for a,b in SUBSTITUTES:
                s = frozenset((a,b))
                if k in s:
                    the_other = list(s - {k})[0]
                    if the_other in template:
                        p *= 0.5
            penalty += p

    score = round(10.0 - penalty)
    return max(-10, min(10, int(score)))

def _noisify(rng: random.Random, template: Dict[str,int]) -> Dict[str,int]:
    """
    基于模板做一次随机扰动：
      - 以 0.6 概率调整已有参数的档位（-1/0/+1/+2 档；越大越少见）
      - 以 0.15 概率删除一个参数
      - 以 0.25 概率新增一个参数（按先验采样）
      - 以 0.15 概率对若干参数“轻微抖动”（±5~10）
    """
    v = dict(template)

    # 调整已有
    for k in list(v.keys()):
        if rng.random() < 0.6:
            cur = v[k]
            base_lv = _value_level(cur)
            step = rng.choices([0,1,2,-1], weights=[2,1,0.5,1])[0]
            # 方向：可能翻转
            sign_flip = (rng.random() < 0.15)
            # 目标档
            idx = max(0, min(len(LEVEL_ORDER)-1, _level_index(base_lv)+step))
            tgt_lv = LEVEL_ORDER[idx]
            # 在目标档随机取值
            bias = DIRECTION_BIAS.get(k, 0.0)
            new_val = _sample_value_in_level(rng, tgt_lv, bias=bias)
            if sign_flip:
                new_val = -new_val
            v[k] = new_val

    # 删除一部分
    keys = list(v.keys())
    if keys and rng.random() < 0.15:
        drop_k = rng.choice(keys)
        v.pop(drop_k, None)

    # 新增一部分
    if rng.random() < 0.25:
        cand = [x for x in ALLOWED if x not in v]
        if cand:
            weights = [TOOL_PRIOR.get(x,0.1) for x in cand]
            new_k = _weighted_choice_k_without_replacement(rng, cand, weights, 1)[0]
            lvl = _pick_level(rng)
            val = _sample_value_in_level(rng, lvl, DIRECTION_BIAS.get(new_k,0.0))
            v[new_k] = val

    # 轻微抖动
    for k in list(v.keys()):
        if rng.random() < 0.15:
            v[k] = int(max(-100, min(100, v[k] + rng.randint(-10,10))))

    return v

def _make_template_10(rng: random.Random) -> Dict[str,int]:
    # 选择工具个数：正态截断，偏好 3–5
    k = _norm_int(rng, mean=4, sd=1.0, lo=2, hi=7)
    items = list(ALLOWED)
    weights = [TOOL_PRIOR.get(x,0.1) for x in items]
    picked = _weighted_choice_k_without_replacement(rng, items, weights, k)

    params = {}
    for tool in picked:
        lvl = _pick_level(rng)
        bias = DIRECTION_BIAS.get(tool, 0.0)
        params[tool] = _sample_value_in_level(rng, lvl, bias)
    return params

def _bin_ok(target: int, got: int, tol: int = SCORE_BIN_TOL) -> bool:
    return (got >= target - tol) and (got <= target + tol)


def _read_negative_instructions(jsonl_path: str, rng: random.Random, need_n: int = -1) -> List[str]:
    path = Path(jsonl_path)
    lines: List[str] = []
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                obj = json.loads(line)
                ins = obj.get("instruction") or obj.get("request") or obj.get("expert_summary") or ""
                if isinstance(ins, list): 
                    ins = "；".join(map(str, ins))
                if isinstance(ins, str) and ins.strip():
                    lines.append(ins.strip())
            except Exception:
                # 纯文本行
                lines.append(line)
    rng.shuffle(lines)
    return lines[:need_n]

# =============== 指令生成（LLM 或规则） ===============


def _strength_bucket(v: int) -> str:
    """Map absolute value to a natural-language intensity adverb."""
    a = abs(int(v))
    buckets = [(20, [("slightly", 0.5), ("a bit", 0.3), ("just a touch", 0.15), ("barely", 0.05)]), (40, [("gently", 0.4), ("mildly", 0.3), ("softly", 0.2), ("subtly", 0.1)]), (60, [("moderately", 0.4), ("somewhat", 0.3), ("noticeably", 0.2), ("fairly", 0.1)]), (80, [("noticeably", 0.35), ("markedly", 0.3), ("distinctly", 0.2), ("clearly", 0.15)]),
        (100, [("strongly", 0.4), ("significantly", 0.3), ("extremely", 0.2), ("radically", 0.1), ("as much as possible", 0.25)]), ]
    for threshold, adverbs in buckets:
        if a <= threshold:
            words, weights = zip(*adverbs)
            return random.choices(words, weights=weights, k=1)[0]
    return "dramatically"

def _join_phrases(parts: List[str]) -> str:
    """Join phrases with commas and 'and' for the last one."""
    parts = [p for p in parts if p]
    if not parts: return ""
    if len(parts) == 1: return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _pick_template(templates: List[Tuple[str, float]]) -> str:
    """从 (模板, 权重) 列表中加权选择一个模板。"""
    if not templates:
        return ""
    phrases, weights = zip(*templates)
    return random.choices(phrases, weights=weights, k=1)[0]

def _phrase_light(k: str, v: int) -> str:
    s = _strength_bucket(v)
    templates = []
    if k == "exposure":
        if v > 0:
            templates = [(f"{s} increase overall exposure", 0.4), (f"brighten up the whole photo {s}", 0.3), (f"make the entire image {s} brighter", 0.2), (f"add a stop of light {s}", 0.1), ]
        else:
            templates = [(f"{s} reduce overall exposure", 0.4), (f"darken the entire image {s}", 0.3), (f"decrease the overall brightness {s}", 0.2), (f"pull the exposure down {s}", 0.1), ]
    elif k == "brightness":
        if v > 0:
            templates = [(f"{s} raise overall brightness", 0.4), (f"lift the midtones {s}", 0.4), (f"make it look a bit brighter, focusing on the middle tones", 0.2), ]
        else:
            templates = [(f"{s} lower overall brightness", 0.4), (f"bring down the midtones {s}", 0.4), (f"reduce the brightness a tad, especially in the mid-range", 0.2), ]
    elif k == "contrast":
        if v > 0:
            templates = [(f"{s} increase contrast for more punch", 0.4), (f"make the image pop more by adding {s} contrast", 0.3), (f"add some definition with {s} more contrast", 0.2), (f"deepen the darks and brighten the lights {s}", 0.1), ]
        else:
            templates = [(f"{s} soften overall contrast", 0.4), (f"reduce the contrast for a flatter look", 0.3), (f"make the transition between light and dark areas {s} smoother", 0.3), ]
    elif k == "natural_contrast":
        if v > 0:
            templates = [(f"{s} add an S-curve pop (more contrast)", 0.5), (f"apply a gentle S-curve to enhance contrast {s}", 0.3), (f"boost the tonal range naturally {s}", 0.2), ]
        else:
            templates = [(f"{s} flatten the tone curve slightly", 0.5), (f"reduce the S-curve effect {s}", 0.3), (f"make the tones more compressed and gentle {s}", 0.2), ]
    elif k == "highlights":
        if v < 0:
            templates = [(f"{s} pull down highlights to recover detail", 0.5), (f"recover detail in the brightest parts of the image {s}", 0.3), (f"tame the hot spots by reducing highlights {s}", 0.2), ]
        else:
            templates = [(f"{s} boost highlights for sparkle", 0.5), (f"make the bright areas {s} brighter", 0.3), (f"add a bit of shine to the highlights {s}", 0.2), ]
    elif k == "shadows":
        if v > 0:
            templates = [(f"{s} lift shadows to reveal detail", 0.5), (f"open up the dark areas {s}", 0.3), (f"bring out information hidden in the shadows {s}", 0.2), ]
        else:
            templates = [(f"{s} deepen shadows for richer depth", 0.5), (f"crush the shadows {s}", 0.3), (f"make the darks darker for more drama {s}", 0.2), ]
    elif k == "whites":
        if v > 0:
            templates = [(f"{s} raise the white point for a cleaner white", 0.5), (f"make the whites truly white {s}", 0.3), (f"set a brighter white point {s}", 0.2), ]
        else:
            templates = [(f"{s} tame whites to avoid clipping", 0.5), (f"pull back on the brightest whites {s}", 0.3), (f"prevent the whites from blowing out {s}", 0.2), ]
    elif k == "blacks":
        if v < 0:
            templates = [(f"{s} deepen blacks to anchor contrast", 0.5), (f"set a true black point {s}", 0.3), (f"make the blacks richer and deeper {s}", 0.2), ]
        else:
            templates = [(f"{s} lift the black point for a matte look", 0.5), (f"raise the blacks for a faded effect {s}", 0.3), (f"make the darkest parts a bit grayish {s}", 0.2), ]
    return _pick_template(templates)

def _phrase_color(k: str, v: int) -> str:
    s = _strength_bucket(v)
    templates = []
    if k == "temperature":
        if v > 0:
            templates = [(f"{s} warm the image", 0.4), (f"add a {s} warmer color cast", 0.25), (f"make it feel sunnier by warming it up {s}", 0.2), (f"give it a more golden hour feel", 0.15), ]
        else:
            templates = [(f"{s} cool the image", 0.4), (f"add a {s} cooler color tone", 0.25), (f"shift the white balance to be {s} cooler", 0.2), (f"give it a cooler, more bluish tone", 0.15), ]
    elif k == "tint":
        if v > 0:  # Note: Historically, positive tint is green in many editors like Adobe's.
            templates = [(f"{s} shift tint toward green", 0.5), (f"add a {s} green tint", 0.3), (f"neutralize a magenta cast by adding green {s}", 0.2), ]
        else:
            templates = [(f"{s} shift tint toward magenta", 0.5), (f"add a {s} magenta tint", 0.3), (f"counteract a green cast by adding magenta {s}", 0.2), ]
    elif k == "saturation":
        if v > 0:
            templates = [(f"{s} boost overall saturation", 0.4), (f"make all colors {s} more intense", 0.25), (f"increase the color saturation {s}", 0.2), (f"make the colors more vivid", 0.15), ]
        else:
            templates = [(f"{s} mute overall saturation", 0.4), (f"desaturate the image {s}", 0.3), (f"pull back on the colors {s}", 0.2), (f"create a more muted color palette", 0.1), ]
    elif k == "vibrance":
        if v > 0:
            templates = [(f"{s} boost vibrance to enhance muted colors", 0.4), (f"intelligently boost the less saturated colors {s}", 0.3), (f"make the colors pop {s} without overdoing skin tones", 0.2), (f"make colors more lively without affecting skin {s}", 0.1), ]
        else:
            templates = [(f"{s} reduce vibrance to keep colors restrained", 0.5), (f"tone down the most vibrant colors {s}", 0.3), (f"subtly desaturate the most intense colors", 0.2), ]
    return _pick_template(templates)


def _phrase_texture(k: str, v: int) -> str:
    s = _strength_bucket(v)
    templates = []
    if k == "sharpness":
        if v > 0:
            templates = [(f"{s} sharpen fine details", 0.4), (f"increase the acutance {s}", 0.2), (f"make the edges {s} crisper", 0.2), (f"enhance texture and clarity {s}", 0.2), ]
        else:
            templates = [(f"{s} soften details", 0.5), (f"reduce the sharpness for a softer look", 0.3), (f"apply a slight blur for a dreamy effect", 0.2), ]
    elif k == "vignette":
        if v < 0:
            templates = [(f"{s} add a subtle dark vignette to draw focus", 0.4), (f"darken the corners {s}", 0.3), (f"apply a {s} post-crop vignette", 0.2), (f"focus the viewer's eye on the center with a dark vignette", 0.1), ]
        else:
            templates = [(f"{s} brighten the corners", 0.5), (f"add a white vignette {s}", 0.3), (f"create a light, airy feel with a white vignette", 0.2), ]
    elif k == "fade":
        if v > 0:
            templates = [(f"{s} add a matte, washed-out fade", 0.4), (f"give it a {s} faded film look", 0.3), (f"lift the black point for a faded style", 0.2), (f"give it a washed-out, cinematic feel {s}", 0.1), ]
        else:
            templates = [(f"{s} reduce fade", 1), ]
    elif k == "grain":
        if v > 0:
            templates = [(f"add {s} film grain for texture", 0.4), (f"give it a gritty, analog feel with some grain", 0.3), (f"introduce a bit of noise to emulate film {s}", 0.3), ]
        else:
            templates = [(f"{s} reduce visible grain", 1), ]
    return _pick_template(templates)

    

_LIGHT_KEYS = {
    "exposure","brightness","contrast","natural_contrast",
    "highlights","shadows","whites","blacks"
}
_COLOR_KEYS = {"temperature","tint","saturation","vibrance"}
_TEXTURE_KEYS = {"sharpness","vignette","fade","grain"}

def _rule_based_instruction(params: Dict[str,int]) -> str:
    """
    English, explicit increase/decrease, richer wording.
    Produces up to 3 clauses: Lighting; Color; Texture/Style.
    The intensity adverbs scale with the magnitude, so a '6' set reads milder than a '10' set.
    """
    lighting, color, texture = [], [], []

    # Lighting block
    for k in ["exposure","brightness","contrast","natural_contrast","highlights","shadows","whites","blacks"]:
        if k in params and params[k] != 0:
            lighting.append(_phrase_light(k, params[k]))
    # Color block
    for k in ["temperature","tint","saturation","vibrance"]:
        if k in params and params[k] != 0:
            color.append(_phrase_color(k, params[k]))
    # Texture/Style block
    for k in ["sharpness","vignette","fade","grain"]:
        if k in params and params[k] != 0:
            texture.append(_phrase_texture(k, params[k]))

    parts = []
    if lighting:
        parts.append(_join_phrases(lighting))
    if color:
        parts.append(_join_phrases(color))
    if texture:
        parts.append(_join_phrases(texture))

    if not parts:
        return ""

    # 1–2 sentences total; merge gracefully
    if len(parts) == 1:
        return parts[0].rstrip(".") + "."
    if len(parts) == 2:
        return parts[0].rstrip(".") + "; " + parts[1].rstrip(".") + "."
    return parts[0].rstrip(".") + "; " + parts[1].rstrip(".") + "; " + parts[2].rstrip(".") + "."


def llm_generate_instruction(params: Dict[str, int], use_llm=True):
    expert_instruction = _rule_based_instruction(params)
    amateur_instruction = ""
    
    if use_llm:
        amateur_instruction = generate_amateur_instruction(params, expert_instruction)
    
    return expert_instruction, amateur_instruction


def _generate_repeating_instruction(instruction: str) -> str:
    """
    重复原始指令的末尾m个单词n次，制造冗长的错误样例
    """
    words = instruction.replace(".", "").split()
    l = len(words)
    n = random.randint(2, l)
    m = random.randint(2, 10)
    tail = " ".join(words[-m:])
    new_instruction = " ".join(words) + " " + (" " + tail) * n
    return new_instruction


# # =============== 总调度函数 ===============
#
# def generate_instruction_sets(
#     num_groups: int,
#     per_group_counts: Dict[int, int],
#     negative_jsonl_path: Optional[str] = None,
#     seed: int = 0,
#     use_llm: bool = True,
#     max_trials_per_bin: int = 500,
# ) -> List[Dict[str, Any]]:
#     """
#     生成若干“以 10 分模板为中心”的分数段指令集合。
#     - per_group_counts 例：{10:2, 8:4, 6:4, 4:4, 2:4, 0:4, -2:4, -4:2, -10:2}
#     - 若包含 -10，则从 negative_jsonl_path 随机抽该数量的现成指令
#     """
#     rng = _rng(seed)
#     results: List[Dict[str, Any]] = []
#
#     # 预先加载 -10 指令池
#     neg_needed_total = sum(per_group_counts.get(-10, 0) for _ in range(num_groups))
#     neg_pool = _read_negative_instructions(negative_jsonl_path, need_n=neg_needed_total, rng=rng) if neg_needed_total>0 and negative_jsonl_path else []
#
#     neg_idx = 0
#
#
#     for gi in tqdm(range(num_groups)):
#         all_insts=[]
#         template = _make_template_10(rng)
#         group_items: Dict[int, List[Dict[str,Any]]] = {sc: [] for sc in per_group_counts.keys()}
#
#         # 先放入 10 分模板（如果 10 分需要）
#         for _ in range(per_group_counts.get(10, 0)):
#             expert_instruction, amateur_instruction = llm_generate_instruction(template, use_llm)
#             if not expert_instruction:
#                 print("[Warn] 生成 10 分指令失败，跳过该条。")
#                 continue
#             group_items[10].append({"params": template,  "score": 10,"instruction": expert_instruction,"source":"template"})
#             all_insts.append(expert_instruction)
#             if amateur_instruction:
#                 group_items[10].append({"params": template,  "score": 10,"instruction": amateur_instruction,"source":"generated"})
#                 all_insts.append(amateur_instruction)
#
#         # 其他分数—接受采样直到凑满
#         for score_target, need_n in per_group_counts.items():
#             if score_target == 10 or need_n <= 0:
#                 continue
#             if score_target == -10:
#                 # 从负样本池填充
#                 take = min(need_n, len(neg_pool) - neg_idx)
#                 for _ in range(take):
#                     ins = neg_pool[neg_idx]
#                     neg_idx += 1
#                     group_items[-10].append({"instruction": ins, "score": -10, "source": "jsonl"})
#
#                 for _ in range(need_n):
#                     # pick one inst from all_insts to make a repeating error
#                     if all_insts:
#                         ins = rng.choice(all_insts)
#                         ins = _generate_repeating_instruction(ins)
#                         group_items[-10].append({"instruction": ins, "score": -10, "source": "jsonl"})
#
#                 continue
#
#             trials = 0
#             while len(group_items[score_target]) < need_n and trials < max_trials_per_bin:
#                 trials += 1
#                 cand = _noisify(rng, template)
#                 sc = _score_pair(template, cand)
#                 if _bin_ok(score_target, sc, SCORE_BIN_TOL):
#                     expert_instruction, amateur_instruction =  llm_generate_instruction(cand, use_llm)
#                     group_items[score_target].append({"params": cand, "score": sc,"instruction": expert_instruction,"source":"generated"})
#                     all_insts.append(expert_instruction)
#                     if amateur_instruction:
#                         group_items[score_target].append({"params": cand, "score": sc,"instruction": amateur_instruction,"source":"generated"})
#                         all_insts.append(amateur_instruction)
#
#         results.append({"items": group_items})
#
#     return results

# if __name__ == "__main__":
#     # 你要的固定配置（随时改数值即可）
#     GROUPS = 100
#     PER_GROUP_COUNTS = {10: 3,  6: 3, 3: 2,  0: 2, -3: 2, -6: 3, -10: 2}
#     OUT = r"D:\Codes\Image_Edit_Agent\datasets\GIER_synthesis\instructions_100.jsonl"
#     NEG_JSONL = r"D:\Codes\Image_Edit_Agent\datasets\GIER_synthesis\negative_examples.jsonl"   # -10 分指令的来源；没有就留空或去掉-10桶
#     SEED = 42
#     USE_LLM = True  # 关掉则用规则兜底生成中文指令
#
#     # 生成各分数段的参数+中文指令
#     results = generate_instruction_sets(
#         num_groups=GROUPS,
#         per_group_counts=PER_GROUP_COUNTS,
#         negative_jsonl_path=NEG_JSONL,
#         seed=SEED,
#         use_llm=USE_LLM,
#     )
#
#     # 写 JSONL（每行一条）
#     written = save_instruction_sets_jsonl(results, OUT, ensure_ascii=False)
#     print(f"[OK] 写入 {OUT} ，共 {written} 条")
#
# ---- 保存为 JSONL 的函数 ----
def save_instruction_sets_jsonl(
    results: List[Dict[str, Any]],
    out_path: str,
    *,
    ensure_ascii: bool = False
) -> int:
    """
    将 generate_instruction_sets 的结果扁平化保存为 JSONL。
    每一行包含：group_id, score, instruction, params, template_params（同组共享）
    返回写入的样本条数。
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for gid, group in enumerate(results):
            template = group.get("template_params", {})
            items_by_score: Dict[int, List[Dict[str, Any]]] = group.get("items", {})
            # items_by_score 是 {score: [ {params, instruction, score, ...}, ... ]}
            for score, items in items_by_score.items():
                if not items:
                    continue
                for it in items:
                    line_obj = {
                        "group_id": gid + 101,
                        "score": int(it.get("score", score)),
                        "instruction": it.get("instruction", ""),
                        "params": it.get("params", {}),
                        "source": it.get("source", ""),
                    }
                    f.write(json.dumps(line_obj, ensure_ascii=ensure_ascii) + "\n")
                    n += 1
    return n


# =============== 单组生成（线程工作单元） ===============

def _generate_single_group(group_id: int, per_group_counts: Dict[int, int], use_llm: bool, max_trials_per_bin: int, base_seed: int, neg_pool: List[str]) -> Dict[str, Any]:
    """
    为单个group生成所有分数段的指令，作为线程池的工作单元。
    注意：随机数生成器(rng)在此函数内部基于 group_id 和 base_seed 初始化，以保证线程安全和结果可复现。
    """
    rng = random.Random(base_seed + group_id)
    
    # per_group_counts
    # {10: 3,  6: 3, 3: 2,  0: 2, -3: 2, -6: 3, -10: 2}
    # 随机修改这个模板
    per_group_counts_list = [{
        10: 3,
        6: 3,
        2: 2,
        0: 2,
        -2: 2,
        -7: 3,
        -10: 1
    }, {
        10: 3,
        7: 3,
        2: 2,
        0: 2,
        -3: 2,
        -6: 3,
        -10: 1
    }, {
        10: 3,
        7: 3,
        3: 2,
        0: 2,
        -4: 2,
        -8: 3,
        -10: 1
    }, {
        10: 3,
        7: 3,
        4: 2,
        0: 2,
        -4: 2,
        -7: 3,
        -10: 1
    }, {
        10: 3,
        8: 3,
        4: 2,
        0: 2,
        -2: 2,
        -6: 3,
        -10: 1
    }, ]
    per_group_counts = rng.choice(per_group_counts_list)
    for key, val in per_group_counts.items():
        # 0.2的概率减少1
        random_number = rng.random()
        if random_number < 0.2:
            per_group_counts[key] = val - 1
        elif random_number > 0.8:
            per_group_counts[key] = val + 1
        else:
            per_group_counts[key] = val
    
    template = _make_template_10(rng)
    group_items: Dict[int, List[Dict[str, Any]]] = {sc: [] for sc in per_group_counts.keys()}
    all_insts = []
    
    # 1. 处理 10 分样本
    if 10 in per_group_counts:
        for _ in range(per_group_counts[10]):
            expert_instruction, amateur_instruction = llm_generate_instruction(template, use_llm)
            if not expert_instruction:
                continue
            group_items[10].append({
                "params": template,
                "score": 10,
                "instruction": expert_instruction,
                "source": "template"
            })
            all_insts.append(expert_instruction)
            if amateur_instruction:
                group_items[10].append({
                    "params": template,
                    "score": 10,
                    "instruction": amateur_instruction,
                    "source": "generated"
                })
                all_insts.append(amateur_instruction)
    
    # 2. 处理其他分数段
    for score_target, need_n in per_group_counts.items():
        if score_target == 10 or need_n <= 0:
            continue
        
        # 2a. 处理 -10 分（从共享的负样本池中安全地抽样）
        if score_target == -10 and need_n > 0:
            # 从负样本池填充
            if neg_pool:
                # 随机抽取，避免多线程同时从一个位置取
                chosen_negs = rng.sample(neg_pool, min(need_n, len(neg_pool)))
                for ins in chosen_negs:
                    group_items[-10].append({
                        "instruction": ins,
                        "score": -10,
                        "source": "irrelevant_error"
                    })
            
            # 基于已有指令制造重复错误
            if all_insts:
                for _ in range(need_n):
                    ins = rng.choice(all_insts)
                    ins_repeat = _generate_repeating_instruction(ins)
                    group_items[-10].append({
                        "instruction": ins_repeat,
                        "score": -10,
                        "source": "repeating_error"
                    })
            continue
        
        # 2b. 通过加噪生成其他分数样本
        trials = 0
        while len(group_items.get(score_target, [])) < need_n and trials < max_trials_per_bin:
            trials += 1
            cand = _noisify(rng, template)
            sc = _score_pair(template, cand)
            
            if _bin_ok(score_target, sc, SCORE_BIN_TOL):
                expert_instruction, amateur_instruction = llm_generate_instruction(cand, use_llm)
                if not expert_instruction: continue
                
                group_items.setdefault(score_target, []).append({
                    "params": cand,
                    "score": sc,
                    "instruction": expert_instruction,
                    "source": "template"
                })
                all_insts.append(expert_instruction)
                
                if amateur_instruction:
                    # 确保不超过需要的数量
                    group_items[score_target].append({
                        "params": cand,
                        "score": sc,
                        "instruction": amateur_instruction,
                        "source": "generated"
                    })
                    all_insts.append(amateur_instruction)
    
    return {
        "template_params": template,
        "items": group_items
    }


# =============== 总调度函数（多线程版） ===============

def generate_instruction_sets_multithreaded(num_groups: int, per_group_counts: Dict[int, int], negative_jsonl_path: Optional[str] = None, seed: int = 0, use_llm: bool = True, max_trials_per_bin: int = 500, max_workers: int = 8,  # 控制并发线程数
                                            ) -> List[Dict[str, Any]]:
    """
    多线程版本：生成若干“以 10 分模板为中心”的分数段指令集合。
    每个 group 的生成任务被分配给一个独立的线程。
    """
    # 预加载所有需要的负样本指令，供所有线程共享读取
    # 每个线程将从中独立、随机地抽样，因此线程安全
    main_rng = random.Random(seed)
    neg_pool = _read_negative_instructions(negative_jsonl_path, main_rng)
    print(f"已加载 {len(neg_pool)} 条负样本指令到共享池。")
    
    results = [None] * num_groups  # 预分配列表以保证顺序
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 为每个group提交一个任务
        futures = {executor.submit(_generate_single_group, group_id=i + 101, per_group_counts=per_group_counts, use_llm=use_llm, max_trials_per_bin=max_trials_per_bin, base_seed=seed, neg_pool=neg_pool): i for i in range(num_groups)}
        
        # 使用tqdm显示进度并收集结果
        for future in tqdm(as_completed(futures), total=num_groups, desc="Generating Groups"):
            original_index = futures[future]
            try:
                result = future.result()
                results[original_index] = result
            except Exception as e:
                print(f"线程 {original_index} 发生错误: {e}")
                # 可以选择将错误信息放入结果或直接跳过
                results[original_index] = {
                    "error": str(e),
                    "items": {}
                }
    
    # 过滤掉可能失败的任务
    return [res for res in results if res and "error" not in res]



if __name__ == "__main__":
    # 你要的固定配置（随时改数值即可）
    GROUPS = 1900
    PER_GROUP_COUNTS = {
        10: 3,
        6: 3,
        3: 2,
        0: 2,
        -3: 2,
        -6: 3,
        -10: 2
    }
    OUT = r"D:\Codes\Image_Edit_Agent\datasets\GIER_synthesis\instructions_2000.jsonl"  # 建议换个文件名
    NEG_JSONL = r"D:\Codes\Image_Edit_Agent\datasets\GIER_synthesis\negative_examples.jsonl"
    SEED = 0
    USE_LLM = True
    MAX_WORKERS = 16  # <--- 在此调整并发线程数，根据你的网络和CPU情况设置
    
    # 调用多线程函数生成各分数段的参数+指令
    results = generate_instruction_sets_multithreaded(
        num_groups=GROUPS,
        per_group_counts=PER_GROUP_COUNTS,
        negative_jsonl_path=NEG_JSONL,
        seed=SEED,
        use_llm=USE_LLM, max_workers=MAX_WORKERS,
    )

    # 写 JSONL（每行一条）
    written = save_instruction_sets_jsonl(results, OUT, ensure_ascii=False)
    print(f"[OK] 写入 {OUT} ，共 {written} 条")
