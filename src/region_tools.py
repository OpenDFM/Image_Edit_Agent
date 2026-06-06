from __future__ import annotations

import os
from typing import List, Tuple, Optional
import torch
import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils
# --- GroundingDINO ---
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T
from torchvision.ops import box_convert

# --- SAM3 ---
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 可从 config 读取默认配置（若不存在则以环境变量/自动推断为准）
try:
    from . import config
except Exception:
    config = None  # 允许独立运行本文件

# 硬件选择
DEVICE = config.DEVICE
Top_K = config.Top_K

# ------------------------- 基础工具 -------------------------

def clamp_box_xyxy(box, w, h):
    """
    把 xyxy box clamp 到图像边界内，并保证 x2>=x1, y2>=y1
    """
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w - 1)))
    y2 = int(max(0, min(y2, h - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def mask_inside_ratio(mask_bool, box_xyxy_int):
    """mask像素有多少比例落在该box内：inside(mask∩box) / area(mask)"""
    x1, y1, x2, y2 = box_xyxy_int
    if x2 <= x1 or y2 <= y1:
        return 0.0
    total = float(mask_bool.sum())
    if total <= 0:
        return 0.0
    inside = float(mask_bool[y1:y2+1, x1:x2+1].sum())
    return inside / (total + 1e-6)

def union_masks(masks_bool_list):
    """
    对多个 bool mask 做 OR/union
    - 用于：all 模式下合并多个实例
    """
    if len(masks_bool_list) == 0:
        return None
    out = masks_bool_list[0].copy()
    for m in masks_bool_list[1:]:
        out |= m
    return out

def box_intersection(a, b):
    """
    返回a,b两个box框的交叉面积
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    return iw * ih

# --- 局部工具：box -> mask（用于“直接使用mllm_box作为mask”兜底） ---
def box_to_mask_bool(box_int, h, w):
    if box_int is None:
        return None
    x1, y1, x2, y2 = box_int
    if x2 <= x1 or y2 <= y1:
        return None
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2+1, x1:x2+1] = True
    return m

def build_header_adaptive(total_w: int, title: str, lines: list, img_h: int, img_w: int):
    """
    根据图像尺寸自适应字号与行距，并让 title 自动换行（最多 2 行）。
    lines: [(text, color_bgr, thickness), ...]
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 基于「短边」缩放（你现在就是短边）
    short_side = min(int(img_h), int(img_w))
    short_side = max(320, min(short_side, 2000))

    font_scale_title = max(0.5, min(1.2, short_side / 800.0 * 0.9))
    font_scale       = max(0.45, min(1.0, short_side / 900.0 * 0.8))

    line_spacing = int(max(18, min(38, short_side / 30.0)))
    base_h       = int(max(36, min(80, short_side / 18.0)))

    left_pad, right_pad = 20, 20
    max_text_w = max(80, total_w - left_pad - right_pad)

    def _wrap_text_2lines(text: str, max_lines=2):
        # 已经手动含 \n 的，优先按 \n
        parts = str(text).split("\n")
        raw_lines = []
        for p in parts:
            p = p.strip()
            if p:
                raw_lines.append(p)
        if not raw_lines:
            raw_lines = [""]

        out_lines = []
        for seg in raw_lines:
            if len(out_lines) >= max_lines:
                break
            cur = ""
            words = seg.split(" ")
            used_all = True
            for wi, w in enumerate(words):
                cand = w if cur == "" else (cur + " " + w)
                (tw, th), _ = cv2.getTextSize(cand, font, font_scale_title, 2)
                if tw <= max_text_w or cur == "":
                    cur = cand
                else:
                    out_lines.append(cur)
                    cur = w
                    if len(out_lines) >= max_lines - 1:
                        used_all = (wi == len(words) - 1)
                        break
            if len(out_lines) < max_lines:
                out_lines.append(cur)

            # 如果还没用完，最后一行加 ...
            if len(out_lines) == max_lines and not used_all:
                last = out_lines[-1]
                while True:
                    (tw, _), _ = cv2.getTextSize(last + "...", font, font_scale_title, 2)
                    if tw <= max_text_w or len(last) <= 1:
                        break
                    last = last[:-1]
                out_lines[-1] = last + "..."
            break

        return out_lines[:max_lines]

    title_lines = _wrap_text_2lines(title, max_lines=2)

    # 计算 title 行高
    (_, th_title), _ = cv2.getTextSize("Ag", font, font_scale_title, 2)
    title_step = max(th_title + 8, int(line_spacing * 1.1))

    top_pad = 10
    gap_after_title = 8
    info_h = max(1, len(lines)) * line_spacing
    header_h = top_pad + len(title_lines) * title_step + gap_after_title + info_h + 10

    header = np.zeros((header_h, total_w, 3), dtype=np.uint8) + 255

    # draw title (multi-line)
    y = top_pad + th_title + 2
    for tln in title_lines:
        cv2.putText(header, tln, (left_pad, y), font, font_scale_title, (0, 0, 0), 2, cv2.LINE_AA)
        y += title_step

    # draw info lines
    y = top_pad + len(title_lines) * title_step + gap_after_title + th_title
    for text, color, thick in lines:
        cv2.putText(header, text, (left_pad, y), font, font_scale, color, int(thick), cv2.LINE_AA)
        y += line_spacing

    return header


def overlay_mask_single_color_bgr(
    img_bgr: np.ndarray,
    mask_bool: np.ndarray,
    color_bgr=(0, 0, 255),
    alpha: float = 0.45
):
    """把一个 mask 用单一颜色叠加到图上（只影响 mask 区域，不会越叠越白）。"""
    out = img_bgr.copy()
    if mask_bool is None:
        return out

    m = mask_bool.astype(bool)
    if not m.any():
        return out

    # 只在 mask 区域做 blend：out = (1-a)*out + a*color
    out_f = out.astype(np.float32)
    color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    out_f[m] = out_f[m] * (1.0 - alpha) + color * alpha
    return np.clip(out_f, 0, 255).astype(np.uint8)


def make_palette(n: int):
    """
    生成 n 个相对区分度高的 BGR 颜色
    """
    # 一组手工挑选的高区分度 BGR（可扩展）
    base = [
        (255,  80,  80),   # blue-ish
        ( 80, 255,  80),   # green
        (255, 200,  80),   # cyan-ish
        (255,  80, 255),   # magenta-ish
        ( 80, 255, 255),   # yellow-ish
        (180, 180,  50),
        ( 50, 180, 180),
        (180,  50, 180),
        (220, 120,  30),
        ( 30, 120, 220),
        (120,  30, 220),
    ]
    if n <= len(base):
        return base[:n]
    # 不够就循环
    out = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


def overlay_multi_masks_bgr(
    img_bgr: np.ndarray,
    masks_bool: np.ndarray,  # (N,H,W) bool
    colors_bgr,
    alpha: float = 0.45,
    draw_index: bool = True,
):
    """
    在同一张图上叠加多个 mask，不同 mask 不同颜色。
    修复：不再用 addWeighted(out,1.0,layer,alpha)，避免饱和发白。
    """
    out = img_bgr.copy()
    if masks_bool is None or masks_bool.size == 0:
        return out

    h, w = out.shape[:2]
    out_f = out.astype(np.float32)

    N = int(masks_bool.shape[0])
    for i in range(N):
        m = masks_bool[i].astype(bool)
        if not m.any():
            continue

        color = np.array(colors_bgr[i], dtype=np.float32).reshape(1, 1, 3)
        out_f[m] = out_f[m] * (1.0 - alpha) + color * alpha

    out = np.clip(out_f, 0, 255).astype(np.uint8)

    if draw_index:
        for i in range(N):
            m = masks_bool[i].astype(bool)
            if not m.any():
                continue
            ys, xs = np.where(m)
            cx = int(np.mean(xs)); cy = int(np.mean(ys))
            cx = max(0, min(cx, w - 1)); cy = max(0, min(cy, h - 1))
            cv2.putText(out, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def cut_by_score_gap(
    scores,
    max_keep=None,
    min_keep=1,
    gap_ratio=0.2
):
    """
    根据 score 的“最大断崖”自动决定保留数量 K

    Args:
        scores (array-like): shape (N,), 分数，越大越好
        max_keep (int | None): 最多保留多少个（防止爆）
        min_keep (int): 至少保留多少个
        gap_ratio (float): gap 相对阈值（防止过度截断）
            - gap < gap_ratio * score[0] 的断崖会被忽略

    Returns:
        k (int): 建议保留的数量
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        return 0

    order = np.argsort(-scores)
    scores = scores[order]
    N = len(scores)

    if N <= min_keep:
        return N

    top1 = scores[0]

    gaps = scores[:-1] - scores[1:]
    gap_idx = int(np.argmax(gaps))
    max_gap = gaps[gap_idx]

    k = N

    # 只有当 gap 足够大，才认为这里有“断崖”，才截断
    if max_gap >= gap_ratio * top1:
        k = gap_idx + 1   # 保留到最大间隔前一个

    k = max(k, min_keep)
    if max_keep is not None:
        k = min(k, max_keep)

    return k

def get_mllm_box_int(mllm_bbox_obj,w,h):
    """根据大模型返回的float的"bboxes": [[x1,y1,x2,y2]]，返回真实图片大小的x1,y1,x2,y2
    Args:
        mllm_bbox_obj (List[List[float]]): MLLM生成的归一化的float类型的box
        w (int): 图片的宽
        h (int): 图片的高
    Returns:
        [x1,x2,y1,y2]: 返回真实像素位置的[x1,x2,y1,y2]
    """
 
    if mllm_bbox_obj is None:
        return None

    b = None
    try:
        if isinstance(mllm_bbox_obj, dict):
            # {"bboxes": [[...], ...]}
            arr = mllm_bbox_obj.get("bboxes", None)
            if arr and isinstance(arr, (list, tuple)) and len(arr) > 0:
                b = arr[0]
        elif isinstance(mllm_bbox_obj, (list, tuple)):
            # could be [[...]] or [...]
            if len(mllm_bbox_obj) == 0:
                return None
            if isinstance(mllm_bbox_obj[0], (list, tuple)) and len(mllm_bbox_obj[0]) == 4:
                b = mllm_bbox_obj[0]
            elif len(mllm_bbox_obj) == 4 and all(isinstance(x, (int, float)) for x in mllm_bbox_obj):
                b = mllm_bbox_obj
    except Exception:
        b = None

    if b is None or len(b) != 4:
        return None

    x1n, y1n, x2n, y2n = [float(x) for x in b]
    # 归一化 -> 像素；注意 clamp 到 [0,1]
    x1n = max(0.0, min(1.0, x1n))
    y1n = max(0.0, min(1.0, y1n))
    x2n = max(0.0, min(1.0, x2n))
    y2n = max(0.0, min(1.0, y2n))

    # 转像素（xyxy）
    x1 = int(round(x1n * (w - 1)))
    y1 = int(round(y1n * (h - 1)))
    x2 = int(round(x2n * (w - 1)))
    y2 = int(round(y2n * (h - 1)))
    return clamp_box_xyxy([x1, y1, x2, y2], w, h)

def auto_feather_from_mask(
    mask_bool: np.ndarray,
    *,
    img_h: int,
    img_w: int,
    min_px: int = 3,
    max_px: int = 60,
) -> int:
    """
    根据 mask 的几何大小 + 图像大小 自动估计羽化半径（像素）。

    Args:
        mask_bool: (H, W) 的布尔或 0/1 mask
        img_h, img_w: 原图高度、宽度
        min_px: 最小羽化半径（下限）
        max_px: 最大羽化半径（上限）

    Returns:
        feather_px: 一个 int，表示建议的羽化像素
    """
    if mask_bool is None:
        return min_px

    mask = mask_bool.astype(bool)
    H, W = mask.shape

    # 没有前景的话，给个最小值
    ys, xs = np.where(mask)
    if ys.size == 0:
        return min_px

    # 1) 原图的短边，用来判断“目标对于整图有多大”
    img_short = max(320, min(int(img_h), int(img_w)))  # 加个下限防止过小

    # 2) mask 的包围框尺寸
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    box_h = y2 - y1 + 1
    box_w = x2 - x1 + 1
    box_short = max(4, min(box_h, box_w))  # 给个下限防止太极端

    # 3) 目标相对于整图的大小比例
    ratio = box_short / float(img_short)  # 越大，说明目标越大

    # 4) 根据比例映射到一个“羽化占 box_short 的比例”
    #    - 小目标（ratio ~ 0.05）：羽化占 3% 左右
    #    - 中等目标（ratio ~ 0.3）：羽化占 7% 左右
    #    - 大目标（ratio ~ 0.6+）：羽化占 12% 左右
    #    你可以根据实际效果改这几个点
    if ratio <= 0.05:
        alpha = 0.03
    elif ratio >= 0.6:
        alpha = 0.12
    else:
        # 在线性插值：ratio 从 0.05 -> 0.6，alpha 从 0.03 -> 0.12
        alpha = 0.03 + (ratio - 0.05) * (0.12 - 0.03) / (0.6 - 0.05)

    feather = int(alpha * box_short)

    # 5) 最后限制在一个绝对区间内
    feather = int(np.clip(feather, min_px, max_px))
    return feather
# ------------------------------SAM3调用-------------------------------------
def sam3_on_pil(
    pil_img,
    text_prompt,
    sam3_processor,
    top_k=Top_K,
    score_thr: float = 0.05,
    prefer_scores: bool = True,
):
    """在 pil_img 上根据 text_prompt 直接跑 SAM3。

    Args:
        pil_img (PIL.Image): 原图或者划分好的bbox
        text_prompt (str): obj描述
        sam3_processor (_type_): SAM模型
        top_k (int): 截断30个mask
        score_thr(float): 最低分数阈值
        prefer_scores(bool): 是否按照分数排序

    Returns:
         masks_bool (np.ndarray | None):
            形状为 (N, H, W) 的布尔掩码数组，其中：
                - N 为预测得到的 mask 数量
                - H, W 为图像高和宽
            每个 mask 表示一个候选分割区域（True 表示属于该区域）。
            若未检测到任何 mask，则返回 None。

        scores_np (np.ndarray | None):
            形状为 (N,) 的置信度分数数组，对应每个 mask 的预测分数。
            若模型未返回分数，或分数数量与 mask 数量不一致，则返回 None。
    """
    try:
        state = sam3_processor.set_image(pil_img)
        out = sam3_processor.set_text_prompt(state=state, prompt=text_prompt)

        masks = out.get("masks", None)
        scores = out.get("scores", None)

        if masks is None or len(masks) == 0:
            return None, None

        # ---- masks -> (N, H, W) bool ----
        if torch.is_tensor(masks):
            masks_t = masks.detach().cpu()
        else:
            masks_t = torch.as_tensor(masks).cpu()

        if masks_t.ndim == 4:
            masks_t = masks_t[:, 0, :, :]

        masks_bool = (masks_t.numpy() > 0.0)
        N = int(masks_bool.shape[0])

        # ---- scores -> (N,) float or None ----
        scores_np = None
        has_scores = False
        if scores is not None:
            if torch.is_tensor(scores):
                scores_np = scores.detach().cpu().float().numpy()
            else:
                scores_np = torch.as_tensor(scores).cpu().float().numpy()
            if scores_np.size == N:
                has_scores = True
            else:
                scores_np = None
                has_scores = False

        # ---- 先阈值筛选（仅当 scores 可用时）----
        if has_scores and score_thr is not None:
            keep = np.where(scores_np >= float(score_thr))[0]
            if keep.size == 0:
                return None, None
            masks_bool = masks_bool[keep]
            scores_np = scores_np[keep]
            N = int(masks_bool.shape[0])

        # ---- 再 top_k 截断（按 score 或 area 排序）----
        if top_k is not None:
            k = int(top_k)
            if k <= 0:
                k = 1
            k = min(k, N)

            if has_scores and prefer_scores and scores_np is not None:
                order = np.argsort(-scores_np)[:k]  # score 降序
                masks_bool = masks_bool[order]
                scores_np = scores_np[order]
            else:
                areas = masks_bool.reshape(N, -1).sum(axis=1)
                order = np.argsort(-areas)[:k]      # area 降序
                masks_bool = masks_bool[order]
                if scores_np is not None:
                    scores_np = scores_np[order]

        return masks_bool, scores_np

    except Exception as e:
        print(f"[Warning][SAM3] set_image/text_prompt failed: {e}")
        return None, None

def sam3_crop_and_paste(sam3_processor,full_pil, box_int, text_prompt,w,h):
    """
    在给定的 box 区域内裁剪图像，基于文本提示运行 SAM3，
    从裁剪区域中选取最优 mask，并将其映射（paste）回整图坐标系。

    Args:
        full_pil (PIL.Image.Image):
            原始整图（RGB）。

        box_int (tuple[int, int, int, int]):
            目标区域的整数边界框 (x1, y1, x2, y2)，
            坐标基于整图像素坐标系，且满足左上闭、右下闭的语义。
            若 x2 <= x1 或 y2 <= y1，则视为非法 box。

        text_prompt (str):
            用于 SAM3 分割的目标对象文本描述。

        h (int):
            原始整图高度（用于构造返回的整图 mask）。

        w (int):
            原始整图宽度（用于构造返回的整图 mask）。

    Returns:
        full_mask (np.ndarray | None):
            形状为 (h, w) 的布尔掩码，对应整图坐标系。
            mask 中 True 的区域表示在 box 内由 SAM3 预测得到的目标区域。
            若 box 非法或裁剪区域内未预测到任何 mask，则返回 None。

        conf (float | None):
            最优 mask 的置信度指标：
                - 若 SAM3 返回 scores，则为对应 mask 的 score；
                - 否则为该 mask 在裁剪区域内的像素面积。
            若未生成有效 mask，则返回 None。
    """
    try:
        x1, y1, x2, y2 = box_int
        if x2 <= x1 or y2 <= y1:
            return None, None

        crop_pil = full_pil.crop((x1, y1, x2 + 1, y2 + 1))
        # crop 场景下：prompt 可能为空/更不稳定，默认不强行用 score_thr 过滤，避免把有效 mask 过滤掉
        crop_masks_bool, crop_scores = sam3_on_pil(
            crop_pil,
            text_prompt,
            sam3_processor,
            score_thr=0.0
        )
        if crop_masks_bool is None:
            return None, None

        ch, cw = crop_masks_bool.shape[1], crop_masks_bool.shape[2]

        # 选 crop 内 best mask（score 优先，否则 area）
        if crop_scores is not None:
            best_i = int(np.argmax(crop_scores))
            conf = float(crop_scores[best_i])
        else:
            areas = crop_masks_bool.reshape(crop_masks_bool.shape[0], -1).sum(axis=1)
            best_i = int(np.argmax(areas))
            conf = float(areas[best_i])

        best_crop_mask = crop_masks_bool[best_i]  # (ch,cw)

        # paste 回全图
        full_mask = np.zeros((h, w), dtype=bool)
        yy2 = min(y1 + ch, h)
        xx2 = min(x1 + cw, w)
        full_mask[y1:yy2, x1:xx2] = best_crop_mask[: (yy2 - y1), : (xx2 - x1)]
        return full_mask, conf
    except Exception as e:
        print(f"[Warning][SAM3] set_image/text_prompt failed: {e}")
        return None, None

# ------------------------- 对外主入口：获得mask结果 -------------------------

def process_single_image(dino_model, sam3_processor, image_pil, prompt, target_scope, target_mllm_bbox, output_path):
    """
        1. 使用SAM3和prompt对全图作用，筛选分数大于0.1的top_k，并通过最大间隔选定前gap个最优的mask_1[gap]
        2. 使用GroundingDINO和prompt对全图作用，筛选出top_k，并通过最大间隔选定前gap个最优的box_1[gap]
        3. target_scope参数
            - single：为单个目标
            1) mask_1有结果，用box_1验证mask_1[0]在box_1的区域中
            2) mask_1失败或者没有结果，用box_1[0]单独做SAM3的prompt=""(prompt为空的)识别mask_2
            3) mask_1失败或者没有结果，box_1识别或者没有结果，用mllm_box单独做SAM3的prompt=""(prompt为空的)识别mask_3
            4）mask_2/mask_3失败，则直接使用mllm_box作为mask使用
            - all: 为多个目标
            1) mask_1[gap]有结果，用box_1[gap]验证mask_1[gap]在box_1的区域中
            2) mask_1[gap]失败或者没有结果，用每张box_1[gap]单独做SAM3的prompt=""(prompt为空的)识别mask_2[0]，之后合并作为mask
            3) mask_1失败或者没有结果，box_1识别或者没有结果，用mllm_box单独做SAM3的prompt=""(prompt为空的)识别mask_3[gap]
            4）mask_2/mask_3失败，则直接使用mllm_box作为mask使用
    """
    result_dict = {
        "success": False,               # 是否成功得到最终mask
        "target_scope": target_scope,   # "single" / "all"
        # DINO 输出
        "best_bbox": None,              # single时可记录最优框（xyxy float list），all时一般None
        "all_bboxes": [],               # DINO筛选后的候选框列表（含id/bbox/confidence/phrase）
        "phrase": "",                   # single: best_phrase；all: 可为空

        # 验证信息
        "verify": "",                   # "SAM3" / "SAM3 GroundingDINO" / "DINO crop SAM3" / "MLLM crop SAM3" / "MLLM box mask"
        "verify_box_idx": "",           # DINO验证或crop命中的box索引；无则空

        # SAM3信息（调参/调试用）
        "sam3_mask_idx": "",            # single选中的mask索引（这里常为0）；all可为空
        "sam3_kept_masks": 0,           # SAM3过滤+截断后保留的mask数量
        "sam3_crop_kept_masks": 0,      # 走 DINO->crop->SAM3 时，成功 paste 回全图的 mask 数

        # MLLM信息
        "mllm_bbox_used": None,         # 若用到mllm box（crop或直接box mask），记录其 int clamp 后xyxy

        # 置信度（按统一规则）
        "confidence": 0.0,              # SAM3来源->SAM3分数/均值；MLLM box mask->-1

        # mask输出
        "mask_rle": None                # 最终mask的RLE
    }

    try:
        image_np = np.array(image_pil)
        h, w, _ = image_np.shape

        # =========================================================
        # A) DINO：得到候选 bbox
        # =========================================================
        dino_found = False
        all_bboxes_list = []
        best_box_xyxy = None
        best_phrase = ""
        all_boxes_int = []

        try:
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_transformed, _ = transform(image_pil, None)
            image_transformed = image_transformed.to(DEVICE)

            boxes_norm, logits, phrases = predict(
                model=dino_model,
                image=image_transformed,
                caption=prompt,
                box_threshold=0.05,
                text_threshold=0.5
            )
            if len(boxes_norm) > 0:
                dino_found = True

                logits_1d = logits.squeeze()
                dino_scores_np = logits_1d.detach().cpu().numpy()

                k_gap = cut_by_score_gap(
                    dino_scores_np,
                    max_keep=int(Top_K),
                    min_keep=1,
                    gap_ratio=0.2
                )
                k = min(k_gap, len(boxes_norm))

                topk_vals, topk_idx = torch.topk(
                    logits_1d, k=k, largest=True, sorted=True
                )

                boxes_norm = boxes_norm[topk_idx]
                logits = logits_1d[topk_idx]
                phrases = [phrases[int(i)] for i in topk_idx]

                boxes_scaled = boxes_norm * torch.Tensor([w, h, w, h]).to(boxes_norm.device)
                all_boxes_xyxy = box_convert(
                    boxes=boxes_scaled,
                    in_fmt="cxcywh",
                    out_fmt="xyxy"
                ).cpu().numpy()

                for i in range(len(all_boxes_xyxy)):
                    all_bboxes_list.append({
                        "id": i,
                        "bbox": all_boxes_xyxy[i].tolist(),
                        "confidence": float(logits[i].item()),
                        "phrase": phrases[i]
                    })

                all_boxes_int = []
                for b in all_boxes_xyxy:
                    b_int = clamp_box_xyxy(list(map(int, b.tolist())), w, h)
                    all_boxes_int.append(b_int)

                # topk 已按分数降序，所以 0 就是最高分 box_1[0]
                best_box_xyxy = all_boxes_xyxy[0]
                best_phrase = phrases[0]

        except Exception as e:
            print(f"[Warning] GroundingDINO failed. err={e}")
            dino_found = False

        # =========================================================
        # B) SAM3：先整图 text prompt -> 得到 **候选 mask 集合（只做阈值0.05 + top_k）**
        #    - 这些候选会用于「左下角 SAM3 面板」可视化
        #    - 后续真正用于决策的 mask，会在本函数里再做“最大间隔”筛选
        # =========================================================
        sam3_cand_masks, sam3_cand_scores = sam3_on_pil(
            image_pil,
            prompt,
            sam3_processor,
        )

        # confidence
        confidence_source = ""   # "SAM_FULL" / "SAM_CROP" / "MLLM_BOX"
        sam_conf_list = None     # 全图 SAM（gap 后）分数数组（用于 mean）
        sam_crop_conf = None     # crop SAM 分数（用于单值或均值）

        # 记录候选数量（用于日志/可视化）
        if sam3_cand_masks is not None:
            result_dict["sam3_kept_masks"] = int(sam3_cand_masks.shape[0])

        # ===== 在 process_single_image 内做最大间隔筛选 =====
        sam3_gap_k = 0
        masks_bool, scores_np = None, None  # 用于决策的 SAM3 masks/scores
        if sam3_cand_masks is not None and sam3_cand_masks.shape[0] > 0:
            if sam3_cand_scores is not None and sam3_cand_scores.size == sam3_cand_masks.shape[0]:
                sam3_gap_k = cut_by_score_gap(
                    sam3_cand_scores,
                    max_keep=int(min(Top_K, sam3_cand_scores.size)),
                    min_keep=1,
                    gap_ratio=0.2,
                )
                sam3_gap_k = int(max(1, min(sam3_gap_k, sam3_cand_scores.size)))
                masks_bool = sam3_cand_masks[:sam3_gap_k]
                scores_np = sam3_cand_scores[:sam3_gap_k]
                sam_conf_list = scores_np
                result_dict["sam3_gap_keep"] = int(sam3_gap_k)
            else:
                # 没有 scores：不做 gap（按 area 排序后的 top_k 直接用）
                sam3_gap_k = int(sam3_cand_masks.shape[0])
                masks_bool = sam3_cand_masks
                scores_np = sam3_cand_scores

        # =========================================================
        # C) 根据 target_scope 选择 mask
        # =========================================================
        best_mask_idx = 0
        final_mask_bool = None
        chosen_boxes_for_all = []
        chosen_boxes_type = ""
        kept_masks = []

        # ---------- (1) SINGLE ----------
        if target_scope == "single":

            # S1: SAM3 有结果：直接用 SAM3 分最高的 mask 作为最终 mask
            if masks_bool is not None and masks_bool.shape[0] > 0:
                mask0 = masks_bool[0]
                final_mask_bool = mask0
                best_mask_idx = 0

                confidence_source = "SAM_FULL"
                result_dict["verify"] = "SAM3"
                result_dict["verify_box_idx"] = ""
                result_dict["sam3_mask_idx"] = int(best_mask_idx)

                # 若 DINO 也有结果：检查 mask0 的 0.75 像素是否落在任意一个 DINO box 中
                if dino_found and len(all_boxes_int) > 0:
                    THR = 0.75
                    best_i = -1
                    best_ratio = -1.0
                    for i, b_int in enumerate(all_boxes_int):
                        r = mask_inside_ratio(mask0, b_int)  # overlap(mask∩box) / area(mask)
                        if r > best_ratio:
                            best_ratio = r
                            best_i = i
                    if best_ratio >= THR:
                        result_dict["verify"] = "SAM3 GroundingDINO"
                        result_dict["verify_box_idx"] = int(best_i)

            # S2/S3/S4：只有当 SAM3 没有结果时才进入兜底链
            if (final_mask_bool is None or final_mask_bool.sum() == 0):

                # S2: 用 box_1[0] crop + SAM3(prompt="") 得到 mask_2
                if dino_found and len(all_boxes_int) > 0:
                    final_mask_bool, conf = sam3_crop_and_paste(
                        sam3_processor, image_pil, all_boxes_int[0], "", w, h
                    )
                    if final_mask_bool is not None and final_mask_bool.sum() > 0:
                        result_dict["verify"] = "DINO crop SAM3"
                        result_dict["verify_box_idx"] = 0
                        sam_crop_conf = float(conf)
                        confidence_source = "SAM_CROP"

                # S3: mask_2 失败 或 box_1 无 -> 用 mllm_box crop + SAM3(prompt="") 得到 mask_3
                if (final_mask_bool is None or final_mask_bool.sum() == 0):
                    mllm_box_int = get_mllm_box_int(target_mllm_bbox, w, h)
                    if mllm_box_int is not None:
                        final_mask_bool, conf3 = sam3_crop_and_paste(
                            sam3_processor, image_pil, mllm_box_int, "", w, h
                        )
                        if final_mask_bool is not None and final_mask_bool.sum() > 0:
                            result_dict["verify"] = "MLLM crop SAM3"
                            result_dict["verify_box_idx"] = ""
                            result_dict["mllm_bbox_used"] = mllm_box_int
                            sam_crop_conf = float(conf3)
                            confidence_source = "SAM_CROP"
                    else:
                        final_mask_bool = None

                # S4: mask_2/mask_3 仍失败 -> 直接用 mllm_box 作为 mask
                if (final_mask_bool is None or final_mask_bool.sum() == 0):
                    mllm_box_int = get_mllm_box_int(target_mllm_bbox, w, h)
                    if mllm_box_int is not None:
                        final_mask_bool = box_to_mask_bool(mllm_box_int, h, w)
                        if final_mask_bool is not None and final_mask_bool.sum() > 0:
                            result_dict["verify"] = "MLLM box mask"
                            result_dict["verify_box_idx"] = ""
                            result_dict["mllm_bbox_used"] = mllm_box_int
                            confidence_source = "MLLM_BOX"
                    else:
                        print(f"[mask跳过] SINGLE: 无有效 MLLM bbox 且所有路径失败: {os.path.basename(output_path)}")
                        return result_dict

            # SINGLE 下的 bbox/phrase 记录
            result_dict["best_bbox"] = None if best_box_xyxy is None else best_box_xyxy.tolist()
            result_dict["all_bboxes"] = all_bboxes_list
            result_dict["phrase"] = best_phrase

        # ---------- (2) ALL ----------
        else:
            # A0: 只要 SAM3 有结果 -> 直接用 SAM3 作为最终结果（ALL: union）
            if masks_bool is not None and masks_bool.shape[0] > 0:
                final_mask_bool = union_masks([masks_bool[i] for i in range(masks_bool.shape[0])])

                confidence_source = "SAM_FULL"
                result_dict["verify"] = "SAM3"
                result_dict["verify_box_idx"] = ""
                result_dict["sam3_mask_idx"] = ""  # all不强调单个idx

                # 若 DINO 也有结果：对 SAM3 最终 mask 通过 DINO 的任意 box 验证
                if dino_found and len(all_boxes_int) > 0:
                    THR = 0.75
                    best_i = -1
                    best_ratio = -1.0
                    for i, b_int in enumerate(all_boxes_int):
                        r = mask_inside_ratio(final_mask_bool, b_int)
                        if r > best_ratio:
                            best_ratio = r
                            best_i = i
                    if best_ratio >= THR:
                        result_dict["verify"] = "SAM3 GroundingDINO"
                        result_dict["verify_box_idx"] = int(best_i)

            # A2/A3/A4：只有当 SAM3 没有结果时，才进入兜底链
            if (final_mask_bool is None or final_mask_bool.sum() == 0):

                # A2: 对每个 DINO box crop + SAM3(prompt="") 得到 mask_2[0]，union
                if dino_found and len(all_boxes_int) > 0:
                    chosen_boxes_for_all = all_boxes_int
                    chosen_boxes_type = "gap_boxes"

                    kept_masks = []
                    conf_list = []
                    for box_int in chosen_boxes_for_all:
                        m_full, conf = sam3_crop_and_paste(
                            sam3_processor, image_pil, box_int, "", w, h
                        )
                        if m_full is not None and m_full.sum() > 0:
                            kept_masks.append(m_full)
                            conf_list.append(float(conf))

                    if len(kept_masks) > 0:
                        final_mask_bool = union_masks(kept_masks)
                        result_dict["verify"] = "DINO boxes crop SAM3"
                        result_dict["verify_box_idx"] = ""
                        result_dict["dino_chosen_boxes_type"] = chosen_boxes_type
                        result_dict["dino_chosen_boxes"] = chosen_boxes_for_all
                        result_dict["sam3_crop_kept_masks"] = int(len(kept_masks))

                        # ALL + DINO-crop 路径：用 SAM3 conf 的均值
                        if len(conf_list) > 0:
                            sam_crop_conf = float(np.mean(np.array(conf_list, dtype=np.float32)))
                        confidence_source = "SAM_CROP"

                # A3: mask_2 失败 或 box 无 -> 用 mllm_box crop + SAM3(prompt="") 得到 mask_3
                if (final_mask_bool is None or final_mask_bool.sum() == 0):
                    mllm_box_int = get_mllm_box_int(target_mllm_bbox, w, h)
                    if mllm_box_int is not None:
                        m_full, conf3 = sam3_crop_and_paste(
                            sam3_processor, image_pil, mllm_box_int, "", w, h
                        )
                        if m_full is not None and m_full.sum() > 0:
                            final_mask_bool = m_full
                            result_dict["verify"] = "MLLM crop SAM3"
                            result_dict["verify_box_idx"] = ""
                            result_dict["mllm_bbox_used"] = mllm_box_int
                            sam_crop_conf = float(conf3)
                            confidence_source = "SAM_CROP"

                # A4: mask_2/mask_3 仍失败 -> 直接用 mllm_box 作为 mask
                if (final_mask_bool is None or final_mask_bool.sum() == 0):
                    mllm_box_int = get_mllm_box_int(target_mllm_bbox, w, h)
                    if mllm_box_int is not None:
                        final_mask_bool = box_to_mask_bool(mllm_box_int, h, w)
                        if final_mask_bool is not None and final_mask_bool.sum() > 0:
                            result_dict["verify"] = "MLLM box mask"
                            result_dict["verify_box_idx"] = ""
                            result_dict["mllm_bbox_used"] = mllm_box_int
                            confidence_source = "MLLM_BOX"
                    else:
                        print(f"[mask跳过] ALL: 无有效 MLLM bbox 且所有路径失败: {os.path.basename(output_path)}")
                        return result_dict

            # ALL 下的 bbox/phrase 记录（你也可以按需汇总 phrase）
            result_dict["best_bbox"] = None
            result_dict["all_bboxes"] = all_bboxes_list
            result_dict["phrase"] = ""

        # confidence
        if confidence_source == "MLLM_BOX":
            result_dict["confidence"] = -1.0
        elif confidence_source == "SAM_CROP":
            if sam_crop_conf is not None:
                result_dict["confidence"] = float(sam_crop_conf)
            else:
                result_dict["confidence"] = float(result_dict.get("confidence", 0.0))
        elif confidence_source == "SAM_FULL":
            if sam_conf_list is not None and len(sam_conf_list) > 0:
                result_dict["confidence"] = float(np.mean(sam_conf_list))
            else:
                # 全图没有scores时无法算均值：保持默认0
                result_dict["confidence"] = float(result_dict.get("confidence", 0.0))
        else:
            result_dict["confidence"] = float(result_dict.get("confidence", 0.0))

        # =========================================================
        # 最终检查：如果还是 None/空，直接返回失败
        # =========================================================
        if final_mask_bool is None or final_mask_bool.sum() == 0:
            print(f"[跳过] final_mask_bool 为空/全0: {os.path.basename(output_path)}")
            return result_dict

        # =========================================================
        # D) 保存最初的mask为RLE，并且同时羽化mask保存作为编辑需要
        # =========================================================

        # 标准 COCO RLE
        binary_mask = (final_mask_bool.astype(np.uint8) > 0).astype(np.uint8)  # shape (H, W)
        binary_f = np.asfortranarray(binary_mask[:, :, None])  # (H, W, 1), F-contiguous
        rle = maskUtils.encode(binary_f)[0]  # encode 返回 list，这里取第 0 个
        rle["counts"] = rle["counts"].decode("utf-8")
        result_dict["mask_rle"] = rle

        H, W = binary_mask.shape
        # 根据 mask 尺寸自适应一个 feather_px（你可以用之前的函数）
        feather_px = auto_feather_from_mask(
            binary_mask,
            img_h=H,
            img_w=W,
            min_px=3,
            max_px=60,
        )
        # 简单用高斯模糊作为羽化
        ksize = max(3, feather_px * 2 + 1)  # kernel size 要是奇数
        mask_float = cv2.GaussianBlur(
            binary_mask.astype(np.float32),
            (ksize, ksize),
            0
        )
        # 限制在 [0,1]
        mask_float = np.clip(mask_float, 0.0, 1.0)
        result_dict["feather_mask"] = mask_float



        # =========================================================
        # E) mask对比图可视化保存（2x2：raw / final_mask / sam3 / dino）
        # =========================================================

        # --- 0) 原图读取
        img_raw = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        h0, w0 = img_raw.shape[:2]

        # --- 1) 右上：最终 mask 图（灰度显示） ---
        mask_vis = (binary_mask * 255).astype(np.uint8)
        mask_vis_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

        # --- 2) 左下：SAM3 结果图 ---
        # 规则：
        # - 如果 verify 是 "SAM3" 或 "SAM3 GroundingDINO"：显示“候选 masks_bool”，其中被选中的标红，其他用 palette
        # - 如果 verify 是 "DINO crop SAM3" / "DINO boxes crop SAM3" / "MLLM crop SAM3"：显示最终 final_mask_bool（红色）
        sam_panel = img_raw.copy()

        verify_mode = str(result_dict.get("verify", ""))

        # - sam3_cand_masks/sam3_cand_scores：来自 sam3_on_pil 的 **阈值(0.05) + top_k** 候选（用于可视化）
        # - masks_bool/scores_np：在 process_single_image 内又做了 **最大间隔(gap)** 截断后的子集（用于后续决策）
        # 这些变量用于让“header文字颜色”与图中颜色保持一致
        sam_vis_colors = None          # list[(B,G,R)]，长度=N
        sam_vis_chosen_ids = set()     # 被选中的 mask 索引集合
        sam_vis_scores = sam3_cand_scores  # 与 sam3_cand_masks 对齐的 score

        if verify_mode in ["SAM3", "SAM3 GroundingDINO"]:
            # 最终来自“全图 SAM3”时：左下角要显示 *候选* masks
            if sam3_cand_masks is not None and sam3_cand_masks.shape[0] > 0:
                N = int(sam3_cand_masks.shape[0])
                colors = make_palette(N)

                # 选中mask变红：
                # - single：只选 mask0（即 best_mask_idx=0）
                # - all：选 gap 后保留的前 sam3_gap_k 个（用于 union）
                if target_scope == "single":
                    chosen_mask_ids = {0}
                else:
                    k_sel = int(min(max(0, sam3_gap_k), N)) if sam3_gap_k is not None else N
                    chosen_mask_ids = set(range(k_sel))

                sam_vis_chosen_ids = set(chosen_mask_ids)

                # chosen -> 红色；其它保持 palette 颜色
                for i in range(N):
                    if i in chosen_mask_ids:
                        colors[i] = (0, 0, 255)  # red (BGR)

                sam_vis_colors = colors

                sam_panel = overlay_multi_masks_bgr(
                    img_bgr=img_raw,
                    masks_bool=sam3_cand_masks,
                    colors_bgr=colors,
                    alpha=0.45,
                    draw_index=True,
                )
            else:
                # SAM3 没有候选就画最终 mask
                sam_panel = overlay_mask_single_color_bgr(img_raw, final_mask_bool, color_bgr=(0, 0, 255), alpha=0.45)

        else:
            # DINO crop / MLLM crop / MLLM box mask：只显示最终mask（红色）
            sam_panel = overlay_mask_single_color_bgr(img_raw, final_mask_bool, color_bgr=(0, 0, 255), alpha=0.45)

        # --- 3) 右下：DINO 结果图（框可视化） ---
        dino_panel = img_raw.copy()

        # chosen box 的规则：
        # - SINGLE：如果 DINO 有框，则优先 chosen = verify_box_idx(通过验证命中的box)，否则默认 0
        # - ALL：如果 chosen_boxes_for_all 非空，chosen_set = chosen_boxes_for_all；否则若 verify_box_idx 有值，就只选它
        chosen_box_ids = set()

        # 从 verify_box_idx 取一个“命中的 id”
        v_idx = result_dict.get("verify_box_idx", "")
        v_idx_int = None
        try:
            if v_idx != "" and v_idx is not None:
                v_idx_int = int(v_idx)
        except Exception:
            v_idx_int = None

        if dino_found and len(all_bboxes_list) > 0:
            if target_scope == "single":
                if v_idx_int is not None:
                    chosen_box_ids.add(v_idx_int)
                else:
                    chosen_box_ids.add(0)
            else:
                # ALL
                if len(chosen_boxes_for_all) > 0:
                    # 这里用坐标集合匹配
                    chosen_set = {tuple(b) for b in chosen_boxes_for_all}
                else:
                    chosen_set = set()

            # 为每个 DINO box 分配一个独立颜色（按 id 顺序）
            N_box = len(all_bboxes_list)
            dino_colors = make_palette(N_box)  # list of BGR

            # 统一：选中框 -> 红色；未选中框 -> palette 颜色
            CHOSEN_RED = (0, 0, 255)
            if target_scope == "all" and len(chosen_boxes_for_all) > 0:
                for box_data in all_bboxes_list:
                    bid = int(box_data.get("id", 0))
                    box_int = clamp_box_xyxy(list(map(int, box_data["bbox"])), w, h)

                    is_chosen = (tuple(box_int) in chosen_set)
                    color = CHOSEN_RED if is_chosen else dino_colors[bid % N_box]
                    thick = 2 if is_chosen else 1

                    cv2.rectangle(dino_panel, (box_int[0], box_int[1]), (box_int[2], box_int[3]), color, thick)
            else:
                for box_data in all_bboxes_list:
                    bid = int(box_data.get("id", 0))
                    box_int = clamp_box_xyxy(list(map(int, box_data["bbox"])), w, h)

                    is_chosen = (bid in chosen_box_ids)
                    color = CHOSEN_RED if is_chosen else dino_colors[bid % N_box]
                    thick = 2 if is_chosen else 1

                    cv2.rectangle(dino_panel, (box_int[0], box_int[1]), (box_int[2], box_int[3]), color, thick)

        # --- 4) 组装 2x2 面板 ---
        # 左上 raw | 右上 final mask
        top_row = np.hstack([img_raw, mask_vis_bgr])
        # 左下 sam | 右下 dino
        bot_row = np.hstack([sam_panel, dino_panel])
        content_grid = np.vstack([top_row, bot_row])
        total_w_vis = int(content_grid.shape[1])
        total_h_vis = int(content_grid.shape[0])

        # --- 5) Header：文字颜色与面板颜色一致（自适应字号） ---
        title = f"[{target_scope.upper()}] Prompt='{prompt}' | verify={result_dict.get('verify','')} | conf={float(result_dict.get('confidence',0.0)):.4f}"

        lines = []

        # ===== SAM3 文本（与左下 SAM 面板颜色对应） =====
        sam_kept = int(result_dict.get('sam3_kept_masks', 0))
        sam_gap_keep = int(result_dict.get('sam3_gap_keep', 0))
        lines.append((f"SAM3: cand={sam_kept} | gap_keep={sam_gap_keep} | score_thr={0.05:.2f} | topk={int(Top_K)}", (0, 0, 0), 2))

        if sam3_cand_masks is None or sam3_cand_masks.shape[0] == 0:
            lines.append(("SAM3: none", (0, 0, 0), 2))
        else:
            N = int(sam3_cand_masks.shape[0])
            # 单独处理 crop 情况：没有“多个候选 mask”的颜色列表
            if sam_vis_colors is None or len(sam_vis_colors) != N:
                sam_vis_colors = make_palette(N)
                # 仅当来自 SAM3(full) 路径时，按 chosen_ids 涂红；否则就不强行标红
                for i in sam_vis_chosen_ids:
                    if 0 <= int(i) < N:
                        sam_vis_colors[int(i)] = (0, 0, 255)

            # 每个 mask 一行：颜色与左下角 overlay 一致
            show_sam_k = min(N, 20)  # 太多会把 header 撑得很高
            for i in range(show_sam_k):
                score_str = "N/A" if sam_vis_scores is None else f"{float(sam_vis_scores[i]):.4f}"
                tag = "CHOSEN" if i in sam_vis_chosen_ids else "omit"
                color = sam_vis_colors[i]
                thick = 2 if i in sam_vis_chosen_ids else 1
                lines.append((f"[SAM {i}] score={score_str} | {tag}", color, thick))
            if N > show_sam_k:
                lines.append((f"[SAM ...] +{N - show_sam_k} more", (0, 0, 0), 1))

        # ===== DINO 文本（与右下 DINO 面板颜色对应） =====
        if not (dino_found and len(all_bboxes_list) > 0):
            lines.append(("DINO: none", (0, 0, 0), 2))
        else:
            lines.append((f"DINO: total={len(all_bboxes_list)}", (0, 0, 0), 2))
            show_dino_k = min(len(all_bboxes_list), 20)
            for i in range(show_dino_k):
                bd = all_bboxes_list[i]
                bid = int(bd.get("id", i))
                conf = float(bd.get("confidence", 0.0))
                phrase = str(bd.get("phrase", ""))
                is_chosen = (bid in chosen_box_ids)
                # 与右下框颜色一致：chosen 红，其他 palette
                N_box = len(all_bboxes_list)
                dino_colors = make_palette(N_box)
                CHOSEN_RED = (0, 0, 255)
                color = CHOSEN_RED if is_chosen else dino_colors[bid % N_box]
                thick = 2 if is_chosen else 1
                prefix = "CHOSEN" if is_chosen else "ID"
                lines.append((f"[DINO {prefix}:{bid}] conf={conf:.4f} | {phrase}", color, thick))
            if len(all_bboxes_list) > show_dino_k:
                lines.append((f"[DINO ...] +{len(all_bboxes_list) - show_dino_k} more", (0, 0, 0), 1))

        header = build_header_adaptive(
            total_w=total_w_vis,
            title=title,
            lines=lines,
            img_h=total_h_vis,
            img_w=total_w_vis,
        )

        final_vis = np.vstack([header, content_grid])
        cv2.imwrite(output_path, final_vis)


        # =========================================================
        # F) 返回结果
        # =========================================================
        result_dict["success"] = True
       

        return result_dict

    except Exception as e:
        print(f"[异常] 处理 {os.path.basename(output_path)} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return result_dict



