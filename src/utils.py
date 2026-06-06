import json
import math
import os.path
import re

import colorama
import cv2
import numpy as np

try:
    # Attempt relative imports for when used as a package module
    from . import config
except ImportError:
    # Fallback to absolute imports for standalone execution
    import config

# init colorama
colorama.init(autoreset=True)


def print_with_color(content, front_color="reset", back_color="reset", style="normal", pad=0, pad_token="="):
    """
    !pip install colorama
    Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
    Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
    Style: DIM, NORMAL, BRIGHT, RESET_ALL
    """
    content = str(content)
    if len(content) + 4 <= pad:
        content = pad_token * ((pad - (len(content) + 2)) // 2) + " " + content + " " + pad_token * (pad - (pad - (len(content) + 2)) // 2 - (len(content) + 2))
    
    content_with_color = getattr(colorama.Fore, front_color.upper()) + getattr(colorama.Back, back_color.upper()) + getattr(colorama.Style, style.upper()) + content
    print(content_with_color)


def extract_json_from_string(response):
    """提取JSON参数：只匹配 ```json ... ``` 包裹的内容"""
    if not isinstance(response, str) or len(response) == 0:
        print_with_color("Warning: 响应为空，无法提取JSON", "RED")
        return {}

    # 正则规则：匹配 ```json 开头、``` 结尾的代码块（忽略中间空格和换行）
    json_pattern = r"```json\s*([\s\S]*?)\s*```"
    match_result = re.search(json_pattern, response)

    if not match_result:
        print_with_color(f"Warning: 未找到 ```json 包裹的参数块\n响应内容：{response[:200]}...", "RED")
        return {}

    try:
        # 解析匹配到的JSON字符串（strip()去除前后空格）
        json_str = match_result.group(1).strip()
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print_with_color(f"Warning: JSON格式错误\n错误详情：{str(e)}\nJSON内容：{json_str[:100]}...", "RED")
        return {}


def extract_code_from_string(response: str) -> dict:
    """
    Extracts the first ArkTS code block from a string. Start with ```ArkTS
    """
    code_pattern = r'```ArkTS\s*([\s\S]*?)\s*```'
    
    code_snippet = {
        "import": "",
        "function": ""
    }
    
    match = re.search(code_pattern, response, re.DOTALL)
    if match:
        code_str = match.group(1).strip()
        for line in code_str.splitlines():
            if line.strip().startswith("import"):
                code_snippet["import"] += line.strip() + "\n"
            else:
                code_snippet["function"] += line.strip() + "\n"
    
    return code_snippet


def extract_cot_from_string(response: str) -> str:
    if not response:
        return ""
    # 分割“```json”，取前面的文本（即思维链）
    cot_part = response.split("```json")[0].strip()
    return cot_part if cot_part else response


def extract_answer_from_string(response: str) -> str:
    answer_pattern = r'<answer>\s*([\s\S]*?)\s*$'
    
    match = re.search(answer_pattern, response, re.DOTALL)
    if match:
        return match.group(1).replace('</answer>', '').strip()
    
    return ""


def extract_score_from_string(string: str) -> float:
    # extract a integer number from the string, possibly from -10 to 10
    score_pattern = r'\s*(-?\d+(\.\d+)?)\s*'
    
    match = re.search(score_pattern, string, re.DOTALL)
    if match:
        return min(max(float(match.group(1).strip()), -10.0), 10)
    
    return -10

def replace_keywords(data: dict | list | str, replacement_map: dict[str, str]) -> dict | list | str:
    patterns = {word: re.compile(r'\b' + re.escape(word) + r'\b') for word in replacement_map.keys()}
    
    def replace_text(text: str) -> str:
        """替换字符串中的关键词"""
        for word, pattern in patterns.items():
            text = pattern.sub(replacement_map[word], text)
        return text
    
    def _traverse(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                new_key = replace_text(key)
                new_value = _traverse(value)
                new_dict[new_key] = new_value
            return new_dict
        elif isinstance(obj, list):
            return [_traverse(item) for item in obj]
        elif isinstance(obj, str):
            return replace_text(obj)
        return obj
    
    return _traverse(data)


def compute_diff_score(image_1, image_2) -> dict:
    if image_1.shape != image_2.shape:
        h, w = image_1.shape[:2]
        image_2 = cv2.resize(image_2, (w, h), interpolation=cv2.INTER_CUBIC)
    
    image_1_norm = image_1.astype(np.float32) / 255.0
    image_2_norm = image_2.astype(np.float32) / 255.0
    
    l1_loss = float(np.mean(np.abs(image_1_norm - image_2_norm)))
    l2_loss = float(np.sqrt(np.mean((image_1_norm - image_2_norm) ** 2)))
    
    return {
        "pixel_L1": l1_loss,
        "pixel_L2": l2_loss,
        "final_score": (l1_loss + l2_loss) / 2.0
    }


def load_and_resize_image(image_path: str, resize=True) -> np.ndarray:
    """
    Load an image from the given path and resize it to the specified dimensions.
    If resize_dim is -1, the image will not be resized.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    if config.IMAGE_RESIZE != -1 and resize:
        w_, h_ = image.shape[1], image.shape[0]
        match config.IMAGE_RESIZE_BASE:
            case "MAX":
                resize_scale = max(w_, h_) / config.IMAGE_RESIZE
            case "MIN":
                resize_scale = min(w_, h_) / config.IMAGE_RESIZE
            case "SQUARE":
                resize_scale = math.sqrt(w_ * h_) / config.IMAGE_RESIZE
            case _:
                raise ValueError(f"Invalid IMAGE_RESIZE_BASE: {config.IMAGE_RESIZE_BASE}")
        w, h = int(w_ / resize_scale), int(h_ / resize_scale)
        image = cv2.resize(image, (w, h))
    
    return image


def merge_images(left_image_path: str, right_image_path: str, merged_image_path: str):
    """
    Merge two images side by side.
    """
    if os.path.exists(merged_image_path):
        return
    
    left_image = load_and_resize_image(left_image_path)
    right_image = load_and_resize_image(right_image_path)
    
    if left_image.shape[0] != right_image.shape[0]:
        # resize higher image to match the height of the lower one
        if left_image.shape[0] > right_image.shape[0]:
            right_image = cv2.resize(right_image, (right_image.shape[1], left_image.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:
            left_image = cv2.resize(left_image, (left_image.shape[1], right_image.shape[0]), interpolation=cv2.INTER_CUBIC)
        if left_image.shape[0] != right_image.shape[0]:
            raise ValueError(f"Images must have the same height to merge side by side. Left image height: {left_image.shape}, Right image height: {right_image.shape}")
    
    merged_image = np.hstack((left_image, right_image))
    
    cv2.imwrite(merged_image_path, merged_image)


def compute_string_similarity(str1: str, str2: str) -> dict:
    """
    Compute the similarity between two strings using ROUGE and BLEU metrics.
    Returns a float value between 0.0 and 1.0, where 1.0 means the strings are identical.
    """
    if not str1 or not str2:
        return {
            "Rouge-1": 0.0,
            "Rouge-2": 0.0,
            "Rouge-L": 0.0,
            "BLEU": 0.0,
            "final_score": 0.0
        }
    
    from rouge import Rouge
    from nltk.translate.bleu_score import sentence_bleu
    rouge = Rouge()
    rouge_scores = rouge.get_scores(str1, str2)[0]
    rouge_score = (rouge_scores['rouge-1']['f'] + rouge_scores['rouge-2']['f'] + rouge_scores['rouge-l']['f']) / 3.0
    bleu_score = sentence_bleu([str1.split()], str2.split())
    return {
        "Rouge-1": rouge_scores['rouge-1']['f'],
        "Rouge-2": rouge_scores['rouge-2']['f'],
        "Rouge-L": rouge_scores['rouge-l']['f'],
        "BLEU": bleu_score,
        "final_score": (rouge_score + bleu_score) / 2.0
    }
