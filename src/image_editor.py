# -*- coding: utf-8 -*-

"""
A comprehensive image editor using OpenCV in Python.

This script implements various image adjustment features similar to those found in
modern photo editing software (like Huawei Gallery). All adjustments are controlled
by a value ranging from -100 to 100, where 0 represents no change.

Required libraries:
- opencv-python: For image processing functions. (pip install opencv-python)
- numpy: For numerical operations, especially on image arrays. (pip install numpy)
"""

import cv2
import numpy as np
import math
import pdb
from typing import Dict

try:
    # Attempt relative imports for when used as a package module
    from .utils import replace_keywords, print_with_color, load_and_resize_image
    from . import config
except ImportError:
    # Fallback to absolute imports for standalone execution
    from utils import replace_keywords, print_with_color, load_and_resize_image
    import config


class ImageEditor:
    """
    A class to encapsulate all image editing functionalities.
    It takes a dictionary of adjustment parameters and applies them to an image.
    """
    
    def __init__(self, image_path: str, resize=True) -> None:
        """
        Initializes the editor with an image.

        Args:
            image_path (str): The path to the image file.
        """
        self.original_image = load_and_resize_image(image_path, resize)
        if self.original_image is None:
            raise FileNotFoundError(f"Could not open or find the image at: {image_path}")
        # Keep a floating point version for precision in calculations
        self.image_float = self.original_image.astype(np.float32) / 255.0
    
    def _map_value(self, value: float, from_min: float, from_max: float, to_min: float, to_max: float) -> float:
        """Helper function to map a value from one range to another."""
        # Clamp value to the source range
        value = max(from_min, min(value, from_max))
        return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min
    
    def _apply_lut(self, image: np.ndarray, lut: np.ndarray) -> np.ndarray:
        """Helper function to apply a lookup table to a BGR image."""
        # Ensure LUT is uint8
        lut_uint8 = np.clip(lut, 0, 255).astype('uint8')
        return cv2.LUT(image, lut_uint8)
    
    def _create_s_curve_lut(self, contrast_value: float) -> np.ndarray:
        """Creates a LUT based on a sigmoidal S-curve for natural contrast."""
        # Map -100 to 100 range to a suitable factor for the curve
        factor = self._map_value(contrast_value, -100, 100, -10, 10)
        lut = np.arange(256, dtype=np.float32)
        # Apply sigmoidal function
        lut = 255.0 / (1.0 + np.exp(-factor * (lut - 128.0) / 255.0))
        return lut
    
    def _create_highlight_shadow_lut(self, value: float, curve_type: str = 'highlight') -> np.ndarray:
        """Creates a LUT to adjust highlights or shadows."""
        lut = np.arange(256, dtype=np.float32)
        # Map -100 to 100 to a strength factor
        strength = self._map_value(value, -100, 100, -0.8, 0.8)
        
        for i in range(256):
            # Normalize pixel value to 0-1 range
            norm_val = i / 255.0
            if curve_type == 'highlight':
                # Affects brighter pixels more
                # When strength > 0, highlights are boosted.
                # When strength < 0, highlights are toned down.
                new_val = norm_val + strength * (1.0 - norm_val) * norm_val
            elif curve_type == 'shadow':
                # Affects darker pixels more
                # When strength > 0, shadows are lifted.
                # When strength < 0, shadows are darkened.
                new_val = norm_val + strength * (1.0 - norm_val ** 2) * (1.0 - norm_val)
            else:  # for 'whites' and 'blacks' which are more aggressive
                if curve_type == 'whites':
                    new_val = norm_val + strength * norm_val
                else:  # blacks
                    new_val = norm_val + strength * (1.0 - norm_val)
            
            lut[i] = new_val * 255.0
        
        return lut
    
    # --- Main Adjustment Functions ---
    
    def adjust_exposure(self, image: np.ndarray, value: float) -> np.ndarray:  # ok
        """曝光 (Exposure)"""
        if value == 0:
            return image
        # Map -100 to 100 -> -1.0 to 1.0
        factor = self._map_value(value, -100, 100, -1.0, 1.0)
        # Additive exposure adjustment
        return image * (2 ** factor)
    
    def adjust_brightness(self, image: np.ndarray, value: float) -> np.ndarray:  # ok
        """亮度 (Brightness)"""
        if value == 0:
            return image
        # Map -100 to 100 -> -1.0 to 1.0 (Brightness is usually more subtle than exposure)
        factor = self._map_value(value, -100, 100, -0.5, 0.5)
        out = image + factor
        # return np.clip(out, 0.0, 1.0)   
        return out
    
    def adjust_contrast(self, image: np.ndarray, value: float) -> np.ndarray:  # ok
        """对比度 (Contrast)"""
        if value == 0:
            return image
        # Map -100 to 100 -> 0.5 to 1.5 (or a wider range if desired)
        factor = self._map_value(value, -100, 100, 0.5, 2.0)
        # Multiplicative contrast adjustment around the midpoint (0.5)
        return 0.5 + factor * (image - 0.5)
    
    def adjust_saturation(self, image_hsv: np.ndarray, value: float) -> np.ndarray:  # ok
        """饱和度 (Saturation)"""
        if value == 0:
            return image_hsv
        # Map -100 to 100 -> 0.0 to 2.0
        factor = self._map_value(value, -100, 100, 0.0, 2.0)
        image_hsv[:, :, 1] *= factor
        # image_hsv[:, :, 1] = np.clip(image_hsv[:, :, 1], 0.0, 1.0)
        return image_hsv
    
    def adjust_vibrance(self, image_hsv: np.ndarray, value: float) -> np.ndarray:  #
        """
        自然饱和度（Vibrance）
        value ∈ [-100, 100]；正值增艳，负值去艳
        兼容两种输入：
            • hsv[...,1] ∈ 0-1  float32
            • hsv[...,1] ∈ 0-255 uint8/float32
        其他通道不变，只改 S
        """
        if value == 0:
            return image_hsv
        
        # ------- 1. 拿出 S、V 并检测量纲 -------
        hsv_f = image_hsv.astype(np.float32)  # 不破坏原数组 dtype
        s = hsv_f[:, :, 1]
        v = hsv_f[:, :, 2]
        
        if s.max() > 1.5:  # 说明是 0-255 标度
            s /= 255.0
            v /= 255.0
            scale_back = 255.0
        else:  # 已是 0-1
            scale_back = 1.0
        
        # ------- 2. 把滑块映射到 [-1, 1] -------
        amount = self._map_value(value, -100, 100, -1.0, 1.0)
        
        # ------- 3. 亮度权重：暗部/高光影响更小 -------
        weight = 1.0 - np.abs(v - 0.5) * 2.0  # V=0.5 时最大，0/1 时为 0
        weight = np.clip(weight, 0.0, 1.0)
        
        # ------- 4. 计算 ΔS（对称公式） -------
        if amount >= 0:
            # 仅增强低饱和，避免高饱和过曝
            delta = amount * (1.0 - s) * weight
        else:
            # 按比例降低
            delta = amount * s
        
        s_new = np.clip(s + delta, 0.0, 1.0)
        
        # ------- 5. 写回并保持原标度 -------
        hsv_f[:, :, 1] = s_new * scale_back
        return hsv_f.astype(image_hsv.dtype)  # 还回调用方原来的 dtype
    
    def adjust_temperature(self, image: np.ndarray, value: float) -> np.ndarray:  # ok
        """色温 (Color Temperature)"""
        if value == 0:
            return image
        # Map -100 (cool) to 100 (warm)
        # This creates two simple linear LUTs for blue and red channels.
        value = self._map_value(value, -100, 100, -50, 50)
        b, g, r = cv2.split(image)
        b = b.astype(np.float32)
        g = g.astype(np.float32)
        r = r.astype(np.float32)
        
        r += value
        g += 0.6 * value
        b -= value
        
        b = np.clip(b, 0, 255).astype(np.uint8)
        g = np.clip(g, 0, 255).astype(np.uint8)
        r = np.clip(r, 0, 255).astype(np.uint8)
        
        return cv2.merge((b, g, r))
    
    def adjust_tint(self, image: np.ndarray, value: float) -> np.ndarray:  # ok
        """色调 (Tint)"""
        if value == 0:
            return image
        # Map -100 (magenta) to 100 (green)
        value = self._map_value(value, -100, 100, -30, 30)
        
        b_channel, g_channel, r_channel = cv2.split(image)
        
        # Add to green channel
        g_channel = cv2.add(g_channel, value)
        
        # To make it more magenta, we can subtract from green
        # or add to red and blue. Adding to green is simpler.
        
        return cv2.merge((b_channel, g_channel, r_channel))
    
    def adjust_sharpen(self, image: np.ndarray, value: float) -> np.ndarray:  # ok
        """锐度 (Sharpness)"""
        if value == 0:
            return image
        
        # Map 0..100 to a strength factor
        strength = self._map_value(value, 0, 100, 0, 1.5)
        
        if value < 0:  # Blur for negative values
            # Map -100..0 to kernel size 3..15 (must be odd)
            ksize = int(self._map_value(abs(value), 0, 100, 3, 15))
            if ksize % 2 == 0: ksize += 1
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        
        # Unsharp masking for sharpening
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return sharpened
    
    def adjust_vignette(self, image: np.ndarray, value: float) -> np.ndarray:
        """暗角 (Vignette)"""
        if value == 0:
            return image
        
        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
        r /= r.max()  # 0~1 归一化半径
        
        # --- 滑块映射 ---
        abs_v = abs(value)
        strength = self._map_value(abs_v, 0, 100, 0.0, 0.9)  # 边缘最大 90% 影响
        k = self._map_value(abs_v, 0, 100, 1.0, 0.5)  # 覆盖系数
        gamma = self._map_value(abs_v, 0, 100, 0.5, 2)  # 过渡软硬
        
        r_scaled = np.clip(r / k, 0, 1) ** gamma  # base = rᵞ
        base = r_scaled  # 方便下面阅读
        
        # --- 输入归一化到 0~1 float ---
        if image.dtype == np.uint8:
            img = image.astype(np.float32) / 255.0
        else:  # 已是 float
            img = image.astype(np.float32).clip(0.0, 1.0)
        
        if value > 0:  # ----- 暗角：乘法衰减 -----
            mask = 1.0 - strength * base
            out = img * mask[..., None]
        else:  # ----- 亮角：向白色插值 -----
            out = img + strength * base[..., None] * (1.0 - img)
        
        # --- 收尾 ---
        if image.dtype == np.uint8:
            out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        else:
            out = np.clip(out, 0.0, 1.0)
        
        return out
    
    def adjust_fade(self, image: np.ndarray, value: float) -> np.ndarray:  # ok
        """褪色 (Fade)"""
        if value == 0:
            return image
        
        # Map 0..100 to fade strength
        strength = self._map_value(value, 0, 100, 0, 0.6)
        
        # Lift the black point
        faded = image + strength * (1.0 - image)
        
        # Slightly reduce contrast
        faded = 0.5 + (1 - strength * 0.5) * (faded - 0.5)
        
        return faded
    
    def adjust_grain(self, image: np.ndarray, value: float) -> np.ndarray:  # ok
        """颗粒 (Grain)"""
        if value <= 0:
            return image
        
        # Map 0..100 to noise standard deviation
        std_dev = self._map_value(value, 0, 100, 0, 0.3)
        
        h, w, c = image.shape
        noise = np.random.randn(h, w, c) * std_dev
        
        return image + noise
    
    def process_image(self, params: Dict[str, float], save_path: str = "") -> np.ndarray:
        """
        Applies a dictionary of adjustments to the image.
        The order of operations is important for a good result.

        Args:
            params (dict): A dictionary where keys are adjustment names
                           and values are in the range -100 to 100.
        """
        # print_with_color(params,"CYAN")
        # pdb.set_trace()
        params = replace_keywords(params, config.REPLACE_TABLE)
        # print_with_color(params,"YELLOW")
        
        # Start with a fresh float copy of the original image
        processed_image = self.image_float.copy()
        
        # 1. Basic tonal adjustments (operate on float image)
        processed_image = self.adjust_exposure(processed_image, params.get('exposure', 0))
        processed_image = self.adjust_brightness(processed_image, params.get('brightness', 0))
        processed_image = self.adjust_contrast(processed_image, params.get('contrast', 0))
        processed_image = self.adjust_fade(processed_image, params.get('fade', 0))
        
        # Clip to 0-1 range after basic adjustments
        processed_image = np.clip(processed_image, 0, 1)
        
        # Convert to uint8 for LUT-based operations
        processed_image_uint8 = (processed_image * 255).astype(np.uint8)
        
        # 2. Advanced tonal adjustments (LUTs on uint8)
        if params.get('natural_contrast', 0) != 0:
            lut = self._create_s_curve_lut(params['natural_contrast'])
            processed_image_uint8 = self._apply_lut(processed_image_uint8, lut)
        if params.get('highlights', 0) != 0:
            lut = self._create_highlight_shadow_lut(params['highlights'], 'highlight')
            processed_image_uint8 = self._apply_lut(processed_image_uint8, lut)
        if params.get('shadows', 0) != 0:
            lut = self._create_highlight_shadow_lut(params['shadows'], 'shadow')
            processed_image_uint8 = self._apply_lut(processed_image_uint8, lut)
        if params.get('whites', 0) != 0:
            lut = self._create_highlight_shadow_lut(params['whites'], 'whites')
            processed_image_uint8 = self._apply_lut(processed_image_uint8, lut)
        if params.get('blacks', 0) != 0:
            lut = self._create_highlight_shadow_lut(params['blacks'], 'blacks')
            processed_image_uint8 = self._apply_lut(processed_image_uint8, lut)
        
        # 3. Color adjustments
        # Convert to HSV for saturation/vibrance
        processed_hsv = cv2.cvtColor(processed_image_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
        processed_hsv[:, :, 1] /= 255.0  # Normalize S channel to 0-1
        
        processed_hsv = self.adjust_saturation(processed_hsv, params.get('saturation', 0))
        processed_hsv[:, :, 1] = np.clip(processed_hsv[:, :, 1], 0.0, 1.0)
        processed_hsv = self.adjust_vibrance(processed_hsv, params.get('vibrance', 0))
        
        # Denormalize S and convert back
        processed_hsv[:, :, 1] *= 255.0
        processed_hsv = np.clip(processed_hsv, 0, 255)
        processed_image_uint8 = cv2.cvtColor(processed_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Temperature and Tint on uint8 BGR
        processed_image_uint8 = self.adjust_temperature(processed_image_uint8, params.get('temperature', 0))
        processed_image_uint8 = self.adjust_tint(processed_image_uint8, params.get('tint', 0))
        
        # 4. Effects
        # Convert back to float for precision
        processed_image = processed_image_uint8.astype(np.float32) / 255.0
        
        # Apply effects that need float precision
        processed_image = self.adjust_grain(processed_image, params.get('grain', 0))
        processed_image = self.adjust_vignette(processed_image, params.get('vignette', 0))
        
        # Clip final result
        processed_image = np.clip(processed_image, 0, 1)
        
        # Sharpen should be applied near the end on a uint8 image
        processed_image_uint8 = (processed_image * 255).astype(np.uint8)
        final_image = self.adjust_sharpen(processed_image_uint8, params.get('sharpness', 0))
        
        if save_path:
            cv2.imwrite(save_path, final_image)  # print_with_color(f"Image saved to {save_path}", "GREEN")

        return final_image


if __name__ == '__main__':
    # --- 使用说明 (HOW TO USE) ---
    try:
        editor = ImageEditor("../sample/1cu88x_1cu88x.jpg")
    except FileNotFoundError as e:
        print(e)
        dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(dummy_img, "Image not found", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Error", dummy_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()
    
    # 所有可调参数
    all_params = ['exposure', 'brightness', 'contrast', 'natural_contrast', 'highlights', 'shadows', 'whites', 'blacks', 'saturation', 'vibrance', 'temperature', 'tint', 'sharpness', 'vignette', 'fade', 'grain']
    
    # 默认参数（全部为0）
    base_params = {k: 0 for k in all_params}
    
    # 输出目录
    import os
    
    output_dir = "../sample/debug_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原图并resize
    h, w = editor.original_image.shape[:2]
    max_height = 800
    if h > max_height:
        scale = max_height / h
        new_w, new_h = int(w * scale), int(h * scale)
        original_resized = cv2.resize(editor.original_image, (new_w, new_h))
    else:
        original_resized = editor.original_image
    
    # 遍历每个参数
    for param in all_params:
        params_neg = base_params.copy()
        params_pos = base_params.copy()
        params_neg[param] = -100
        params_pos[param] = 100
        
        # 处理-100
        img_neg = editor.process_image(params_neg)
        if h > max_height:
            img_neg = cv2.resize(img_neg, (new_w, new_h))
        
        # 原图
        img_orig = original_resized
        
        # 处理+100
        img_pos = editor.process_image(params_pos)
        if h > max_height:
            img_pos = cv2.resize(img_pos, (new_w, new_h))
        
        # 拼接
        comparison = np.hstack([img_neg, img_orig, img_pos])
        
        # 保存
        out_path = os.path.join(output_dir, f"{param}_compare.jpg")
        cv2.imwrite(out_path, comparison)
        print(f"Saved: {out_path}")
    
    print("All debug images saved.")
