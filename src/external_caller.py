# src/external_caller.py
import requests
import json
import re
import base64
import cv2
import numpy as np
import os
from typing import Dict, Any, Optional, List

EXTERNAL_LLM_BASE_URL = os.getenv("IEA_EXTERNAL_BASE_URL", "")
EXTERNAL_LLM_API_KEY = os.getenv("IEA_EXTERNAL_API_KEY", "")
EXTERNAL_LLM_DEFAULT_MODEL = os.getenv("IEA_EXTERNAL_MODEL", "")
EXTERNAL_LLM_TIMEOUT = int(os.getenv("IEA_EXTERNAL_TIMEOUT", "60"))
EXTERNAL_LLM_MAX_RETRY = int(os.getenv("IEA_EXTERNAL_MAX_RETRY", "3"))
EXTERNAL_LLM_MAX_RESPONSE_TOKENS = int(
    os.getenv("IEA_EXTERNAL_MAX_RESPONSE_TOKENS", "10240")
)


def authorization_header(token: str) -> str:
    if not token:
        return ""
    return token if token.lower().startswith("bearer ") else f"Bearer {token}"

try:
    import external_prompts
    import prompts
except:
    from . import external_prompts
    from . import prompts


def contains_chinese(text: str) -> bool:
    """检查文本是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def encode_image_to_base64(image_path: str) -> str:
    """
    将图片编码为base64字符串
    """
    try:
        with open(image_path, "rb") as image_file:
            # 不需要调整图片大小，直接编码
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return base64_image
    except Exception as e:
        print(f"图片编码失败: {e}")
        raise


def call_external_vlm_api(system_prompt: str, user_prompt: str, image_paths: List[str] = None,
                          model: str = None) -> str:
    """
    调用外部VLM API，使用硬编码的xi-ai配置
    """
    if model is None:
        model = EXTERNAL_LLM_DEFAULT_MODEL
    if not EXTERNAL_LLM_BASE_URL or not EXTERNAL_LLM_API_KEY or not model:
        return ""

    # 构建用户消息内容
    user_content = [{"type": "text", "text": user_prompt.strip()}]

    # 添加图片（最多1张）
    if image_paths:
        for image_path in image_paths[:1]:  # 硬编码限制1张图片
            try:
                base64_image = encode_image_to_base64(image_path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            except Exception as e:
                print(f"添加图片失败 {image_path}: {e}")
                continue

    json_dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": EXTERNAL_LLM_MAX_RESPONSE_TOKENS,
        "temperature": 0,
        "top_p": 1,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": authorization_header(EXTERNAL_LLM_API_KEY),
    }

    # 重试机制
    retry_count = 0
    response_text = ""
    while retry_count < EXTERNAL_LLM_MAX_RETRY:
        try:
            response = requests.post(
                EXTERNAL_LLM_BASE_URL,
                json=json_dict,
                headers=headers,
                timeout=EXTERNAL_LLM_TIMEOUT
            )
            response.raise_for_status()
            response_data = response.json()
            response_text = response_data["choices"][0]["message"]["content"]
            break
        except KeyboardInterrupt:
            raise Exception("用户终止了操作")
        except Exception as e:
            print(f"API调用错误: {str(e)}")
            retry_count += 1
            if retry_count < EXTERNAL_LLM_MAX_RETRY:
                print(f"重试中...({retry_count}/{EXTERNAL_LLM_MAX_RETRY})")

    if not response_text:
        print("达到最大重试次数，API调用失败")
        return ""

    return response_text.strip()


def task_analysis(user_instruction: str, image_paths: List[str] = None) -> Dict[str, Any]:
    """
    分析用户请求，决定是否使用IEA工具还是外部模型
    """
    # 构建系统提示词
    system_prompt = external_prompts.TASK_ANALYSIS_PROMPT.format(
        image_edit_tools=prompts.IMAGE_EDIT_TOOLS,
        user_instruction=user_instruction
    )

    # 用户提示词
    user_prompt = "请分析这个图像编辑请求。"

    try:
        response = call_external_vlm_api(system_prompt, user_prompt, image_paths)

        # 从响应中提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 如果没有代码块，尝试直接解析整个响应
            json_str = response

        result = json.loads(json_str)

        # 验证必需的字段
        required_fields = ['need_editing', 'model_to_use', 'clear_instruction']
        for field in required_fields:
            if field not in result:
                raise ValueError(f"缺少必需字段: {field}")

        return result

    except Exception as e:
        print(f"任务分析失败: {e}")
        # 返回默认结果
        return {
            "need_editing": True,
            "model_to_use": "IEA",
            "clear_instruction": user_instruction
        }


def reflection_summary(original_instruction: str, reasoning_chain: str,
                       tool_operations: str, image_paths: List[str] = None) -> str:
    """
    对编辑过程进行反思和总结
    """
    # 构建系统提示词
    system_prompt = external_prompts.REFLECTION_SUMMARY_PROMPT.format(
        original_instruction=original_instruction,
        reasoning_chain=reasoning_chain,
        tool_operations=tool_operations
    )

    # 用户提示词
    user_prompt = "请提供编辑过程的反思总结。"

    try:
        response = call_external_vlm_api(system_prompt, user_prompt, image_paths)
        return response.strip()

    except Exception as e:
        print(f"反思总结生成失败: {e}")
        return reasoning_chain  # 失败时返回原始思维链


def call_external_image_edit(original_image_path: str, instruction: str) -> tuple:
    """
    调用外部图像生成模型进行复杂编辑

    Args:
        original_image_path: 原始图片路径
        instruction: 清晰的编辑指令

    Returns:
        tuple: (编辑后图片路径, 思维链内容)
    """
    try:
        # 这里调用外部图像生成模型，比如Qwen-Image-Edit
        # 暂时返回模拟数据，后续替换为真实API调用
        edited_image_path = original_image_path.replace('.', '_edited.')

        # 模拟思维链
        reasoning_chain = f"根据用户指令'{instruction}'，使用外部图像生成模型进行复杂编辑处理。"

        # 这里应该调用真实的外部模型API
        # 示例：调用Qwen-Image-Edit或类似模型
        # edited_image = qwen_image_edit_pipeline(original_image_path, instruction)
        # edited_image.save(edited_image_path)

        return edited_image_path, reasoning_chain

    except Exception as e:
        print(f"外部图像编辑失败: {e}")
        # 返回原始图片和错误信息
        return original_image_path, f"外部图像编辑失败: {str(e)}"


if __name__ == "__main__":
    # 测试代码
    test_instruction = "把这张照片调亮一点，颜色更鲜艳些"

    # 测试时需要提供实际的图片路径
    test_image_paths = ["chat_interface/tmp/test1.jpg"]

    result = task_analysis(test_instruction, test_image_paths)
    print("任务分析结果:", json.dumps(result, ensure_ascii=False, indent=2))

    # 测试反思总结
    summary = reflection_summary(
        original_instruction=test_instruction,
        reasoning_chain="分析图片亮度不足，需要增加曝光和饱和度",
        tool_operations="调整曝光+20，饱和度+15",
        image_paths=test_image_paths
    )
    print("反思总结:", summary)
