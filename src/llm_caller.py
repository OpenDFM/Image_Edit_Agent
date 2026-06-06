# === file: llm_caller.py ===
"""Call the LLM API with request and return the response."""
import pdb
import random
import time
import os
from typing import Dict
from typing import Dict, List, Tuple
import requests
import base64
import json
import cv2
import re
from pathlib import Path

# import prompts_pairwise as pairwise_prompts

try:
    # Attempt relative imports for when used as a package module
    from .utils import extract_json_from_string, extract_cot_from_string, extract_score_from_string, print_with_color
    from . import config, prompts
except ImportError:
    # Fallback to absolute imports for standalone execution
    from utils import extract_json_from_string, extract_cot_from_string, extract_score_from_string, print_with_color
    import config, prompts


def call_llm_api(system_prompt: str, user_prompt: str, model: str) -> str:
    model = model if model else config.LLM_DEFAULT_MODEL
    json_dict = {
        "model": model,
        "messages": [{
            "role": "system",
            "content": system_prompt.strip()
        }, {
            "role": "user",
            "content": user_prompt.strip()
        }],
        "max_tokens": config.LLM_MAX_RESPONSE_TOKENS,
        "temperature": 0,
        "top_p": 1,
    }
    header = {
        "Content-Type": "application/json",
        "Authorization": config.LLM_API_KEY,
    }
    i = 0
    while i < config.LLM_MAX_RETRY:
        try:
            response_ = requests.post(
                config.LLM_BASE_URL,
                json=json_dict,
                headers=header,
                timeout=config.LLM_API_TIMEOUT,
            )
            response_ = response_.json()
            response = response_["choices"][0]["message"]["content"]  # print(response)
        except KeyboardInterrupt:
            raise Exception("Terminated by user.")
        except Exception as e:
            try:
                print(e)
                error = response_["error"]["message"].split("(request id:")[0].strip()
                print(error)
                print(response_.json())
            except:
                pass
            i += 1
            print(f"Failed to get response. Retry {i} times.")
        else:
            break
    
    if i >= config.LLM_MAX_RETRY:
        return ""
    else:
        return response


def call_vlm_api(system_prompt: str, user_prompt: str, model: str, image_list: list[str]) -> str:
    model = model if model else config.VLM_DEFAULT_MODEL
    
    user_content = [{
        "type": "text",
        "text": user_prompt.strip()
    }]
    for image_path in image_list[:config.VLM_MAX_IMAGE_NUMBER]:
        try:
            with open(image_path, "rb") as image_file:
                if config.IMAGE_RESIZE > 0:
                    image = cv2.imread(image_path)
                    height, width = image.shape[:2]
                    if height > config.IMAGE_RESIZE or width > config.IMAGE_RESIZE:
                        scale = config.IMAGE_RESIZE / max(height, width)
                        new_size = (int(width * scale), int(height * scale))
                        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
                    _, buffer = cv2.imencode('.jpg', image)
                    base64_image = base64.b64encode(buffer).decode('utf-8')
                else:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return ""
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return ""
    
    json_dict = {
        "model": model,
        "messages": [{
            "role": "system",
            "content": system_prompt.strip()
        }, {
            "role": "user",
            "content": user_content
        }],
        "max_tokens": config.LLM_MAX_RESPONSE_TOKENS,
        "temperature": 0,
        "top_p": 1,
    }
    header = {
        "Content-Type": "application/json",
        "Authorization": config.LLM_API_KEY,
    }
    i = 0
    response_text = ""
    while i < config.LLM_MAX_RETRY:
        try:
            response_ = requests.post(
                config.LLM_BASE_URL,
                json=json_dict,
                headers=header,
                timeout=config.LLM_API_TIMEOUT,
            )
            response_ = response_.json()
            response = response_["choices"][0]["message"]["content"]  # print(response)
            model = response_["model"]
            usage = {
                "prompt_tokens": response_["usage"]["prompt_tokens"],
                "completion_tokens": response_["usage"]["completion_tokens"],
                "total_tokens": response_["usage"]["total_tokens"]
            }
        except KeyboardInterrupt:
            raise Exception("Terminated by user.")
        except Exception as e:
            try:
                print(e)
                error = response_["error"]["message"].split("(request id:")[0].strip()
                print(error)
                print(response_.json())
            except:
                pass
            i += 1
            print(f"Failed to get response. Retry {i} times.")
        else:
            break
    
    if i >= config.LLM_MAX_RETRY:
        return "", "", ""
    
    else:
        return response, model, usage

def call_vlm_api_n(system_prompt: str, user_prompt: str, model: str, image_list: list[str], n: int = 1):
    """调用VLM API，支持n参数返回多个响应"""
    model = model if model else config.VLM_DEFAULT_MODEL

    # 构建用户消息内容
    user_content = [{"type": "text", "text": user_prompt.strip()}]

    # 添加图片（原有代码不变）
    for image_path in image_list[:config.VLM_MAX_IMAGE_NUMBER]:
        try:
            with open(image_path, "rb") as image_file:
                if config.IMAGE_RESIZE > 0:
                    image = cv2.imread(image_path)
                    height, width = image.shape[:2]
                    if height > config.IMAGE_RESIZE or width > config.IMAGE_RESIZE:
                        scale = config.IMAGE_RESIZE / max(height, width)
                        new_size = (int(width * scale), int(height * scale))
                        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
                    _, buffer = cv2.imencode('.jpg', image)
                    base64_image = base64.b64encode(buffer).decode('utf-8')
                else:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return "", "", {}
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return "", "", {}

    # 构建请求JSON，添加n参数
    json_dict = {
        "model": model,
        "messages": [{
            "role": "system",
            "content": system_prompt.strip()
        }, {
            "role": "user",
            "content": user_content
        }],
        "max_tokens": config.LLM_MAX_RESPONSE_TOKENS,
        "temperature": 0.5,
        "top_p": 1,
        "n": n,  # 关键：添加n参数
    }

    header = {
        "Content-Type": "application/json",
        "Authorization": config.LLM_API_KEY,
    }

    i = 0
    response_text = ""
    while i < config.LLM_MAX_RETRY:
        try:
            response_ = requests.post(
                config.LLM_BASE_URL,
                json=json_dict,
                headers=header,
                timeout=config.LLM_API_TIMEOUT,
            )
            response_ = response_.json()

            # 处理多个响应
            if n > 1:
                responses = []
                for choice in response_["choices"]:
                    responses.append(choice["message"]["content"])
                response = responses
            else:
                response = response_["choices"][0]["message"]["content"]

            model = response_["model"]
            usage = {
                "prompt_tokens": response_["usage"]["prompt_tokens"],
                "completion_tokens": response_["usage"]["completion_tokens"],
                "total_tokens": response_["usage"]["total_tokens"]
            }

        except KeyboardInterrupt:
            raise Exception("Terminated by user.")
        except Exception as e:
            print(f"Error: {e}")
            i += 1
            print(f"Failed to get response. Retry {i} times.")
            continue
        else:
            break

    if i >= config.LLM_MAX_RETRY:
        return [], "", {}
    else:
        return response, model, usage

def call_rm_api(system_prompt: str, user_prompt: str, model: str = "") -> str:
    model = model if model else config.RM_DEFAULT_MODEL
    json_dict = {
        "model": model,
        "messages": [{
            "role": "system",
            "content": system_prompt.strip()
        }, {
            "role": "user",
            "content": user_prompt.strip()
        }],
        "max_tokens": config.RM_MAX_RESPONSE_TOKENS,
        "temperature": 1,
        "top_p": 1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
    }
    header = {
        "Content-Type": "application/json",
        "Authorization": config.RM_API_KEY,
    }
    i = 0
    while i < config.RM_MAX_RETRY:
        try:
            response_ = requests.post(
                config.RM_BASE_URL,
                json=json_dict,
                headers=header,
                timeout=config.LLM_API_TIMEOUT,
            )
            response_ = response_.json()
            response = response_["choices"][0]["message"]["content"]  # print(response)
        except KeyboardInterrupt:
            raise Exception("Terminated by user.")
        except Exception as e:
            try:
                print(e)
                error = response_["error"]["message"].split("(request id:")[0].strip()
                print(error)
                print(response_.json())
            except:
                pass
            i += 1
            print(f"Failed to get response. Retry {i} times.")
        else:
            break
    
    if i >= config.LLM_MAX_RETRY:
        return ""
    else:
        return response


def predict_operations_gier(origin_image_path: str, reference_image_path: str, instruction: str = "", inst_type: int = -1, cot: int = 1, use_ref_op: list[str] = [], log_file: str = None):
    """
    Calls the VLM to determine image editing adjustments.
    """
    system_prompt = (prompts.IMAGE_ANALYSIS_SYS_PROMPT.replace("{image_edit_tools}", prompts.IMAGE_EDIT_TOOLS))
    
    if use_ref_op:
        operators_str = ", ".join(use_ref_op)
        constraint_prompt = prompts.OPERATOR_CONSTRAINT_TEMPLATE[1].replace('{operators}', operators_str)
    else:
        constraint_prompt = prompts.OPERATOR_CONSTRAINT_TEMPLATE[0]
    
    user_prompt = (prompts.IMAGE_ANALYSIS_USER_PROMPT.replace("{instruction_template}", prompts.IMAGE_ANALYSIS_USER_INSTRUCTION_TEMPLATE[inst_type]).replace("{instruction}", instruction).replace("{operator_constraint}", constraint_prompt).replace("{chain_of_thought}", prompts.IMAGE_ANALYSIS_COT_TEMPLATE[cot]))
    
    image_list = [origin_image_path, reference_image_path]
    
    i = 0
    while i < config.LLM_MAX_RETRY:
        response, model, usage = call_vlm_api(system_prompt=system_prompt, user_prompt=user_prompt, model=config.VLM_DEFAULT_MODEL, image_list=image_list)
        
        if log_file:
            with open(log_file, 'a') as log_f:
                log_f.write(f"{' System Prompt ':#^100}\n\n{system_prompt.strip()}\n\n")
                log_f.write(f"{' User Prompt ':#^100}\n\n{user_prompt.strip()}\n\n")
                log_f.write(f"{' Input Image Path ':#^100}\n\n{image_list}\n\n")
                log_f.write(f"{' Response ':#^100}\n\n{response.strip()}\n\n")
        
        if response:
            extracted_json = extract_json_from_string(response)
            extracted_cot = extract_cot_from_string(response)
            if extracted_json:
                return extracted_json, extracted_cot, model, usage
            else:
                print_with_color("Warning: Could not extract JSON from LLM response.")
                print_with_color(f"Response was: {response}")
        i += 1
        print(f"Failed to extract response. Retry {i} times.")
    return {}, "", "", {}


def predict_operations_gier_reflect(origin_image_path: str, reference_image_path: str, best_edit_image_path: str, instruction: str, best_params: dict, inst_type: int = -1, cot: int = 1, use_ref_op: list[str] = [], log_file: str = None):
    """
    Calls the VLM to determine image editing adjustments.
    """
    system_prompt = (prompts.IMAGE_ANALYSIS_SYS_PROMPT.replace("{image_edit_tools}", prompts.IMAGE_EDIT_TOOLS))
    
    if use_ref_op:
        operators_str = ", ".join(use_ref_op)
        constraint_prompt = prompts.OPERATOR_CONSTRAINT_TEMPLATE[2]
    else:
        constraint_prompt = prompts.OPERATOR_CONSTRAINT_TEMPLATE[0]
    
    user_prompt = (
        prompts.IMAGE_REFLECT_USER_PROMPT.replace("{instruction_template}", prompts.IMAGE_ANALYSIS_USER_INSTRUCTION_TEMPLATE[inst_type]).replace("{instruction}", instruction).replace("{operator_constraint}", constraint_prompt).replace("{chain_of_thought}", prompts.IMAGE_REFLECT_COT_TEMPLATE[cot]).replace("{best_params}", json.dumps(best_params, indent=4)))
    
    image_list = [origin_image_path, reference_image_path, best_edit_image_path]
    
    i = 0
    while i < config.LLM_MAX_RETRY:
        response, model, usage = call_vlm_api(system_prompt=system_prompt, user_prompt=user_prompt, model=config.VLM_DEFAULT_MODEL, image_list=image_list)
        
        if log_file:
            with open(log_file, 'a') as log_f:
                log_f.write(f"{' System Prompt ':#^100}\n\n{system_prompt.strip()}\n\n")
                log_f.write(f"{' User Prompt ':#^100}\n\n{user_prompt.strip()}\n\n")
                log_f.write(f"{' Input Image Path ':#^100}\n\n{image_list}\n\n")
                log_f.write(f"{' Response ':#^100}\n\n{response.strip()}\n\n")
        
        if response:
            extracted_json = extract_json_from_string(response)
            extracted_cot = extract_cot_from_string(response)
            if extracted_json:
                return extracted_json, extracted_cot, model, usage
            else:
                print_with_color("Warning: Could not extract JSON from LLM response.")
                print_with_color(f"Response was: {response}")
        i += 1
        print(f"Failed to extract response. Retry {i} times.")
    return {}, "", "", {}


def predict_operations_gier_eval(origin_image_path: str, reference_image_path: str = "", instruction: str = "", inst_type: int = -1, cot: int = 1, use_ref_op: list[str] = [], log_file: str = None):
    """
    Calls the VLM to determine image editing adjustments.
    """
    system_prompt = prompts.SFT_SYSTEM_PROMPT
    
    if inst_type == -1 or not instruction:
        user_prompt = prompts.SFT_USER_PROMPT
    else:
        user_prompt = prompts.SFT_USER_PROMPT_INST.replace("{instruction}", instruction)
    
    image_list = [origin_image_path]
    
    i = 0
    while i < config.LLM_MAX_RETRY:
        response, model, usage = call_vlm_api(system_prompt=system_prompt, user_prompt=user_prompt, model=config.VLM_DEFAULT_MODEL, image_list=image_list)
        
        if log_file:
            with open(log_file, 'a') as log_f:
                log_f.write(f"{' System Prompt ':#^100}\n\n{system_prompt.strip()}\n\n")
                log_f.write(f"{' User Prompt ':#^100}\n\n{user_prompt.strip()}\n\n")
                log_f.write(f"{' Input Image Path ':#^100}\n\n{image_list}\n\n")
                log_f.write(f"{' Response ':#^100}\n\n{response.strip()}\n\n")
        
        if response:
            extracted_json = extract_json_from_string(response)
            extracted_cot = extract_cot_from_string(response)
            if extracted_json:
                return extracted_json, extracted_cot, model, usage
            else:
                print_with_color("Warning: Could not extract JSON from LLM response.")
                print_with_color(f"Response was: {response}")
        i += 1
        print(f"Failed to extract response. Retry {i} times.")
    return {}, "", "", {}


def predict_operations_gier_preference(origin_image_path: str, reference_image_path: str, instruction: str, inst_type: int = -1, cot: int = 1, use_ref_op: list[str] = [], log_file: str = None, preference: str = None, pref_mode: int = 0):
    """
    Calls the VLM to determine image editing adjustments.

    pref_mode:
      0 = direct mode (preference directly added)
      1 = two-stage mode (preference not used in first stage)
      2 = priority mode (preference integrated into CoT)
    """
    # Handle preference information
    preference_info = ""
    cot_pref_analysis = ""
    
    if preference:
        if pref_mode == 0:  # Direct mode
            preference_info = prompts.PREFERENCE_TEMPLATE["direct"].format(preference=preference)
        elif pref_mode == 2:  # Priority mode
            preference_info = prompts.PREFERENCE_TEMPLATE["priority"].format(preference=preference)
            cot_pref_analysis = prompts.PREFERENCE_ANALYSIS_TEMPLATE.format(preference=preference)
    
    # Get instruction template
    inst_template = prompts.IMAGE_ANALYSIS_USER_INSTRUCTION_TEMPLATE[inst_type]
    
    # For two-stage mode, remove preference info in first stage
    if pref_mode == 1:
        inst_template = inst_template.replace("{preference_info}", "")
    else:
        # Format instruction template with preference info
        format_kwargs = {}
        if "{instruction}" in inst_template:
            format_kwargs["instruction"] = instruction
        if "{preference_info}" in inst_template:
            format_kwargs["preference_info"] = preference_info
        inst_template = inst_template.format(**format_kwargs)
    
    # Handle CoT template
    if cot == 3:  # Priority CoT template
        cot_prompt = prompts.IMAGE_ANALYSIS_COT_TEMPLATE[cot].format(preference_analysis=cot_pref_analysis)
    else:
        cot_prompt = prompts.IMAGE_ANALYSIS_COT_TEMPLATE[cot]
    
    system_prompt = (prompts.IMAGE_ANALYSIS_SYS_PROMPT.replace("{image_edit_tools}", prompts.IMAGE_EDIT_TOOLS))
    
    if use_ref_op:
        operators_str = ", ".join(use_ref_op)
        constraint_prompt = prompts.OPERATOR_CONSTRAINT_TEMPLATE[1].replace('{operators}', operators_str)
    else:
        constraint_prompt = prompts.OPERATOR_CONSTRAINT_TEMPLATE[0]
    
    user_prompt = prompts.IMAGE_ANALYSIS_USER_PROMPT.replace("{instruction_template}", inst_template).replace("{operator_constraint}", constraint_prompt).replace("{chain_of_thought}", cot_prompt)
    
    image_list = [origin_image_path, reference_image_path]
    
    i = 0
    while i < config.LLM_MAX_RETRY:
        response, model, usage = call_vlm_api(system_prompt=system_prompt, user_prompt=user_prompt, model=config.VLM_DEFAULT_MODEL, image_list=image_list)
        
        if log_file:
            with open(log_file, 'a') as log_f:
                log_f.write(f"{' System Prompt ':#^100}\n\n{system_prompt.strip()}\n\n")
                log_f.write(f"{' User Prompt ':#^100}\n\n{user_prompt.strip()}\n\n")
                log_f.write(f"{' Input Image Path ':#^100}\n\n{image_list}\n\n")
                log_f.write(f"{' Response ':#^100}\n\n{response.strip()}\n\n")
        
        if response:
            extracted_json = extract_json_from_string(response)
            extracted_cot = extract_cot_from_string(response)
            if extracted_json:
                return extracted_json, extracted_cot, model, usage
            else:
                print_with_color("Warning: Could not extract JSON from LLM response.")
                print_with_color(f"Response was: {response}")
        i += 1
        print(f"Failed to extract response. Retry {i} times.")
    return {}, "", "", {}

def predict_operations_gier_reflect_preference(origin_image_path: str, reference_image_path: str,
                                               best_edit_image_path: str, instruction: str, best_params: dict,
                                               inst_type: int = -1, cot: int = 1, use_ref_op: list[str] = [],
                                               log_file: str = None, preference: str = None):
    """
    使用两阶段编辑模式处理多轮编辑请求
    """
    # 系统提示词保持不变
    system_prompt = (prompts.IMAGE_ANALYSIS_SYS_PROMPT.replace("{image_edit_tools}", prompts.IMAGE_EDIT_TOOLS))

    # 确保 use_ref_op 总是列表类型
    if use_ref_op is None:
        use_ref_op = []
    elif not isinstance(use_ref_op, list):
        use_ref_op = []

    # 现在可以安全使用
    if use_ref_op:
        operators_str = ", ".join(use_ref_op)
        constraint_prompt = prompts.OPERATOR_CONSTRAINT_TEMPLATE[2]
    else:
        constraint_prompt = prompts.OPERATOR_CONSTRAINT_TEMPLATE[0]

    # 获取指令模板并添加偏好（如果提供）
    inst_template = prompts.IMAGE_ANALYSIS_USER_INSTRUCTION_TEMPLATE[inst_type]
    if preference:
        preference_info = prompts.PREFERENCE_TEMPLATE["direct"].format(preference=preference)
    else:
        preference_info = ""

    # 格式化指令模板
    format_kwargs = {}
    if "{instruction}" in inst_template:
        format_kwargs["instruction"] = instruction
    if "{preference_info}" in inst_template:
        format_kwargs["preference_info"] = preference_info
    inst_template = inst_template.format(**format_kwargs)

    # 准备思维链模板
    cot_prompt = prompts.IMAGE_REFLECT_COT_TEMPLATE[cot]

    # 准备用户提示词
    user_prompt = prompts.IMAGE_REFLECT_USER_PROMPT.replace(
        "{instruction_template}", inst_template
    ).replace(
        "{operator_constraint}", constraint_prompt
    ).replace(
        "{chain_of_thought}", cot_prompt
    ).replace(
        "{best_params}", json.dumps(best_params, indent=4)
    )

    # 创建图片列表（只包含存在的图片）
    image_list = []
    if origin_image_path and os.path.exists(origin_image_path):
        image_list.append(origin_image_path)
    if reference_image_path and os.path.exists(reference_image_path):
        image_list.append(reference_image_path)
    if best_edit_image_path and os.path.exists(best_edit_image_path):
        image_list.append(best_edit_image_path)

    # 最大重试次数
    max_retry = config.LLM_MAX_RETRY
    for i in range(max_retry):
        try:
            # 调用视觉语言模型API
            response, model, usage = call_vlm_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=config.VLM_DEFAULT_MODEL,
                image_list=image_list
            )

            # 日志记录
            if log_file:
                with open(log_file, 'a') as log_f:
                    log_f.write(f"{' System Prompt ':#^100}\n\n{system_prompt.strip()}\n\n")
                    log_f.write(f"{' User Prompt ':#^100}\n\n{user_prompt.strip()}\n\n")
                    log_f.write(f"{' Input Image Path ':#^100}\n\n{image_list}\n\n")
                    log_f.write(f"{' Response ':#^100}\n\n{response.strip()}\n\n")

            # 处理响应
            if response:
                extracted_json = extract_json_from_string(response)
                extracted_cot = extract_cot_from_string(response)
                if extracted_json:
                    return extracted_json, extracted_cot, model, usage
                else:
                    print_with_color("警告: 无法从LLM响应中提取JSON")
                    print_with_color(f"响应内容: {response}")
        except Exception as e:
            print_with_color(f"API调用出错: {str(e)}")

        # 重试等待
        time.sleep(1)
        print(f"重试中... ({i + 1}/{max_retry})")

    return {}, "", "", {}

def predict_operations_gier_eval_preference(origin_image_path: str, reference_image_path: str, instruction: str, inst_type: int = -1, cot: int = 1, use_ref_op: list[str] = [], log_file: str = None, preference: str = None):
    """
    Calls the VLM to determine image editing adjustments.
    """
    system_prompt = prompts.SFT_SYSTEM_PROMPT
    
    user_prompt = prompts.SFT_USER_PROMPT.replace("{instruction}", instruction)
    
    if preference:
        preference_info = prompts.PREFERENCE_TEMPLATE["direct"].format(preference=preference)
        user_prompt += f"\n\nAdditionally, the user has the following preferences: {preference}"
    
    image_list = [origin_image_path]
    
    i = 0
    while i < config.LLM_MAX_RETRY:
        response, model, usage = call_vlm_api(system_prompt=system_prompt, user_prompt=user_prompt, model=config.VLM_DEFAULT_MODEL, image_list=image_list)
        
        if log_file:
            with open(log_file, 'a') as log_f:
                log_f.write(f"{' System Prompt ':#^100}\n\n{system_prompt.strip()}\n\n")
                log_f.write(f"{' User Prompt ':#^100}\n\n{user_prompt.strip()}\n\n")
                log_f.write(f"{' Input Image Path ':#^100}\n\n{image_list}\n\n")
                log_f.write(f"{' Response ':#^100}\n\n{response.strip()}\n\n")
        
        if response:
            extracted_json = extract_json_from_string(response)
            extracted_cot = extract_cot_from_string(response)
            if extracted_json:
                return extracted_json, extracted_cot, model, usage
            else:
                print_with_color("Warning: Could not extract JSON from LLM response.")
                print_with_color(f"Response was: {response}")
        i += 1
        print(f"Failed to extract response. Retry {i} times.")
    return {}, "", "", {}


def get_summary_reward(reference_instruction: str, predicted_instruction: str):
    """
    Calls the LLM to get a summary reward for the predicted instruction.
    """
    system_prompt = prompts.SUMMARY_REWARD_SYSTEM_PROMPT
    user_prompt = prompts.SUMMARY_REWARD_USER_PROMPT.replace("{reference}", reference_instruction).replace("{prediction}", predicted_instruction)
    
    response_list = []
    
    for i in range(config.RM_VOTE_TIMES):
        response = call_rm_api(system_prompt=system_prompt, user_prompt=user_prompt)
        if response:
            extracted_score = extract_score_from_string(response)
            response_list.append(extracted_score)
    
    if response_list:
        avg_score = sum(response_list) / len(response_list)
        return avg_score
    return -10


def generate_amateur_instruction(params: Dict[str, int], expert_instruction: str):
    """
    Calls the LLM to generate an amateur instruction based on the expert instruction.
    """
    system_prompt = prompts.GENERATE_INSTRUCTION_SYSTEM_PROMPT.replace("{image_edit_tools}", prompts.IMAGE_EDIT_TOOLS)
    user_prompt = prompts.GENERATE_INSTRUCTION_USER_PROMPT.replace("{params}", json.dumps(params, indent=4)).replace("{expert_instruction}", expert_instruction)
    
    response = call_llm_api(system_prompt=system_prompt, user_prompt=user_prompt, model=config.RM_DEFAULT_MODEL)
    
    if response:
        amateur_instruction = response.strip().replace('\n', ' ')
        return amateur_instruction
    return ""


def generate_second_instruction(old_params, new_params, old_instruction, new_instruction):
    system_prompt = prompts.REFINE_SUMMARY_SYSTEM_PROMPT.replace("{image_edit_tools}", prompts.IMAGE_EDIT_TOOLS)
    user_prompt = prompts.REFINE_SUMMARY_USER_PROMPT.replace("{old_params}", json.dumps(old_params, indent=4)).replace("{new_params}", json.dumps(new_params, indent=4)).replace("{old_instruction}", old_instruction).replace("{new_instruction}", new_instruction)
    
    response = call_llm_api(system_prompt=system_prompt, user_prompt=user_prompt, model=config.RM_DEFAULT_MODEL)
    
    if response:
        refine_instruction = response.strip().replace('\n', ' ')
        return refine_instruction
    return ""


def evaluate_intent_following(
        original_image_path: str,
        edited_image_path: str,
        instruction: str
) -> Tuple[float, List[float]]:
    """
    评估意图遵循程度 - 使用n=5参数采样5次
    返回：(平均分, 所有分数列表)
    """
    system_prompt = prompts.INTENT_FOLLOWING_SYSTEM_PROMPT
    user_prompt = prompts.INTENT_FOLLOWING_USER_PROMPT.format(instruction=instruction)

    # 准备图片列表
    image_list = [original_image_path, edited_image_path]

    all_scores = []

    try:
        # 调用VLM API，使用n=5获取5个独立响应
        responses, model, usage = call_vlm_api_n(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=config.VLM_DEFAULT_MODEL,
            image_list=image_list,
            n= 5
        )
        print(responses)

        # 处理多个响应
        if isinstance(responses, list):
            for i, response in enumerate(responses):
                try:
                    score = extract_score_from_string(response)
                    if score is not None:
                        all_scores.append(score)
                        print(f"[SAMPLE {i + 1}/5] Intent score: {score}")
                except Exception as e:
                    print(f"[WARN] Failed to extract score from sample {i + 1}: {e}")
        else:
            # 如果返回的不是列表，只得到一个响应
            score = extract_score_from_string(responses)
            if score is not None:
                all_scores.append(score)
                print(f"[SAMPLE 1/1] Intent score: {score}")

        # 计算平均分
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            print(f"[RESULT] Intent following: {avg_score:.2f} (samples: {all_scores})")
            return avg_score, all_scores
        else:
            print("[ERROR] No valid scores extracted")
            return -10.0, []

    except Exception as e:
        print(f"[ERROR] Intent following evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return -10.0, []


def evaluate_target_gap(
        original_image_path: str,
        edited_image_path: str,
        instruction: str,
        target_image_path: str
) -> Tuple[float, List[float]]:
    """
    评估目标差距 - 使用n=5参数采样5次
    返回：(平均分, 所有分数列表)
    """
    system_prompt = prompts.TARGET_GAP_SYSTEM_PROMPT
    user_prompt = prompts.TARGET_GAP_USER_PROMPT.format(instruction=instruction)

    # 准备图片列表
    image_list = [original_image_path, edited_image_path, target_image_path]

    all_scores = []

    try:
        # 调用VLM API，使用n=5获取5个独立响应
        responses, model, usage = call_vlm_api_n(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=config.VLM_DEFAULT_MODEL,
            image_list=image_list,
            n=5
        )

        # 处理多个响应
        if isinstance(responses, list):
            for i, response in enumerate(responses):
                try:
                    score = extract_score_from_string(response)
                    if score is not None:
                        all_scores.append(score)
                        print(f"[SAMPLE {i + 1}/5] Target gap score: {score}")
                except Exception as e:
                    print(f"[WARN] Failed to extract score from sample {i + 1}: {e}")
        else:
            # 如果返回的不是列表，只得到一个响应
            score = extract_score_from_string(responses)
            if score is not None:
                all_scores.append(score)
                print(f"[SAMPLE 1/1] Target gap score: {score}")

        # 计算平均分
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            print(f"[RESULT] Target gap: {avg_score:.2f} (samples: {all_scores})")
            return avg_score, all_scores
        else:
            print("[ERROR] No valid scores extracted")
            return -10.0, []

    except Exception as e:
        print(f"[ERROR] Target gap evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return -10.0, []


def evaluate_image_quality(
        original_image_path: str,
        edited_image_path: str,
        instruction: str
) -> Tuple[float, List[float]]:
    """
    评估图像质量 - 使用n=10参数采样10次
    返回：(平均分, 所有分数列表)
    """
    system_prompt = prompts.IMAGE_QUALITY_SYSTEM_PROMPT
    user_prompt = prompts.IMAGE_QUALITY_USER_PROMPT.format(instruction=instruction)

    # 准备图片列表 - 现在只需要2张图片（原始图片和编辑后的图片）
    image_list = [original_image_path, edited_image_path]

    all_scores = []

    try:
        # 调用VLM API，使用n=10获取10个独立响应
        responses, model, usage = call_vlm_api_n(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=config.VLM_DEFAULT_MODEL,
            image_list=image_list,
            n=5
        )

        # 处理多个响应
        if isinstance(responses, list):
            for i, response in enumerate(responses):
                try:
                    score = extract_score_from_string(response)
                    if score is not None:
                        all_scores.append(score)
                        print(f"[SAMPLE {i + 1}/10] Image quality score: {score}")
                except Exception as e:
                    print(f"[WARN] Failed to extract score from sample {i + 1}: {e}")
        else:
            # 如果返回的不是列表，只得到一个响应
            score = extract_score_from_string(responses)
            if score is not None:
                all_scores.append(score)
                print(f"[SAMPLE 1/1] Image quality score: {score}")

        # 计算平均分
        if all_scores:
            avg_score = sum(all_scores) / len(all_scores)
            print(f"[RESULT] Image quality: {avg_score:.2f} (samples: {all_scores})")
            return avg_score, all_scores
        else:
            print("[ERROR] No valid scores extracted")
            return -10.0, []

    except Exception as e:
        print(f"[ERROR] Image quality evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return -10.0, []


def compare_intent_following(
        original_image_path: str,
        edited_image_a: str,
        edited_image_b: str,
        instruction: str
) -> Tuple[Dict, List[Dict]]:
    """
    两两比较意图遵循程度 - 使用n=5参数采样5次，每次交换顺序
    返回：(汇总结果, 所有比较样本列表)
    """
    system_prompt = pairwise_prompts.INTENT_FOLLOWING_COMPARE_SYSTEM_PROMPT
    user_prompt = pairwise_prompts.INTENT_FOLLOWING_COMPARE_USER_PROMPT.format(instruction=instruction)

    all_comparisons = []

    # 第一轮：A vs B
    image_list = [original_image_path, edited_image_a, edited_image_b]

    try:
        # 调用API，n=5获取5个独立响应
        responses, model, usage = call_vlm_api_n(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=config.VLM_DEFAULT_MODEL,
            image_list=image_list,
            n=5
        )
        print('///')
        print(original_image_path, edited_image_a, edited_image_b)
        print(responses)
        print('///')

        # 处理响应
        for i, response in enumerate(responses):
            try:
                result = parse_comparison_result(response, "A", "B", i)
                all_comparisons.append(result)
            except Exception as e:
                print(f"[WARN] Failed to parse comparison sample {i + 1}: {e}")
                all_comparisons.append({
                    "sample_id": f"round1_sample{i + 1}",
                    "winner": "error",
                    "reason": "parsing failed",
                    "order": "A_vs_B"
                })
    except Exception as e:
        print(f"[ERROR] First round comparison failed: {e}")

    # 第二轮：B vs A (交换顺序)
    image_list = [original_image_path, edited_image_b, edited_image_a]
    user_prompt_swapped = pairwise_prompts.INTENT_FOLLOWING_COMPARE_USER_PROMPT.format(instruction=instruction)

    try:
        responses, model, usage = call_vlm_api_n(
            system_prompt=system_prompt,
            user_prompt=user_prompt_swapped,
            model=config.VLM_DEFAULT_MODEL,
            image_list=image_list,
            n=5
        )

        # 处理响应，交换winner标识
        for i, response in enumerate(responses):
            try:
                result = parse_comparison_result(response, "B", "A", i + 5)
                # 交换回来：如果winner是"B"，表示原来的B（现在的A）赢了
                if result["winner"] == "A":
                    result["winner"] = "B"
                elif result["winner"] == "B":
                    result["winner"] = "A"
                result["order"] = "B_vs_A"
                all_comparisons.append(result)
            except Exception as e:
                print(f"[WARN] Failed to parse comparison sample {i + 6}: {e}")
                all_comparisons.append({
                    "sample_id": f"round2_sample{i + 1}",
                    "winner": "error",
                    "reason": "parsing failed",
                    "order": "B_vs_A"
                })
    except Exception as e:
        print(f"[ERROR] Second round comparison failed: {e}")

    # 汇总结果
    summary = aggregate_comparison_results(all_comparisons)

    return summary, all_comparisons


def compare_target_gap(
        original_image_path: str,
        edited_image_a: str,
        edited_image_b: str,
        target_image_path: str,
        instruction: str
) -> Tuple[Dict, List[Dict]]:
    """
    两两比较目标差距 - 使用n=5参数采样5次，每次交换顺序
    返回：(汇总结果, 所有比较样本列表)
    """
    system_prompt = pairwise_prompts.TARGET_GAP_COMPARE_SYSTEM_PROMPT
    user_prompt = pairwise_prompts.TARGET_GAP_COMPARE_USER_PROMPT.format(instruction=instruction)

    all_comparisons = []

    # 第一轮：A vs B
    image_list = [original_image_path, target_image_path, edited_image_a, edited_image_b]

    try:
        responses, model, usage = call_vlm_api_n(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=config.VLM_DEFAULT_MODEL,
            image_list=image_list,
            n=5
        )

        for i, response in enumerate(responses):
            try:
                result = parse_comparison_result(response, "A", "B", i)
                all_comparisons.append(result)
            except Exception as e:
                print(f"[WARN] Failed to parse comparison sample {i + 1}: {e}")
                all_comparisons.append({
                    "sample_id": f"round1_sample{i + 1}",
                    "winner": "error",
                    "reason": "parsing failed",
                    "order": "A_vs_B"
                })
    except Exception as e:
        print(f"[ERROR] First round comparison failed: {e}")

    # 第二轮：B vs A
    image_list = [original_image_path, target_image_path, edited_image_b, edited_image_a]

    try:
        responses, model, usage = call_vlm_api_n(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=config.VLM_DEFAULT_MODEL,
            image_list=image_list,
            n=5
        )

        for i, response in enumerate(responses):
            try:
                result = parse_comparison_result(response, "B", "A", i + 5)
                if result["winner"] == "A":
                    result["winner"] = "B"
                elif result["winner"] == "B":
                    result["winner"] = "A"
                result["order"] = "B_vs_A"
                all_comparisons.append(result)
            except Exception as e:
                print(f"[WARN] Failed to parse comparison sample {i + 6}: {e}")
                all_comparisons.append({
                    "sample_id": f"round2_sample{i + 1}",
                    "winner": "error",
                    "reason": "parsing failed",
                    "order": "B_vs_A"
                })
    except Exception as e:
        print(f"[ERROR] Second round comparison failed: {e}")

    summary = aggregate_comparison_results(all_comparisons)
    return summary, all_comparisons


def compare_image_quality(
        original_image_path: str,
        edited_image_a: str,
        edited_image_b: str,
        instruction: str
) -> Tuple[Dict, List[Dict]]:
    """
    两两比较图像质量 - 使用n=5参数采样5次，每次交换顺序
    返回：(汇总结果, 所有比较样本列表)
    """
    system_prompt = pairwise_prompts.IMAGE_QUALITY_COMPARE_SYSTEM_PROMPT
    user_prompt = pairwise_prompts.IMAGE_QUALITY_COMPARE_USER_PROMPT.format(instruction=instruction)

    all_comparisons = []

    # 第一轮：A vs B
    image_list = [original_image_path, edited_image_a, edited_image_b]

    try:
        responses, model, usage = call_vlm_api_n(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=config.VLM_DEFAULT_MODEL,
            image_list=image_list,
            n=5
        )

        for i, response in enumerate(responses):
            try:
                result = parse_comparison_result(response, "A", "B", i)
                all_comparisons.append(result)
            except Exception as e:
                print(f"[WARN] Failed to parse comparison sample {i + 1}: {e}")
                all_comparisons.append({
                    "sample_id": f"round1_sample{i + 1}",
                    "winner": "error",
                    "reason": "parsing failed",
                    "order": "A_vs_B"
                })
    except Exception as e:
        print(f"[ERROR] First round comparison failed: {e}")

    # 第二轮：B vs A
    image_list = [original_image_path, edited_image_b, edited_image_a]

    try:
        responses, model, usage = call_vlm_api_n(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=config.VLM_DEFAULT_MODEL,
            image_list=image_list,
            n=5
        )

        for i, response in enumerate(responses):
            try:
                result = parse_comparison_result(response, "B", "A", i + 5)
                if result["winner"] == "A":
                    result["winner"] = "B"
                elif result["winner"] == "B":
                    result["winner"] = "A"
                result["order"] = "B_vs_A"
                all_comparisons.append(result)
            except Exception as e:
                print(f"[WARN] Failed to parse comparison sample {i + 6}: {e}")
                all_comparisons.append({
                    "sample_id": f"round2_sample{i + 1}",
                    "winner": "error",
                    "reason": "parsing failed",
                    "order": "B_vs_A"
                })
    except Exception as e:
        print(f"[ERROR] Second round comparison failed: {e}")

    summary = aggregate_comparison_results(all_comparisons)
    return summary, all_comparisons


def parse_comparison_result(response: str, label_a: str, label_b: str, sample_idx: int) -> Dict:
    """解析比较结果JSON - 简化版（只提取winner）"""
    # 清理响应文本
    response = response.strip()

    # 尝试提取JSON
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            result = json.loads(json_str)

            # 只提取winner字段
            winner = str(result.get("winner", "")).upper()
            if winner not in ["A", "B", "TIE"]:
                winner = "error"

            return {
                "sample_id": f"sample{sample_idx + 1}",
                "winner": winner,
                "order": f"{label_a}_vs_{label_b}"
            }
        except json.JSONDecodeError:
            pass

    # 如果没有找到有效的JSON，尝试直接查找winner
    response_lower = response.lower()
    if '"winner": "a"' in response_lower or "'winner': 'a'" in response_lower:
        winner = "A"
    elif '"winner": "b"' in response_lower or "'winner': 'b'" in response_lower:
        winner = "B"
    elif '"winner": "tie"' in response_lower or "'winner': 'tie'" in response_lower:
        winner = "TIE"
    else:
        winner = "error"

    return {
        "sample_id": f"sample{sample_idx + 1}",
        "winner": winner,
        "order": f"{label_a}_vs_{label_b}"
    }


def aggregate_comparison_results(comparisons: List[Dict]) -> Dict:
    """
    汇总比较结果
    """
    valid_comparisons = [c for c in comparisons if c["winner"] in ["A", "B", "TIE"]]

    if not valid_comparisons:
        return {
            "total_samples": 0,
            "valid_samples": 0,
            "wins_A": 0,
            "wins_B": 0,
            "ties": 0,
            "win_rate_A": 0.0,
            "win_rate_B": 0.0,
            "tie_rate": 0.0
        }

    wins_A = sum(1 for c in valid_comparisons if c["winner"] == "A")
    wins_B = sum(1 for c in valid_comparisons if c["winner"] == "B")
    ties = sum(1 for c in valid_comparisons if c["winner"] == "TIE")

    total_valid = len(valid_comparisons)

    return {
        "total_samples": len(comparisons),
        "valid_samples": total_valid,
        "wins_A": wins_A,
        "wins_B": wins_B,
        "ties": ties,
        "win_rate_A": wins_A / total_valid if total_valid > 0 else 0.0,
        "win_rate_B": wins_B / total_valid if total_valid > 0 else 0.0,
        "tie_rate": ties / total_valid if total_valid > 0 else 0.0
    }
