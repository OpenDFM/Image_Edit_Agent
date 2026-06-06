import re
import json
import cv2
import sys
import os
import traceback
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
project_root = Path(os.getenv("IEA_PROJECT_ROOT", Path(__file__).resolve().parents[5]))
sys.path.append(str(project_root))
from src import config
from src.utils import print_with_color, extract_json_from_string, extract_answer_from_string, compute_diff_score, load_and_resize_image
from src.image_editor import ImageEditor
from src.llm_caller import get_summary_reward
from src.utils import compute_string_similarity


def compute_score_00(data_source,solution_str,ground_truth,extra_info=None):
    reward = compute_score(data_source, solution_str, ground_truth, extra_info,0)
    return reward
    
def compute_score_03(data_source,solution_str,ground_truth,extra_info=None):
    reward = compute_score(data_source, solution_str, ground_truth, extra_info,0.3)
    return reward

def compute_score_05(data_source,solution_str,ground_truth,extra_info=None):
    reward = compute_score(data_source, solution_str, ground_truth, extra_info, 0.5)
    return reward

def compute_score_07(data_source,solution_str,ground_truth,extra_info=None):
    reward = compute_score(data_source, solution_str, ground_truth, extra_info, 0.7)
    return reward

def compute_score_10(data_source,solution_str,ground_truth,extra_info=None):
    reward = compute_score(data_source, solution_str, ground_truth, extra_info,1)
    return reward

def compute_score_07_no_rm(data_source,solution_str,ground_truth,extra_info=None):
    reward = compute_score(data_source, solution_str, ground_truth, extra_info, 0.7,False)
    return reward
    
def compute_score(data_source, solution_str, ground_truth, extra_info=None, edit_alpha=0.7, summary_rm=True):
    match data_source:
        case "Image-Edit-Refine-Synthesis":
            try:
                edit_params = extract_json_from_string(solution_str)
                
                # if edit_params is empty
                if not edit_params or len(edit_params) == 0:
                    return -1
                
                original_image = load_and_resize_image(extra_info['origin_image_path'])
                reference_image = load_and_resize_image(extra_info['reference_image_path'])
                editor = ImageEditor(extra_info['origin_image_path'])
                edit_image = editor.process_image(edit_params)
                original_score = compute_diff_score(original_image, reference_image)["final_score"]
                edit_score = compute_diff_score(edit_image, reference_image)["final_score"]
                # print(f"Edit Score for {extra_info['id']}: {edit_score}, Original Score: {original_score}")
                reward_L = max(-1, (original_score - edit_score) / original_score)
                
                reward_U = 0
                origin_params = extra_info['origin_params']
                origin_params = {k: v for k, v in origin_params.items() if v is not None}
                
                refined_params = extra_info['refined_params']
                refined_params = {k: v for k, v in refined_params.items() if v is not None}
                
                #  \frac{\Sigma_{Tool_{related}}(max(-1,1-\frac{Params_{target}-Params_{new}}{Params_{target}-Params_{old}}))}{\# Tool_{related}}+\frac{\#{Tool_{related}}-\# Tool_{unrelated}}{\#{Tool_{related}}+\# Tool_{unrelated}}
                
                all_keys = set(list(origin_params.keys()) + list(refined_params.keys()) + list(edit_params.keys()))
                
                TP_count = 0
                FP_count = 0
                TN_count = 0
                FN_count = 0
                
                all_diff = 0
                
                for key in all_keys:
                    origin_value = origin_params.get(key, 0)
                    refined_value = refined_params.get(key, 0)
                    new_value = edit_params.get(key, 0)
                    if refined_value != origin_value:
                        if new_value != origin_value:
                            TP_count += 1
                        else:
                            FN_count += 1
                        all_diff += min(1, max(-1, 1 - (refined_value - new_value) / (refined_value - origin_value)))
                    else:
                        if new_value != origin_value:
                            FP_count += 1
                        else:
                            TN_count += 1
                
                reward_Diff = all_diff / max(1, TP_count + FN_count)
                
                reward_IoU = (TP_count - FP_count) / max(1, TP_count + FP_count)
                
                alpha = 0.3
                beta = 0.4
                reward = alpha * reward_L + beta * reward_Diff + (1 - alpha - beta) * reward_IoU
            
            except Exception as e:
                # print(f"Error processing data: \n{data_source}\nModel response: \n{solution_str}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                reward = -1
            return reward
        case "Image-Summary" | "Image-Summary-Synthesis":
            try:
                prediction = extract_answer_from_string(solution_str)
                instruction = extra_info["instruction"]
                if summary_rm:
                    score = get_summary_reward(instruction, prediction)
                else:
                    score_ = compute_string_similarity(ground_truth, prediction)
                    score = score_.get('Rouge-L',0)
                    
                # print(f"Summary Score for '{instruction}' and '{prediction}': {score}")
                reward = score / 10.0  # Normalize the score to a range of -1 to 1
            except:
                # print(f"Error processing data: \n{data_source}\nModel response: \n{solution_str}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                reward = -1
            return reward
        case "Image-Edit" | "Image-Edit-Synthesis":
            try:
                edit_params = extract_json_from_string(solution_str)
                
                # if edit_params is empty
                if not edit_params or len(edit_params) == 0:
                    return -1
                
                original_image = load_and_resize_image(extra_info['origin_image_path'])
                reference_image = load_and_resize_image(extra_info['reference_image_path'])
                editor = ImageEditor(extra_info['origin_image_path'])
                edit_image = editor.process_image(edit_params)
                original_score = compute_diff_score(original_image, reference_image)["final_score"]
                edit_score = compute_diff_score(edit_image, reference_image)["final_score"]
                # print(f"Edit Score for {extra_info['id']}: {edit_score}, Original Score: {original_score}")
                reward_L = max(-1, (original_score - edit_score) / original_score)
                
                reward_U = 0
                for key in edit_params.keys():
                    edit_params_ = edit_params.copy()
                    edit_params_[key] = 0
                    edit_image_ = editor.process_image(edit_params_)
                    edit_score_ = compute_diff_score(edit_image_, reference_image)["final_score"]
                    # print(f"Edit Score for {extra_info['id']} with {key} set to 0: {edit_score_}, Original Score: {original_score}")
                    if edit_score < edit_score_:
                        # this tool is useful
                        reward_U += 1
                    else:
                        # this tool is not useful
                        reward_U -= 1
                reward_U = reward_U / len(edit_params)
                # edit_alpha = 0.7
                reward = edit_alpha * reward_L + (1 - edit_alpha) * reward_U
            
            except Exception as e:
                # print(f"Error processing data: \n{data_source}\nModel response: \n{solution_str}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                reward = -1
            return reward

def compute_score_simple(data_source, solution_str, ground_truth, extra_info=None):
    try:
        edit_params = extract_json_from_string(solution_str)
        original_image = load_and_resize_image(extra_info['origin_image_path'])
        reference_image = load_and_resize_image(extra_info['reference_image_path'])
        editor = ImageEditor(extra_info['origin_image_path'])
        edit_image = editor.process_image(edit_params)
        original_score = compute_diff_score(original_image, reference_image)["final_score"]
        edit_score = compute_diff_score(edit_image, reference_image)["final_score"]
        reward = max(-1, (original_score - edit_score) / original_score)
    
    except Exception as e:
        print(f"Error processing data: \n{data_source}\nModel response: \n{solution_str}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        reward = -1
    return reward
