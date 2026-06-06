import argparse
import json
import os
import pdb
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

try:
    from . import config
    from .image_editor import ImageEditor
    from .llm_caller import call_vlm_api, get_summary_reward, predict_operations_gier_eval
    from . import prompts
    from .utils import compute_diff_score, compute_string_similarity, extract_answer_from_string, extract_cot_from_string, extract_json_from_string, load_and_resize_image, print_with_color, replace_keywords
except ImportError:
    import config
    from image_editor import ImageEditor
    from llm_caller import call_vlm_api, get_summary_reward, predict_operations_gier_eval
    import prompts
    from utils import compute_diff_score, compute_string_similarity, extract_answer_from_string, extract_cot_from_string, extract_json_from_string, load_and_resize_image, print_with_color, replace_keywords
    


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    match data_source:
        case "Image-Edit":
            try:
                edit_params = extract_json_from_string(solution_str)
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
                alpha = 0.7
                reward = alpha * reward_L + (1 - alpha) * reward_U
            
            except Exception as e:
                # print(f"Error processing data: \n{data_source}\nModel response: \n{solution_str}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                reward = -1
            return reward
        case "Image-Summary":
            try:
                prediction = extract_answer_from_string(solution_str)
                instruction = extra_info["instruction"]
                score = get_summary_reward(instruction, prediction)
                # print(f"Summary Score for '{instruction}' and '{prediction}': {score}")
                reward = score / 10.0  # Normalize the score to a range of -1 to 1
            except:
                reward = -1
            return reward


def create_comparison_grid(original_image, reference_image, initial_edit_image, original_score, initial_score):
    w_, h_ = original_image.shape[1], original_image.shape[0]
    w, h = int(w_ * 512 / min(w_, h_)), int(h_ * 512 / min(w_, h_))
    
    original_image_rescale = cv2.resize(original_image, (w, h))
    reference_image_rescale = cv2.resize(reference_image, (w, h))
    initial_edit_image_rescale = cv2.resize(initial_edit_image, (w, h))
    
    grid_width = w * 3
    grid_height = h * 1
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, w / 800.0)
    font_thickness = max(1, int(font_scale * 2))
    text_color = (255, 255, 255)
    border_color = (0, 0, 0)
    
    def put_text_with_border(img, text, org, font_face, font_scale, text_color, border_color, thickness):
        # 绘制边框
        cv2.putText(img, text, org, font_face, font_scale, border_color, thickness + 2, cv2.LINE_AA)
        # 绘制文本
        cv2.putText(img, text, org, font_face, font_scale, text_color, thickness, cv2.LINE_AA)
    
    grid[0:h, 0:w] = original_image_rescale
    put_text_with_border(grid, f"Original (Score: {original_score:.4f})", (10, 30), font, font_scale, text_color, border_color, font_thickness)
    
    grid[0:h, w:w * 2] = reference_image_rescale
    put_text_with_border(grid, "Reference", (w + 10, 30), font, font_scale, text_color, border_color, font_thickness)
    
    grid[0:h, w * 2:w * 3] = initial_edit_image_rescale
    put_text_with_border(grid, f"Initial Edit (Score: {initial_score:.4f})", (w * 2 + 10, 30), font, font_scale, text_color, border_color, font_thickness)
    
    return grid


def get_prediction(data_item, args, task_type):
    if task_type == "summary":
        return get_prediction_summary(data_item, args)
    elif task_type == "edit":
        return get_prediction_edit(data_item, args)
    elif task_type == "refine":
        return get_prediction_refine(data_item, args)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def predict_operations_gier_summary_eval(compare_image_path: str, log_file: str = None):
    """
    Calls the VLM to determine image editing adjustments.
    """
    system_prompt = prompts.SFT_SUMMARY_SYSTEM_PROMPT
    
    user_prompt = prompts.SFT_SUMMARY_USER_PROMPT  # + prompts.SFT_SUMMARY_FORMAT
    
    image_list = [compare_image_path]
    
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
            extracted_inst = extract_answer_from_string(response)
            extracted_cot = extract_cot_from_string(response)
            if extracted_inst:
                return extracted_inst, extracted_cot, model, usage
            else:
                print_with_color("Warning: Could not extract answer from LLM response.")
                print_with_color(f"Response was: {response}")
                return response.strip().split("\n")[-1], "", model, usage
        i += 1
        print(f"Failed to extract response. Retry {i} times.")
    return "", "", "", {}


def get_prediction_summary(data_item, args):
    start_time = time.perf_counter()
    merged_image = cv2.imread(data_item['images'][0][3:])  # remove first ../
    inst_pred, reasoning, model, usage = predict_operations_gier_summary_eval(data_item['images'][0][3:], args.log_file)
    score = compute_string_similarity(data_item['extra_info']['label'], inst_pred)
    reward = get_summary_reward(data_item['extra_info']['label'], inst_pred)
    reward = max(-1, min(1, reward / 10.0))
    elapsed = time.perf_counter() - start_time
    data_item_update = {
        "inst_pred": inst_pred,
        "reasoning": reasoning,
        "model": model,
        "usage": usage,
        "score": score,
        "elapsed": elapsed,
        "reward": reward
    }
    return data_item_update, score["Rouge-L"], (reward,)


def predict_operations_gier_edit_eval(origin_image_path: str, instruction: str = "", log_file: str = None):
    """
    Calls the VLM to determine image editing adjustments.
    """
    system_prompt = prompts.SFT_SYSTEM_PROMPT
    
    if not instruction:
        user_prompt = prompts.SFT_USER_PROMPT  # + prompts.SFT_EDIT_FORMAT
    else:
        user_prompt = prompts.SFT_USER_PROMPT_INST.replace("{instruction}", instruction)  #+ prompts.SFT_EDIT_FORMAT
    
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


def get_prediction_edit(data_item, args):
    original_image = cv2.imread(data_item['extra_info']['origin_image_path'][3:])
    reference_image = cv2.imread(data_item['extra_info']['reference_image_path'][3:])
    editor = ImageEditor(data_item['extra_info']['origin_image_path'][3:])
    
    data_item_update = {
        'iteration': [],
        'best_params': {},
        'best_score': {},
    }
    
    start_time = time.perf_counter()
    reward_L = -1
    i = 0
    if reward_L < 0 and i < 3:
        initial_params, reasoning, model, usage = predict_operations_gier_edit_eval(data_item['extra_info']['origin_image_path'][3:], data_item['extra_info']['instruction'], args.log_file)
        # initial_params, reasoning, model, usage = {},"", "", {}
        initial_params = replace_keywords(initial_params, config.REPLACE_TABLE)
        
        try:
            initial_edit_image = editor.process_image(initial_params)
            original_score = compute_diff_score(original_image, reference_image)
            initial_score = compute_diff_score(initial_edit_image, reference_image)
        except Exception as e:
            print(e)
            initial_edit_image = original_image.copy()
            original_score = compute_diff_score(original_image, reference_image)
            initial_score = original_score
        
        reward_L = max(-1, (original_score['final_score'] - initial_score['final_score']) / original_score['final_score'])
        i += 1
    reward_U = 0
    for key in initial_params.keys():
        edit_params_ = initial_params.copy()
        edit_params_[key] = 0
        try:
            edit_image_ = editor.process_image(edit_params_)
            edit_score_ = compute_diff_score(edit_image_, reference_image)["final_score"]
            if initial_score['final_score'] < edit_score_:
                reward_U += 1
            else:
                reward_U -= 1
        except Exception as e:
            continue
    reward_U = reward_U / len(initial_params) if len(initial_params) > 0 else 0
    
    elapsed = time.perf_counter() - start_time
    data_item_update['iteration'].append({
        'method': 'generation',
        'extra_info': {
            'reasoning': reasoning,
            'model': model,
            'usage': usage
        },
        'params': initial_params.copy(),
        'score': initial_score.copy(),
        'elapsed': elapsed
    })
    data_item_update['best_params'] = initial_params.copy()
    data_item_update['best_score'] = initial_score.copy()
    data_item_update['original_score'] = original_score.copy()
    data_item_update['reasoning'] = reasoning
    data_item_update['reward'] = {
        'reward_L': reward_L,
        'reward_U': reward_U
    }
    
    if args.draw:
        comparison_grid = create_comparison_grid(original_image, reference_image, initial_edit_image, original_score['final_score'], initial_score['final_score'])
        output_images_dir = os.path.join(args.output_dir, 'comparisons')
        os.makedirs(output_images_dir, exist_ok=True)
        comparison_image_path = os.path.join(output_images_dir, f"{data_item['id']}_comparison.png")
        cv2.imwrite(comparison_image_path, comparison_grid)
    
    return data_item_update, initial_score['final_score'], (reward_L, reward_U)


def predict_operations_gier_refine_eval(merged_image_path: str, origin_instruction: str, origin_params: dict, refine_instruction: str, log_file: str = None):
    """
    Calls the VLM to determine image editing adjustments.
    """
    system_prompt = prompts.SFT_REFINE_SYSTEM_PROMPT
    
    user_prompt = prompts.SFT_REFINE_USER_PROMPT.replace("{origin_instruction}", origin_instruction).replace("{refine_instruction}", refine_instruction).replace("{old_params}", json.dumps(origin_params, indent=4))

    image_list = [merged_image_path]
    
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


def get_prediction_refine(data_item, args):
    original_image = cv2.imread(data_item['extra_info']['origin_image_path'][3:])
    reference_image = cv2.imread(data_item['extra_info']['reference_image_path'][3:])
    editor = ImageEditor(data_item['extra_info']['origin_image_path'][3:])
    
    data_item_update = {
        'iteration': [],
        'best_params': {},
        'best_score': {},
    }
    
    refine_instruction = data_item['extra_info']['refine_instruction'] if data_item['extra_info']['inst_type'] == "refine" else data_item['extra_info']['refined_instruction']
    
    start_time = time.perf_counter()
    initial_params, reasoning, model, usage = predict_operations_gier_refine_eval(data_item['images'][0][3:], data_item['extra_info']["origin_instruction"], data_item['extra_info']["origin_params"], refine_instruction, args.log_file)
    # initial_params, reasoning, model, usage = {},"", "", {}
    initial_params = replace_keywords(initial_params, config.REPLACE_TABLE)
    
    try:
        initial_edit_image = editor.process_image(initial_params)
        original_score = compute_diff_score(original_image, reference_image)
        initial_score = compute_diff_score(initial_edit_image, reference_image)
    except Exception as e:
        initial_edit_image = original_image.copy()
        original_score = compute_diff_score(original_image, reference_image)
        initial_score = original_score
    
    reward_L = max(-1, (original_score['final_score'] - initial_score['final_score']) / original_score['final_score'])
    
    origin_params = data_item['extra_info']['origin_params']
    origin_params = {k: v for k, v in origin_params.items() if v is not None}
    
    refined_params = data_item['extra_info']['refined_params']
    refined_params = {k: v for k, v in refined_params.items() if v is not None}
    
    #  \frac{\Sigma_{Tool_{related}}(max(-1,1-\frac{Params_{target}-Params_{new}}{Params_{target}-Params_{old}}))}{\# Tool_{related}}+\frac{\#{Tool_{related}}-\# Tool_{unrelated}}{\#{Tool_{related}}+\# Tool_{unrelated}}
    
    all_keys = set(list(origin_params.keys()) + list(refined_params.keys()) + list(initial_params.keys()))
    
    TP_count = 0
    FP_count = 0
    TN_count = 0
    FN_count = 0
    
    all_diff = 0
    
    for key in all_keys:
        origin_value = origin_params.get(key, 0)
        refined_value = refined_params.get(key, 0)
        new_value = initial_params.get(key, 0)
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
    
    elapsed = time.perf_counter() - start_time
    data_item_update['iteration'].append({
        'method': 'generation',
        'extra_info': {
            'reasoning': reasoning,
            'model': model,
            'usage': usage
        },
        'params': initial_params.copy(),
        'score': initial_score.copy(),
        'elapsed': elapsed
    })
    data_item_update['best_params'] = initial_params.copy()
    data_item_update['best_score'] = initial_score.copy()
    data_item_update['original_score'] = original_score.copy()
    data_item_update['reasoning'] = reasoning
    data_item_update['reward'] = {
        'reward_L': reward_L,
        'reward_Diff': reward_Diff,
        'reward_IoU': reward_IoU
    }
    
    return data_item_update, initial_score['final_score'], (reward_L, reward_Diff, reward_IoU)

def pipeline(args):
    for task_type, dataset_path in args.eval_dataset_list:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset_list = json.load(f)
        dataset = {item['id']: item for item in dataset_list}
        
        # if extra_info.inst_type != expert and task_type = image_editing, then skip
        # if task_type == "edit":
        #     dataset = {id: item for id, item in dataset.items() if item['extra_info']['inst_type'] == "expert"}
        
        if args.number_of_samples > 0:
            print_with_color(f"Limiting to {args.number_of_samples} samples from the {task_type} dataset: {dataset_path}", "CYAN")
            dataset = dict(list(dataset.items())[:args.number_of_samples])
        else:
            print_with_color(f"Processing all {len(list(dataset.items()))} samples from the {task_type} dataset: {dataset_path}", "CYAN")
        
        score_list = []
        reward_list = []
        dataset_items = list(dataset.items())
        
        if args.max_workers > 1:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # 提交所有任务
                future_to_id = {executor.submit(get_prediction, data_item, args, task_type): id for id, data_item in dataset_items}
                
                # 处理完成的任务
                with tqdm(as_completed(future_to_id), total=len(future_to_id), desc=f"Evaluating {task_type} task") as pbar:
                    for future in pbar:
                        id = future_to_id[future]
                        score = 'N/A'
                        reward = (-1,)
                        try:
                            updated_data, score, reward = future.result()
                            dataset[id].update(updated_data)
                            score_list.append(score)
                            reward_list.append(reward)
                        except Exception as e:
                            print_with_color(f"Error processing id {id}: {e}", "RED")
                            traceback.print_exc()
                        
                        avg_score = sum(score_list) / len(score_list) if score_list else 0
                        
                        if reward_list:
                            num_rewards = len(reward_list[0])
                            avg_reward_tuple = tuple(sum(col) / len(reward_list) for col in zip(*reward_list))
                            avg_reward_str = ", ".join(f"{r:.4f}" for r in avg_reward_tuple)
                        else:
                            avg_reward_str = "0.0000"
                        
                        reward_str = ", ".join(f"{r:.4f}" for r in reward) if isinstance(reward, tuple) else f"{reward:.4f}"
                        pbar.set_postfix_str(f"id={id}, score={score:.4f}({avg_score:.4f}), reward=({reward_str}) avg=({avg_reward_str})")
        else:
            with tqdm(dataset_items, desc=f"Evaluating {task_type} task") as pbar:
                for id, data_item in pbar:
                    updated_data, score, reward = get_prediction(data_item, args, task_type)
                    dataset[id].update(updated_data)
                    score_list.append(score)
                    reward_list.append(reward)
                    
                    avg_score = sum(score_list) / len(score_list) if score_list else 0
                    if reward_list:
                        num_rewards = len(reward_list[0])
                        avg_reward_tuple = tuple(sum(col) / len(reward_list) for col in zip(*reward_list))
                        avg_reward_str = ", ".join(f"{r:.4f}" for r in avg_reward_tuple)
                    else:
                        avg_reward_str = "0.0000"
                    
                    reward_str = ", ".join(f"{r:.4f}" for r in reward) if isinstance(reward, tuple) else f"{reward:.4f}"
                    pbar.set_postfix_str(f"id={id}, score={score:.4f}({avg_score:.4f}), reward=({reward_str}) avg=({avg_reward_str})")
        
        avg_score_final = sum(score_list) / len(score_list) if score_list else 0
        print_with_color(f"Average Score: {avg_score_final:.4f}", "GREEN")
        if reward_list:
            avg_reward_tuple = tuple(sum(col) / len(reward_list) for col in zip(*reward_list))
            avg_reward_str = ", ".join(f"{r:.4f}" for r in avg_reward_tuple)
            print_with_color(f"Average Reward: ({avg_reward_str})", "GREEN")
        else:
            print_with_color("Average Reward: N/A", "GREEN")
        
        output_file = os.path.join(args.output_dir, f"{task_type}_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
        
        # write a summary log
        with open(args.log_file, 'a', encoding='utf-8') as log_f:
            log_f.write(f"Model: {args.model}\n")
            log_f.write(f"Task: {task_type}\n")
            log_f.write(f"Dataset: {dataset_path}\n")
            log_f.write(f"Number of samples: {len(dataset)}\n")
            log_f.write(f"Average Score: {avg_score_final:.4f}\n")
            if reward_list:
                avg_reward_tuple = tuple(sum(col) / len(reward_list) for col in zip(*reward_list))
                avg_reward_str = ", ".join(f"{r:.4f}" for r in avg_reward_tuple)
                log_f.write(f"Average Reward: ({avg_reward_str})\n")
            else:
                log_f.write("Average Reward: N/A\n")
            log_f.write("\n\n")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process GIER Dataset")
    parser.add_argument('--max_workers', type=int, default=1, help='Number of worker threads for processing.')
    parser.add_argument('--draw', action='store_true', help='Draw comparison images for each processed item.')
    parser.add_argument('--number_of_samples', '-n', type=int, default=-1, help='Number of samples to process from the dataset. -1 means all samples.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output for debugging.')
    parser.add_argument('--model', '-m', type=str, default=config.VLM_DEFAULT_MODEL, help='Model to use for VLM API calls.')
    parser.add_argument('--port', type=int, default=18181, help='Port number for local server (if applicable).')
    parser.add_argument(
        '--task_type',
        type=str,
        default="all",
        choices=["summary", "edit", "refine", "all"],
        help='Task type to evaluate.',
    )
    args = parser.parse_args()
    
    config.VLM_DEFAULT_MODEL=args.model
    config.LLM_BASE_URL=f"http://localhost:{args.port}/v1/chat/completions"
    
    dataset_root = os.path.join(config.PROJECT_ROOT, "datasets")
    dataset_paths = {
        "summary": os.path.join(dataset_root, "GIER-Summary", "sft_test.json"),
        "edit": os.path.join(dataset_root, "GIER-Edit", "sft_test.json"),
        "refine": os.path.join(
            dataset_root,
            "GIER-Edit",
            "sft_synthesis_refine_test.json",
        ),
    }
    if args.task_type == "all":
        args.eval_dataset_list = [
            ("summary", dataset_paths["summary"]),
            ("edit", dataset_paths["edit"]),
        ]
    elif args.task_type in dataset_paths:
        args.eval_dataset_list = [(args.task_type, dataset_paths[args.task_type])]
    else:
        raise ValueError(
            f"Unknown task type {args.task_type!r}; choose summary, edit, refine, or all."
        )
    
    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir = os.path.join(config.PROJECT_ROOT, 'data', 'GIER_Rebuttal_Eval', f"Eval-{args.port}-{time_string}")
    args.log_file = os.path.join(args.output_dir, 'log.txt')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    pipeline(args)
