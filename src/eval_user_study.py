import argparse
import json
import os
import pdb
import time
import traceback
from collections import Counter
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

import config
from image_editor import ImageEditor
from llm_caller import call_vlm_api, get_summary_reward, predict_operations_gier_eval
import prompts
from utils import compute_diff_score, compute_string_similarity, extract_answer_from_string, extract_cot_from_string, extract_json_from_string, print_with_color, replace_keywords


def get_prediction(data_item, args, task_type):
    return get_prediction_edit(data_item, args)


def predict_operations_gier_edit_eval(args, origin_image_path: str, instruction: str = "", log_file: str = None):
    """
    Calls the VLM to determine image editing adjustments.
    """
    if args.list_tools:
        system_prompt = prompts.SFT_SYSTEM_PROMPT + prompts.SFT_EDIT_AVAIL_TOOLS.replace("{image_edit_tools}", prompts.IMAGE_EDIT_TOOLS)
        if not instruction:
            user_prompt = prompts.SFT_USER_PROMPT
        else:
            user_prompt = prompts.SFT_USER_PROMPT_INST.replace("{instruction}", instruction) + prompts.SFT_EDIT_FORMAT
    else:
        system_prompt = prompts.SFT_SYSTEM_PROMPT
        if not instruction:
            user_prompt = prompts.SFT_USER_PROMPT
        else:
            user_prompt = prompts.SFT_USER_PROMPT_INST.replace("{instruction}", instruction)
    
    image_list = [origin_image_path]
    
    i = 0
    while i < config.LLM_MAX_RETRY:
        response, model, usage = call_vlm_api(system_prompt=system_prompt, user_prompt=user_prompt, model=args.model, image_list=image_list)
        
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
    editor = ImageEditor(data_item['origin_image_path'], False)
    
    data_item_update = {
        'iteration': [],
        'best_params': {},
        'best_score': {},
    }
    
    start_time = time.perf_counter()
    initial_params, reasoning, model, usage = predict_operations_gier_edit_eval(args, data_item['origin_image_path'], data_item['prompt'], args.log_file)
    # initial_params, reasoning, model, usage = {},"", "", {}
    initial_params = replace_keywords(initial_params, config.REPLACE_TABLE)
    
    initial_edit_image = editor.process_image(initial_params)
    
    edit_image_path = os.path.join(args.output_dir, f"{data_item['id'].split("_")[0]}_{args.model}.jpg")
    
    cv2.imwrite(edit_image_path, initial_edit_image)
    
    elapsed = time.perf_counter() - start_time
    data_item_update['iteration'].append({
        'method': 'generation',
        'extra_info': {
            'reasoning': reasoning,
            'model': model,
            'usage': usage
        },
        'params': initial_params.copy(),
        'elapsed': elapsed
    })
    
    return data_item_update


def pipeline(args):
    for task_type, dataset_path in args.eval_dataset_list.items():
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset_list = json.load(f)
        dataset = {item['id']: item for item in dataset_list}
        
        if args.number_of_samples > 0:
            print_with_color(f"Limiting to {args.number_of_samples} samples from the {task_type} dataset: {dataset_path}", "CYAN")
            dataset = dict(list(dataset.items())[:args.number_of_samples])
        else:
            print_with_color(f"Processing all {len(list(dataset.items()))} samples from the {task_type} dataset: {dataset_path}", "CYAN")
        
        dataset_items = list(dataset.items())
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # 提交所有任务
            future_to_id = {executor.submit(get_prediction, data_item, args, task_type): id for id, data_item in dataset_items}
            
            # 处理完成的任务
            with tqdm(as_completed(future_to_id), total=len(future_to_id), desc=f"Evaluating {task_type} task") as pbar:
                for future in pbar:
                    id = future_to_id[future]
                    try:
                        updated_data = future.result()
                        dataset[id].update(updated_data)
                    except Exception as e:
                        print_with_color(f"Error processing id {id}: {e}", "RED")
                        traceback.print_exc()
        
        output_file = os.path.join(args.output_dir, f"{task_type}_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
        
        # write a summary log
        with open(args.log_file, 'a', encoding='utf-8') as log_f:
            log_f.write(f"Model: {args.model}\n")
            log_f.write(f"Task: {task_type}\n")
            log_f.write(f"Dataset: {dataset_path}\n")
            log_f.write(f"Number of samples: {len(dataset)}\n")
            log_f.write("\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process GIER Dataset")
    parser.add_argument('--max_workers', type=int, default=1, help='Number of worker threads for processing.')
    parser.add_argument('--number_of_samples', '-n', type=int, default=-1, help='Number of samples to process from the dataset. -1 means all samples.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output for debugging.')
    parser.add_argument('--model', '-m', type=str, default=config.VLM_DEFAULT_MODEL, help='Model to use for VLM API calls.')
    parser.add_argument('--list_tools', action='store_true', help='List available image editing tools')
    args = parser.parse_args()
    
    args.eval_dataset_list = {
        "edit": r"D:\Codes\Image_Edit_Agent\user_study\data_selected_new_for_vlm.json",
    }
    
    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir = r"D:\Codes\Image_Edit_Agent\user_study\images" + "\\" + args.model
    args.log_file = os.path.join(args.output_dir, 'log.txt')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    pipeline(args)
