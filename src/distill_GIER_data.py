import argparse
import json
import os
import pdb
import random
import tempfile
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
from llm_caller import predict_operations_gier, predict_operations_gier_reflect
from utils import compute_diff_score, load_and_resize_image, print_with_color, replace_keywords


def load_gier_dataset(args):
    if os.path.exists(args.json_path):
        json_path = args.json_path
    elif os.path.exists(os.path.join(config.PROJECT_ROOT, args.json_path)):
        json_path = os.path.join(config.PROJECT_ROOT, args.json_path)
    else:
        json_path = os.path.join(config.PROJECT_ROOT, 'datasets', 'GIER', 'GIER.json')
    
    if os.path.exists(args.image_base_path):
        image_base_path = args.image_base_path
    elif os.path.exists(os.path.join(config.PROJECT_ROOT, args.image_base_path)):
        image_base_path = os.path.join(config.PROJECT_ROOT, args.image_base_path)
    else:
        image_base_path = os.path.join(config.PROJECT_ROOT, 'datasets', 'GIER', 'images')
    
    # json_path = os.path.join(config.PROJECT_ROOT, 'sample', 'test_gier_format.json')
    # image_base_path = os.path.join(config.PROJECT_ROOT, 'sample', 'style_example')
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            gier_data = json.load(f)
    except FileNotFoundError:
        print_with_color(f"Error: GIER.json not found at {json_path}", "RED")
        exit()
    
    processed_data = {}
    operators = Counter()
    for item in tqdm(gier_data, desc="Processing GIER dataset"):
        
        no_local = True
        for operator_key, operator_value in item.get('operator', {}).items():
            if isinstance(operator_value, dict):
                no_local = no_local and not operator_value["local"]
            no_local = no_local and (operator_key in config.NUMERICAL_OPERATORS or operator_key in config.REPLACE_TABLE)
        
        if not no_local:
            continue
        
        input_image_path = os.path.join(image_base_path, item['input'])
        output_image_path = os.path.join(image_base_path, item['output'])
        
        if not os.path.exists(input_image_path):
            print_with_color(f"Warning: Input image not found: {input_image_path}", "YELLOW")
            continue
        if not os.path.exists(output_image_path):
            print_with_color(f"Warning: Output image not found: {output_image_path}", "YELLOW")
            continue
        
        match args.inst_type:
            case -1:
                instruction = ""
            case 0:
                instructions = item.get('amateur_summary', [])
                if instructions:
                    instruction = "；".join(instructions) if args.merge_inst else instructions[0]
                else:
                    continue
            case 1:
                instructions = item.get('expert_summary', [])
                if instructions:
                    instruction = "；".join(instructions) if args.merge_inst else instructions[0]
                else:
                    continue
            case 2:
                instruction = item.get('preference', '')
        
        processed_data[item['output'].split('.')[0]] = {
            'id': item['output'].split('.')[0],
            'origin_image_path': input_image_path,
            'reference_image_path': output_image_path,
            'instruction': instruction,
            'reference_operators': list(item.get('operator', {}).keys()),
        }
        operators.update(["total samples"])
        operators.update(item.get('operator', {}).keys())
    
    print_with_color(operators, "CYAN")
    return processed_data


def create_comparison_grid(original_image, reference_image, initial_edit_image, best_edit_image, original_score, initial_score, best_score):
    w_, h_ = original_image.shape[1], original_image.shape[0]
    w, h = int(w_ * 512 / min(w_, h_)), int(h_ * 512 / min(w_, h_))
    
    original_image_rescale = cv2.resize(original_image, (w, h))
    reference_image_rescale = cv2.resize(reference_image, (w, h))
    initial_edit_image_rescale = cv2.resize(initial_edit_image, (w, h))
    best_edit_image_rescale = cv2.resize(best_edit_image, (w, h))
    
    grid_width = w * 2
    grid_height = h * 2
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
    
    grid[h:h * 2, 0:w] = initial_edit_image_rescale
    put_text_with_border(grid, f"Initial Edit (Score: {initial_score:.4f})", (10, h + 30), font, font_scale, text_color, border_color, font_thickness)
    
    grid[h:h * 2, w:w * 2] = best_edit_image_rescale
    put_text_with_border(grid, f"Optimized Edit (Score: {best_score:.4f})", (w + 10, h + 30), font, font_scale, text_color, border_color, font_thickness)
    
    return grid


def simulated_annealing_optimize(initial_params, initial_score, initial_image, reference_image, editor, args):
    best_params = initial_params.copy()
    best_score = initial_score['final_score']
    best_edit_image = initial_image
    current_params = initial_params.copy()
    current_score = initial_score['final_score']
    avail_operators = list(initial_params.keys())  # TODO: update list
    
    max_iter = config.SA_MAX_ITER
    patience = config.SA_PATIENCE
    initial_temp = config.SA_INITIAL_TEMP
    final_temp = config.SA_FINAL_TEMP
    cooling_rate = (final_temp / initial_temp) ** (1 / max_iter)
    temp = initial_temp
    no_improve_count = 0
    start_time = time.perf_counter()
    
    for iteration in range(max_iter):
        new_params = current_params.copy()
        perturbation_factor = max(0.1, temp / initial_temp)
        for param in avail_operators:
            if random.random() < config.SA_PARAM_CHANGE_PROB:
                min_val, max_val = config.NUMERICAL_OPERATORS[param]
                current_value = new_params[param]
                max_perturb = int(config.SA_PERTURBATION_BASE * perturbation_factor)
                perturb = random.randint(-max_perturb, max_perturb)
                new_value = current_value + perturb
                new_value = max(min_val, min(max_val, new_value))
                new_params[param] = new_value
        for param in config.NON_NUMERICAL_OPERATORS:
            if param in initial_params:
                new_params[param] = initial_params[param]
        if args.verbose: print_with_color(f"Iteration {iteration + 1}, New Params: {new_params}", "YELLOW")
        new_edit_image = editor.process_image(new_params)
        new_score_dict = compute_diff_score(new_edit_image, reference_image)
        new_score = new_score_dict['final_score']
        if new_score < current_score:
            if args.verbose: print_with_color(f"Current Score: {new_score:.4f}", "GREEN")
            current_params = new_params
            current_score = new_score
            current_edit_image = new_edit_image
        else:
            if args.verbose: print_with_color(f"Current Score: {new_score:.4f}", "RED")
            score_diff = new_score - current_score
            accept_prob = np.exp(-score_diff / temp)
            if random.random() < accept_prob:
                current_params = new_params
                current_score = new_score
                current_edit_image = new_edit_image
        if current_score < best_score['final_score']:
            best_params = current_params.copy()
            best_score = new_score['final_score']
            best_edit_image = current_edit_image
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience:
            break
        temp *= cooling_rate
    
    elapsed = time.perf_counter() - start_time
    
    if args.verbose: print_with_color(best_params, "YELLOW")
    if args.verbose: print_with_color(f"Best Score after optimization: {best_score['final_score']:.4f}", "GREEN")
    if args.verbose: print_with_color(f"Optimization completed in {elapsed:.2f} seconds", "GREEN")
    
    return best_params, best_score, best_edit_image, iteration, elapsed


def fast_optimize(initial_params, initial_score, initial_edit_image, reference_image, editor, args):
    best_params = initial_params
    best_score = initial_score
    best_edit_image = initial_edit_image
    avail_operators = list(initial_params.keys())  # TODO: update list
    iteration = 0
    
    start_time = time.perf_counter()
    
    for param in avail_operators:
        current_params = best_params.copy()
        current_value = current_params[param]
        min_val, max_val = config.NUMERICAL_OPERATORS.get(param, (-100, 100))
        for perturb in [-50, 25, -10, 5, -5, 10, -25, 50]:
            iteration += 1
            new_value = current_value + perturb
            new_value = max(min_val, min(max_val, new_value))
            if current_value * new_value < 0 or new_value == 0:
                continue
            new_params = current_params.copy()
            new_params[param] = new_value
            if args.verbose: print_with_color(f"Optimizing {param}, New Params: {new_params}", "YELLOW")
            new_edit_image = editor.process_image(new_params)
            new_score = compute_diff_score(new_edit_image, reference_image)
            if args.verbose: print_with_color(f"Current Score: {new_score['final_score']:.4f}", "GREEN")
            if new_score['final_score'] < best_score['final_score']:
                best_params = new_params
                best_score = new_score
                best_edit_image = new_edit_image
    
    elapsed = time.perf_counter() - start_time
    
    if args.verbose: print_with_color(best_params, "YELLOW")
    if args.verbose: print_with_color(f"Best Score after optimization: {best_score['final_score']:.4f}", "GREEN")
    if args.verbose: print_with_color(f"Optimization completed in {elapsed:.2f} seconds", "GREEN")
    
    return best_params, best_score, best_edit_image, iteration, elapsed


def get_prediction(data_item, args):
    ref_op = replace_keywords(data_item['reference_operators'], config.REPLACE_TABLE) if args.use_ref_op else []
    original_image = load_and_resize_image(data_item['origin_image_path'])
    reference_image = load_and_resize_image(data_item['reference_image_path'])
    editor = ImageEditor(data_item['origin_image_path'])
    
    data_item_update = {
        'iteration': [],
        'best_params': {},
        'best_score': {},
    }
    
    iter = 0
    while iter <= config.REFLECT_MAX_ITER:
        start_time = time.perf_counter()
        if iter == 0:
            initial_params, reasoning, model, usage = predict_operations_gier(data_item['origin_image_path'], data_item['reference_image_path'], data_item['instruction'], args.inst_type, args.cot, ref_op, args.log_file)
        else:
            best_edit_image_path = os.path.join(tempfile.gettempdir(), f"best_{data_item['id']}_iter{iter}.png")
            cv2.imwrite(best_edit_image_path, editor.process_image(data_item_update['best_params']))
            initial_params, reasoning, model, usage = predict_operations_gier_reflect(data_item['origin_image_path'], data_item['reference_image_path'], best_edit_image_path, data_item['instruction'], best_params, args.inst_type, args.cot, ref_op, args.log_file)
            os.remove(best_edit_image_path)
        
        initial_params = replace_keywords(initial_params, config.REPLACE_TABLE)
        elapsed = time.perf_counter() - start_time
        
        if args.verbose: print_with_color(initial_params, "YELLOW")
        if args.verbose: print_with_color(f"Generation completed in {elapsed:.2f} seconds", "GREEN")
        
        initial_edit_image = editor.process_image(initial_params)
        initial_score = compute_diff_score(initial_edit_image, reference_image)
        original_score = compute_diff_score(original_image, reference_image)
        if iter == 0: very_first_score = initial_score.copy()
        
        if args.verbose: print_with_color(f"Initial Score: {initial_score['final_score']:.4f}", "GREEN")
        
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
        
        match args.optimize:
            case "sa":
                best_params, best_score, best_edit_image, iteration, elapsed = simulated_annealing_optimize(initial_params, initial_score, initial_edit_image, reference_image, editor, args)
                data_item_update['iteration'].append({
                    'method': 'simulated_annealing',
                    'extra_info': {
                        'optimize_round': iteration
                    },
                    'params': best_params.copy(),
                    'score': best_score.copy(),
                    'elapsed': elapsed
                })
            case "fast":
                best_params, best_score, best_edit_image, iteration, elapsed = fast_optimize(initial_params, initial_score, initial_edit_image, reference_image, editor, args)
                data_item_update['iteration'].append({
                    'method': 'fast_search',
                    'extra_info': {},
                    'params': best_params.copy(),
                    'score': best_score.copy(),
                    'elapsed': elapsed
                })
            # else
            case _:
                best_params = initial_params.copy()
                best_score = initial_score
                best_edit_image = initial_edit_image
        
        if iter == 0 or best_score['final_score'] < data_item_update['best_score']['final_score']:
            data_item_update['best_params'] = best_params.copy()
            data_item_update['best_score'] = best_score.copy()
            data_item_update['original_score'] = original_score.copy()
            data_item_update['reasoning'] = reasoning
        else:
            # skip this iteration if no improvement
            break
        
        if args.draw:
            comparison_grid = create_comparison_grid(original_image, reference_image, initial_edit_image, best_edit_image, original_score, initial_score['final_score'], best_score['final_score'])
            output_images_dir = os.path.join(args.output_dir, 'comparisons')
            os.makedirs(output_images_dir, exist_ok=True)
            comparison_image_path = os.path.join(output_images_dir, f"{data_item['id']}_comparison{'' if iter == 0 else f'_{iter}'}.png")
            cv2.imwrite(comparison_image_path, comparison_grid)
        
        iter += 1
    
    return data_item_update, very_first_score['final_score'], best_score['final_score']


def pipeline(args):
    dataset = load_gier_dataset(args)
    
    if args.number_of_samples > 0:
        print_with_color(f"Limiting to {args.number_of_samples} samples from the dataset", "CYAN")
        dataset = dict(list(dataset.items())[:args.number_of_samples])
    else:
        print_with_color(f"Processing all {len(list(dataset.items()))} samples from the dataset", "CYAN")
    
    score_list = []
    best_score_list = []
    dataset_items = list(dataset.items())
    
    if args.max_workers > 1:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # 提交所有任务
            future_to_id = {executor.submit(get_prediction, data_item, args): id for id, data_item in dataset_items}
            
            # 处理完成的任务
            with tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Processing GIER dataset") as pbar:
                for future in pbar:
                    id = future_to_id[future]
                    score, best_score = 'N/A', 'N/A'
                    try:
                        updated_data, score, best_score = future.result()
                        dataset[id].update(updated_data)
                        score_list.append(score)
                        best_score_list.append(best_score)
                    except Exception as e:
                        print_with_color(f"Error processing id {id}: {e}", "RED")
                        traceback.print_exc()
                    
                    avg_score = sum(score_list) / len(score_list) if score_list else 0
                    avg_best_score = sum(best_score_list) / len(best_score_list) if best_score_list else 0
                    
                    score_str = f"{score:.4f}" if isinstance(score, float) else score
                    best_score_str = f"{best_score:.4f}" if isinstance(best_score, float) else best_score
                    
                    pbar.set_postfix_str(f"id={id}, score={score_str}({avg_score:.4f}), best_score={best_score_str}({avg_best_score:.4f})")  # pbar.update(1)
    else:
        for id, data_item in tqdm(dataset_items, desc="Processing GIER dataset"):
            updated_data, score, best_score = get_prediction(data_item, args)
            dataset[id].update(updated_data)
            score_list.append(score)
            best_score_list.append(best_score)
    print_with_color("Average Score: {:.4f}".format(sum(score_list) / len(score_list)), "GREEN")
    print_with_color("Average Best Score: {:.4f}".format(sum(best_score_list) / len(best_score_list)), "GREEN")
    
    output_json_path = os.path.join(args.output_dir, f'distill_data.json')
    with open(output_json_path, 'w', encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process GIER Dataset")
    parser.add_argument('--json_path', type=str, default='', help='Path to the GIER JSON dataset file.')
    parser.add_argument('--image_base_path', type=str, default='', help='Base path for images in the GIER dataset.')
    parser.add_argument('--inst_type', type=int, default=1, choices=[-1, 0, 1, 2], help='Use -1 for no instructions, 0 for amateur instructions, 1 for expert instructions, 2 for preference instructions.')
    parser.add_argument('--cot', type=int, default=1, choices=[0, 1, 2], help='Add Chain of Thought reasoning to the model calls.')
    parser.add_argument('--merge_inst', action='store_true', help='Merge all instructions into one string for input.')
    parser.add_argument('--max_workers', type=int, default=1, help='Number of worker threads for processing.')
    parser.add_argument('--use_ref_op', action='store_true', help='Use reference operators from the dataset.')
    parser.add_argument('--draw', action='store_true', help='Draw comparison images for each processed item.')
    parser.add_argument('--number_of_samples', '-n', type=int, default=-1, help='Number of samples to process from the dataset. -1 means all samples.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output for debugging.')
    parser.add_argument('--optimize', '-o', type=str, default='sa', choices=['sa', 'fast', 'random', 'greedy', 'none'], help='Optimization method to use: "sa" for simulated annealing, "greedy" for greedy optimization.')
    args = parser.parse_args()
    
    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir = os.path.join(config.PROJECT_ROOT, 'data', 'GIER_distill', f"Distill-{time_string}")
    args.log_file = os.path.join(args.output_dir, 'log.txt')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # save args and configs
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        config_dict = {k: v for k, v in vars(config).items() if not (k.startswith('__') or k in ['Path', 'os'])}
        args_dict = {k: v for k, v in vars(args).items() if not k.startswith('__')}
        config_dict.update(args_dict)
        json.dump(config_dict, f, indent=4)
    
    pipeline(args)
