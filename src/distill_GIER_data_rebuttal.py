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
from Demos.SystemParametersInfo import new_value
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
    best_score = initial_score.copy()
    best_edit_image = initial_image
    current_params = initial_params.copy()
    current_score = initial_score['final_score']
    current_edit_image = initial_image
    avail_operators = list(initial_params.keys())  # TODO: update list
    
    max_iter = config.SA_MAX_ITER
    patience = config.SA_PATIENCE
    initial_temp = config.SA_INITIAL_TEMP
    final_temp = config.SA_FINAL_TEMP
    cooling_rate = (final_temp / initial_temp) ** (1 / max_iter)
    temp = initial_temp
    no_improve_count = 0
    iteration = 0
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
            best_score = new_score_dict.copy()
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


def greedy_optimize(initial_params, initial_score, initial_edit_image, reference_image, editor, args):
    """贪心遍历优化：对每个参数遍历所有可能值，找到最优"""
    best_params = initial_params.copy()
    best_score = initial_score.copy()
    best_edit_image = initial_edit_image
    avail_operators = list(initial_params.keys())
    iteration = 0
    
    start_time = time.perf_counter()
    
    for param in avail_operators:
        current_params = best_params.copy()
        min_val, max_val = config.NUMERICAL_OPERATORS.get(param, (-100, 100))
        # 遍历参数范围，步长为10
        for new_value in range(min_val, max_val + 1, 1):
            iteration += 1
            new_params = current_params.copy()
            new_params[param] = new_value
            if args.verbose: print_with_color(f"Greedy: {param}={new_value}", "YELLOW")
            new_edit_image = editor.process_image(new_params)
            new_score = compute_diff_score(new_edit_image, reference_image)
            if new_score['final_score'] < best_score['final_score']:
                best_params = new_params
                best_score = new_score
                best_edit_image = new_edit_image
    
    elapsed = time.perf_counter() - start_time
    
    if args.verbose: print_with_color(f"Greedy Best Score: {best_score['final_score']:.4f}", "GREEN")
    if args.verbose: print_with_color(f"Greedy completed in {elapsed:.2f} seconds", "GREEN")
    
    return best_params, best_score, best_edit_image, iteration, elapsed


def random_optimize(initial_params, initial_score, initial_edit_image, reference_image, editor, args):
    """随机搜索优化：随机挑选八个值进行测试"""
    best_params = initial_params.copy()
    best_score = initial_score.copy()
    best_edit_image = initial_edit_image
    avail_operators = list(initial_params.keys())
    iteration = 0
    num_random_samples = 8
    
    start_time = time.perf_counter()
    
    for param in avail_operators:
        current_params = best_params.copy()
        current_value = current_params[param]
        min_val, max_val = config.NUMERICAL_OPERATORS.get(param, (-100, 100))
        
        # 随机采样8个不同的值
        random_deltas = random.sample(range(-100, 100), min(num_random_samples, max_val - min_val + 1))
        
        for delta in random_deltas:
            iteration += 1
            new_params = current_params.copy()
            new_value = min(max_val, max(min_val, current_value+ delta))
            new_params[param] = new_value
            if args.verbose: print_with_color(f"Random: {param}={new_value}", "YELLOW")
            new_edit_image = editor.process_image(new_params)
            new_score = compute_diff_score(new_edit_image, reference_image)
            if new_score['final_score'] < best_score['final_score']:
                best_params = new_params
                best_score = new_score
                best_edit_image = new_edit_image
    
    elapsed = time.perf_counter() - start_time
    
    if args.verbose: print_with_color(f"Random Best Score: {best_score['final_score']:.4f}", "GREEN")
    if args.verbose: print_with_color(f"Random completed in {elapsed:.2f} seconds", "GREEN")
    
    return best_params, best_score, best_edit_image, iteration, elapsed


def reflect_optimize(initial_params, initial_score, initial_edit_image, reference_image, editor, data_item, args):
    """模型反思优化：使用模型进行一次反思"""
    best_params = initial_params.copy()
    best_score = initial_score.copy()
    best_edit_image = initial_edit_image
    ref_op = replace_keywords(data_item['reference_operators'], config.REPLACE_TABLE) if args.use_ref_op else []
    
    start_time = time.perf_counter()
    
    # 保存当前最优编辑图像
    best_edit_image_path = os.path.join(tempfile.gettempdir(), f"reflect_{data_item['id']}.png")
    cv2.imwrite(best_edit_image_path, best_edit_image)
    
    try:
        # 调用模型进行反思
        new_params, reasoning, model, usage = predict_operations_gier_reflect(
            data_item['origin_image_path'],
            data_item['reference_image_path'],
            best_edit_image_path,
            data_item['instruction'],
            best_params,
            args.inst_type,
            args.cot,
            ref_op,
            args.log_file
        )
        
        new_params = replace_keywords(new_params, config.REPLACE_TABLE)
        
        if new_params:
            new_edit_image = editor.process_image(new_params)
            new_score = compute_diff_score(new_edit_image, reference_image)
            if new_score['final_score'] < best_score['final_score']:
                best_params = new_params
                best_score = new_score
                best_edit_image = new_edit_image
    finally:
        if os.path.exists(best_edit_image_path):
            os.remove(best_edit_image_path)
    
    elapsed = time.perf_counter() - start_time
    
    if args.verbose: print_with_color(f"Reflect Best Score: {best_score['final_score']:.4f}", "GREEN")
    if args.verbose: print_with_color(f"Reflect completed in {elapsed:.2f} seconds", "GREEN")
    
    return best_params, best_score, best_edit_image, 1, elapsed


def get_prediction(data_item, args):
    """
    执行十种方法的对比实验：
    基于模型生成的初始参数：
    1. 初次生成 (generation)
    2. 贪心遍历 (greedy)
    3. 快速搜索 (fast)
    4. 随机搜索 (random)
    5. 模型反思 (reflect)
    6. 模拟退火 (sa)
    
    基于全零初始参数：
    7. 全零贪心遍历 (zero_greedy)
    8. 全零快速搜索 (zero_fast)
    9. 全零随机搜索 (zero_random)
    10. 全零模拟退火 (zero_sa)
    """
    ref_op = replace_keywords(data_item['reference_operators'], config.REPLACE_TABLE) if args.use_ref_op else []
    original_image = load_and_resize_image(data_item['origin_image_path'])
    reference_image = load_and_resize_image(data_item['reference_image_path'])
    editor = ImageEditor(data_item['origin_image_path'])
    
    results = {}  # 存储所有方法的结果
    
    # ========== 1. 初次生成 ==========
    start_time = time.perf_counter()
    initial_params, reasoning, model, usage = predict_operations_gier(
        data_item['origin_image_path'],
        data_item['reference_image_path'],
        data_item['instruction'],
        args.inst_type,
        args.cot,
        ref_op,
        args.log_file
    )
    initial_params = replace_keywords(initial_params, config.REPLACE_TABLE)
    generation_elapsed = time.perf_counter() - start_time
    
    initial_edit_image = editor.process_image(initial_params)
    initial_score = compute_diff_score(initial_edit_image, reference_image)
    original_score = compute_diff_score(original_image, reference_image)
    
    if args.verbose:
        print_with_color(f"Initial Params: {initial_params}", "YELLOW")
        print_with_color(f"Initial Score: {initial_score['final_score']:.4f}", "GREEN")
    
    results['generation'] = {
        'method': 'generation',
        'params': initial_params.copy(),
        'score': initial_score.copy(),
        'elapsed': generation_elapsed,
        'extra_info': {
            'reasoning': reasoning,
            'model': model,
            'usage': usage
        }
    }
    
    # ========== 2. 贪心遍历 ==========
    greedy_params, greedy_score, greedy_image, greedy_iter, greedy_elapsed = greedy_optimize(
        initial_params, initial_score, initial_edit_image, reference_image, editor, args
    )
    results['greedy'] = {
        'method': 'greedy',
        'params': greedy_params.copy(),
        'score': greedy_score.copy(),
        'elapsed': greedy_elapsed,
        'extra_info': {'iterations': greedy_iter}
    }
    
    # ========== 3. 快速搜索（八个固定的delta） ==========
    fast_params, fast_score, fast_image, fast_iter, fast_elapsed = fast_optimize(
        initial_params, initial_score, initial_edit_image, reference_image, editor, args
    )
    results['fast'] = {
        'method': 'fast',
        'params': fast_params.copy(),
        'score': fast_score.copy(),
        'elapsed': fast_elapsed,
        'extra_info': {'iterations': fast_iter}
    }
    
    # ========== 4. 随机挑选八个值进行测试 ==========
    random_params, random_score, random_image, random_iter, random_elapsed = random_optimize(
        initial_params, initial_score, initial_edit_image, reference_image, editor, args
    )
    results['random'] = {
        'method': 'random',
        'params': random_params.copy(),
        'score': random_score.copy(),
        'elapsed': random_elapsed,
        'extra_info': {'iterations': random_iter}
    }
    
    # ========== 5. 使用模型进行反思 ==========
    reflect_params, reflect_score, reflect_image, reflect_iter, reflect_elapsed = reflect_optimize(
        initial_params, initial_score, initial_edit_image, reference_image, editor, data_item, args
    )
    results['reflect'] = {
        'method': 'reflect',
        'params': reflect_params.copy(),
        'score': reflect_score.copy(),
        'elapsed': reflect_elapsed,
        'extra_info': {'iterations': reflect_iter}
    }
    
    # ========== 6. 使用模拟退火算法 ==========
    sa_params, sa_score, sa_image, sa_iter, sa_elapsed = simulated_annealing_optimize(
        initial_params, initial_score, initial_edit_image, reference_image, editor, args
    )
    results['sa'] = {
        'method': 'sa',
        'params': sa_params.copy(),
        'score': sa_score.copy(),
        'elapsed': sa_elapsed,
        'extra_info': {'iterations': sa_iter}
    }
    
    # ========== 全零初始参数的情况 ==========
    # 创建全零初始参数（包含所有数值操作工具）
    zero_params = {op: 0 for op in config.NUMERICAL_OPERATORS.keys()}
    zero_edit_image = editor.process_image(zero_params)
    zero_score = compute_diff_score(zero_edit_image, reference_image)
    
    if args.verbose:
        print_with_color(f"Zero Params: {zero_params}", "YELLOW")
        print_with_color(f"Zero Score: {zero_score['final_score']:.4f}", "GREEN")
    
    # # ========== 7. 全零贪心遍历 ==========
    # zero_greedy_params, zero_greedy_score, zero_greedy_image, zero_greedy_iter, zero_greedy_elapsed = greedy_optimize(
    #     zero_params, zero_score, zero_edit_image, reference_image, editor, args
    # )
    # results['zero_greedy'] = {
    #     'method': 'zero_greedy',
    #     'params': zero_greedy_params.copy(),
    #     'score': zero_greedy_score.copy(),
    #     'elapsed': zero_greedy_elapsed,
    #     'extra_info': {'iterations': zero_greedy_iter}
    # }
    #
    # # ========== 8. 全零快速搜索 ==========
    # zero_fast_params, zero_fast_score, zero_fast_image, zero_fast_iter, zero_fast_elapsed = fast_optimize(
    #     zero_params, zero_score, zero_edit_image, reference_image, editor, args
    # )
    # results['zero_fast'] = {
    #     'method': 'zero_fast',
    #     'params': zero_fast_params.copy(),
    #     'score': zero_fast_score.copy(),
    #     'elapsed': zero_fast_elapsed,
    #     'extra_info': {'iterations': zero_fast_iter}
    # }
    #
    # # ========== 9. 全零随机搜索 ==========
    # zero_random_params, zero_random_score, zero_random_image, zero_random_iter, zero_random_elapsed = random_optimize(
    #     zero_params, zero_score, zero_edit_image, reference_image, editor, args
    # )
    # results['zero_random'] = {
    #     'method': 'zero_random',
    #     'params': zero_random_params.copy(),
    #     'score': zero_random_score.copy(),
    #     'elapsed': zero_random_elapsed,
    #     'extra_info': {'iterations': zero_random_iter}
    # }
    #
    # # ========== 10. 全零模拟退火 ==========
    # zero_sa_params, zero_sa_score, zero_sa_image, zero_sa_iter, zero_sa_elapsed = simulated_annealing_optimize(
    #     zero_params, zero_score, zero_edit_image, reference_image, editor, args
    # )
    # results['zero_sa'] = {
    #     'method': 'zero_sa',
    #     'params': zero_sa_params.copy(),
    #     'score': zero_sa_score.copy(),
    #     'elapsed': zero_sa_elapsed,
    #     'extra_info': {'iterations': zero_sa_iter}
    # }
    
    # 构建返回数据
    data_item_update = {
        'original_score': original_score.copy(),
        'results': results,
    }
    
    # 绘制对比图（如果启用）
    if args.draw:
        output_images_dir = os.path.join(args.output_dir, 'comparisons')
        os.makedirs(output_images_dir, exist_ok=True)
        
        # 找到最优方法
        best_method = min(results.keys(), key=lambda k: results[k]['score']['final_score'])
        best_edit_image = editor.process_image(results[best_method]['params'])
        best_score = results[best_method]['score']
        
        comparison_grid = create_comparison_grid(
            original_image, reference_image, initial_edit_image, best_edit_image,
            original_score, initial_score['final_score'], best_score['final_score']
        )
        comparison_image_path = os.path.join(output_images_dir, f"{data_item['id']}_comparison.png")
        cv2.imwrite(comparison_image_path, comparison_grid)
    
    return data_item_update, initial_score['final_score']


def pipeline(args):
    dataset = load_gier_dataset(args)
    
    if args.number_of_samples > 0:
        print_with_color(f"Limiting to {args.number_of_samples} samples from the dataset", "CYAN")
        dataset = dict(list(dataset.items())[:args.number_of_samples])
    else:
        print_with_color(f"Processing all {len(list(dataset.items()))} samples from the dataset", "CYAN")
    
    # 十种方法的统计
    methods = ['generation', 'greedy', 'fast', 'random', 'reflect', 'sa',
               
               # 'zero_greedy', 'zero_fast', 'zero_random', 'zero_sa'
               ]
    method_scores = {m: [] for m in methods}
    method_times = {m: [] for m in methods}
    
    dataset_items = list(dataset.items())
    
    if args.max_workers > 1:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # 提交所有任务
            future_to_id = {executor.submit(get_prediction, data_item, args): id for id, data_item in dataset_items}
            
            # 处理完成的任务
            with tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Processing GIER dataset") as pbar:
                for future in pbar:
                    id = future_to_id[future]
                    try:
                        updated_data, initial_score = future.result()
                        dataset[id].update(updated_data)
                        
                        # 收集各方法的分数和耗时
                        for method in methods:
                            if method in updated_data.get('results', {}):
                                method_scores[method].append(updated_data['results'][method]['score']['final_score'])
                                method_times[method].append(updated_data['results'][method]['elapsed'])
                        
                    except Exception as e:
                        print_with_color(f"Error processing id {id}: {e}", "RED")
                        traceback.print_exc()
                    
                    # 更新进度条显示
                    gen_avg = sum(method_scores['generation']) / len(method_scores['generation']) if method_scores['generation'] else 0
                    pbar.set_postfix_str(f"id={id}, gen_avg={gen_avg:.4f}")
    else:
        for id, data_item in tqdm(dataset_items, desc="Processing GIER dataset"):
            try:
                updated_data, initial_score = get_prediction(data_item, args)
                dataset[id].update(updated_data)
                
                # 收集各方法的分数和耗时
                for method in methods:
                    if method in updated_data.get('results', {}):
                        method_scores[method].append(updated_data['results'][method]['score']['final_score'])
                        method_times[method].append(updated_data['results'][method]['elapsed'])
                        
            except Exception as e:
                print_with_color(f"Error processing id {id}: {e}", "RED")
                traceback.print_exc()
    
    # 打印十种方法的统计结果
    print_with_color("\n" + "=" * 80, "CYAN")
    print_with_color("实验结果统计 (Rebuttal Experiment Results)", "CYAN")
    print_with_color("=" * 80, "CYAN")
    
    statistics = {}
    method_names = {
        'generation': '1. 初次生成 (模型生成)',
        'greedy': '2. 贪心遍历 (模型初始)',
        'fast': '3. 快速搜索 (模型初始)',
        'random': '4. 随机搜索 (模型初始)',
        'reflect': '5. 模型反思 (模型初始)',
        'sa': '6. 模拟退火 (模型初始)',
        'zero_greedy': '7. 贪心遍历 (全零初始)',
        'zero_fast': '8. 快速搜索 (全零初始)',
        'zero_random': '9. 随机搜索 (全零初始)',
        'zero_sa': '10. 模拟退火 (全零初始)'
    }
    
    print_with_color("\n--- 基于模型生成初始参数的方法 ---", "CYAN")
    
    model_methods = ['generation', 'greedy', 'fast', 'random', 'reflect', 'sa']
    zero_methods = ['zero_greedy', 'zero_fast', 'zero_random', 'zero_sa']
    
    for method in model_methods:
        if method_scores[method]:
            avg_score = sum(method_scores[method]) / len(method_scores[method])
            avg_time = sum(method_times[method]) / len(method_times[method])
            min_score = min(method_scores[method])
            max_score = max(method_scores[method])
            
            statistics[method] = {
                'avg_score': avg_score,
                'avg_time': avg_time,
                'min_score': min_score,
                'max_score': max_score,
                'count': len(method_scores[method])
            }
            
            print_with_color(f"{method_names[method]}: 平均分数={avg_score:.4f}, 平均耗时={avg_time:.2f}s, 样本数={len(method_scores[method])}", "GREEN")
    
    print_with_color("\n--- 基于全零初始参数的方法 ---", "CYAN")
    
    for method in zero_methods:
        if method_scores[method]:
            avg_score = sum(method_scores[method]) / len(method_scores[method])
            avg_time = sum(method_times[method]) / len(method_times[method])
            min_score = min(method_scores[method])
            max_score = max(method_scores[method])
            
            statistics[method] = {
                'avg_score': avg_score,
                'avg_time': avg_time,
                'min_score': min_score,
                'max_score': max_score,
                'count': len(method_scores[method])
            }
            
            print_with_color(f"{method_names[method]}: 平均分数={avg_score:.4f}, 平均耗时={avg_time:.2f}s, 样本数={len(method_scores[method])}", "GREEN")
    
    print_with_color("=" * 80, "CYAN")
    
    # 保存结果
    output_json_path = os.path.join(args.output_dir, f'distill_data.json')
    with open(output_json_path, 'w', encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    # 保存统计结果
    statistics_path = os.path.join(args.output_dir, 'statistics.json')
    with open(statistics_path, 'w', encoding="utf-8") as f:
        json.dump(statistics, f, indent=4, ensure_ascii=False)
    
    # 保存详细的分数和时间列表
    detailed_results_path = os.path.join(args.output_dir, 'detailed_results.json')
    detailed_results = {
        'method_scores': method_scores,
        'method_times': method_times
    }
    with open(detailed_results_path, 'w', encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4, ensure_ascii=False)
    
    print_with_color(f"\n结果已保存至: {args.output_dir}", "GREEN")


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
    args = parser.parse_args()
    
    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    args.output_dir = os.path.join(config.PROJECT_ROOT, 'data', 'GIER_distill', f"Distill-{time_string}-rebuttal")
    args.log_file = os.path.join(args.output_dir, 'log.txt')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # save args and configs
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        config_dict = {k: v for k, v in vars(config).items() if not (k.startswith('__') or k in ['Path', 'os', 'platform'])}
        args_dict = {k: v for k, v in vars(args).items() if not k.startswith('__')}
        config_dict.update(args_dict)
        print(config_dict)
        json.dump(config_dict, f, indent=4)
    
    pipeline(args)
