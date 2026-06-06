import json
import os
import random

from jupyterlab.semver import test_set
from tqdm import tqdm

import config
import prompts
from src.utils import print_with_color
from utils import merge_images

DEFAULT_TEMPLATE = ["Please simply make this image better.", "Edit this image to make it more beautiful", "What will you do to improve this image?", "Just refine this image, make it better.", "How to enhance this image?"]

test_id_path = r"D:\Codes\Image_Edit_Agent\data\GIER_distill\Pick-20250714-174542\distill_data_test_id_list.txt"
test_id_list = []

if os.path.exists(test_id_path):
    with open(test_id_path, 'r', encoding='utf-8') as f:
        test_id_list = [line.strip() for line in f.readlines() if line.strip()]


def prepare_grpo_dataset_summary(data_path, project_root_server):
    train_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\grpo_train.json"
    test_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\grpo_test.json"
    original_data_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER\GIER.json"
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data_ = json.load(f)  # convert to dict use key "output"
    
    original_data = {item['output'].split(".")[0]: item for item in original_data_}
    
    train_grpo_data = []
    test_grpo_data = []
    
    for item_id, item_data in tqdm(data.items(), desc="Processing data"):
        origin_image_path = item_data.get('origin_image_path')
        reference_image_path = item_data.get('reference_image_path')
        best_params = item_data.get('best_params')
        if not best_params:
            best_params = item_data.get('optimized_params')
        reasoning = item_data.get('reasoning', '')
        
        if not all([origin_image_path, reference_image_path, best_params]):
            continue
        
        if not os.path.exists(origin_image_path):
            print_with_color(f"Origin image path {origin_image_path} does not exist, skipping item {item_id}", "red")
            continue
        if not os.path.exists(reference_image_path):
            print_with_color(f"Reference image path {reference_image_path} does not exist, skipping item {item_id}", "red")
            continue
        
        i = 1
        for instruction in original_data[item_id].get("expert_summary", []):  # + original_data[item_id].get("amateur_summary", []):
            output = f'<think>\n{reasoning}\n</think>\n<answer>\n{instruction}\n</answer>'
            merged_image_path = os.path.join(config.PROJECT_ROOT, 'datasets', 'GIER-Summary', "merge", f'{item_id}_merged.jpg')
            merge_images(origin_image_path, reference_image_path, merged_image_path)
            
            data = {
                "data_source": "Image-Summary",
                "prompt": [{
                    "role": "system",
                    "content": prompts.SFT_SUMMARY_SYSTEM_PROMPT,
                }, {
                    "role": "user",
                    "content": prompts.SFT_SUMMARY_USER_PROMPT,
                }],
                "images": [{
                    "image_url": "file://" + str(project_root_server + "/" + os.path.relpath(merged_image_path, config.PROJECT_ROOT)).replace("\\", "/"),
                }],
                "ability": "summarize user preference",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": output
                },
                "extra_info": {
                    "id": item_id,
                    "instruction": instruction,
                    "task_type": "inst_summary",
                },
            }
            
            if item_id in test_id_list:
                test_grpo_data.append(data)
            else:
                train_grpo_data.append(data)
                
                i += 1
    
    # randomize and split the data
    random.shuffle(train_grpo_data)
    random.shuffle(test_grpo_data)
    
    print_with_color(f"The length of train data is {len(train_grpo_data)}", "green")
    print_with_color(f"The length of test data is {len(test_grpo_data)}", "green")
    
    # dump jsons
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_grpo_data, f, indent=4, ensure_ascii=False)
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_grpo_data, f, indent=4, ensure_ascii=False)


def prepare_grpo_dataset_edit(data_path, project_root_server):
    train_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\grpo_train.json"
    test_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\grpo_test.json"
    original_data_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER\GIER.json"
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data_ = json.load(f)  # convert to dict use key "output"
    
    original_data = {item['output'].split(".")[0]: item for item in original_data_}
    
    train_grpo_data = []
    test_grpo_data = []
    
    for item_id, item_data in tqdm(data.items(), desc="Processing data"):
        origin_image_path = item_data.get('origin_image_path')
        reference_image_path = item_data.get('reference_image_path')
        best_params = item_data.get('best_params')
        if not best_params:
            best_params = item_data.get('optimized_params')
        reasoning = item_data.get('reasoning', '')
        
        if not all([origin_image_path, reference_image_path, best_params]):
            continue
        if not os.path.exists(origin_image_path):
            print_with_color(f"Origin image path {origin_image_path} does not exist, skipping item {item_id}", "red")
            continue
        if not os.path.exists(reference_image_path):
            print_with_color(f"Reference image path {reference_image_path} does not exist, skipping item {item_id}", "red")
            continue
        
        output = f'<think>\n{reasoning}\n</think>\n<answer>\n```json\n{json.dumps(best_params, indent=4)}\n```\n</answer>'
        
        i = 1
        
        # set instruction_list with a tuple, (inst, type)
        instruction_list = []
        if "expert_summary" in original_data[item_id]:
            instruction_list.extend([(inst, "expert") for inst in original_data[item_id]["expert_summary"]])
        if "amateur_summary" in original_data[item_id]:
            instruction_list.extend([(inst, "amateur") for inst in original_data[item_id]["amateur_summary"]])
        instruction_list.extend([(inst, "general") for inst in DEFAULT_TEMPLATE])
        instruction_list.extend([("", "none") for _ in range(2)])
        
        for instruction, instruction_type in instruction_list:
            data = {
                "data_source": "Image-Edit",
                "prompt": [{
                    "role": "system",
                    "content": prompts.SFT_SYSTEM_PROMPT,
                }, {
                    "role": "user",
                    "content": prompts.SFT_USER_PROMPT_INST.replace("{instruction}", instruction) if instruction else prompts.SFT_USER_PROMPT,
                }],
                "images": [{
                    "image_url": "file://" + str(project_root_server + "/" + os.path.relpath(origin_image_path, config.PROJECT_ROOT)).replace("\\", "/"),
                }],
                "ability": "image editing",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": output
                },
                "extra_info": {
                    "id": item_id,
                    "params": best_params,
                    "instruction": instruction,
                    "origin_image_path": str(project_root_server + "/" + os.path.relpath(origin_image_path, config.PROJECT_ROOT).replace('\\', '/')),
                    "reference_image_path": str(project_root_server + "/" + os.path.relpath(reference_image_path, config.PROJECT_ROOT).replace('\\', '/')),
                    "inst_type": instruction_type,  # "expert", "amateur", "general", "none"
                    "task_type": "image_editing",
                },
            }
            
            if item_id in test_id_list and i != len(instruction_list) - 1:
                test_grpo_data.append(data)
            else:
                train_grpo_data.append(data)
            
            i += 1
    
    # randomize and split the data
    random.shuffle(train_grpo_data)
    random.shuffle(test_grpo_data)
    
    print_with_color(f"The length of train data is {len(train_grpo_data)}", "green")
    print_with_color(f"The length of test data is {len(test_grpo_data)}", "green")
    
    # dump jsons
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_grpo_data, f, indent=4, ensure_ascii=False)
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_grpo_data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    dataset_path = r"../data/GIER_distill/Pick-20250714-174542/distill_data.json"
    project_root_server = config.PROJECT_ROOT
    prepare_grpo_dataset_summary(dataset_path, project_root_server)
    prepare_grpo_dataset_edit(dataset_path, project_root_server)
