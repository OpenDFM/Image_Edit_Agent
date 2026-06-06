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


def prepare_sft_dataset_summary(data_path, train_split=1.0):
    train_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_train.json"
    test_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_test.json"
    train_info_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_train_info.json"
    original_data_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER\GIER.json"
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data_ = json.load(f)  # convert to dict use key "output"
    
    original_data = {item['output'].split(".")[0]: item for item in original_data_}
    
    train_sft_data = []
    test_sft_data = []
    
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
            
            # ShareGPT Format
            data = {
                "conversations": [{
                    "role": "system",
                    "content": prompts.SFT_SUMMARY_SYSTEM_PROMPT,
                }, {
                    "role": "user",
                    "content": prompts.SFT_SUMMARY_USER_PROMPT,
                }, {
                    "role": "assistant",
                    "content": output,
                }],
                "images": ['../../' + os.path.relpath(merged_image_path, config.PROJECT_ROOT).replace('\\', '/')],
                "id": item_id + "_" + str(i),
                "extra_info": {
                    "origin_image_path": '../../' + os.path.relpath(origin_image_path, config.PROJECT_ROOT).replace('\\', '/'),
                    "reference_image_path": '../../' + os.path.relpath(reference_image_path, config.PROJECT_ROOT).replace('\\', '/'),
                    "label": instruction,
                    "task_type": "inst_summary",
                }
            }
            
            if item_id in test_id_list:
                test_sft_data.append(data)
            else:
                train_sft_data.append(data)
                
                i += 1
    
    dataset_info = {
        "gier_sft_summary": {
            "file_name": "../" + os.path.relpath(train_json_path).replace('\\', '/'),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system"
            }
        }
    }
    
    # randomize and split the data
    random.shuffle(train_sft_data)
    random.shuffle(test_sft_data)
    
    print_with_color(f"The length of train data is {len(train_sft_data)}", "green")
    print_with_color(f"The length of test data is {len(test_sft_data)}", "green")
    
    # dump jsons
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_sft_data, f, indent=4, ensure_ascii=False)
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_sft_data, f, indent=4, ensure_ascii=False)
    with open(train_info_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)


def prepare_sft_dataset_edit(data_path, train_split=1.0):
    train_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_train.json"
    test_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_test.json"
    train_info_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_train_info.json"
    original_data_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER\GIER.json"
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data_ = json.load(f)  # convert to dict use key "output"
    
    original_data = {item['output'].split(".")[0]: item for item in original_data_}
    
    train_sft_data = []
    test_sft_data = []
    
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
            # ShareGPT Format
            data = {
                "conversations": [{
                    "role": "system",
                    "content": prompts.SFT_SYSTEM_PROMPT,
                }, {
                    "role": "user",
                    "content": prompts.SFT_USER_PROMPT_INST.replace("{instruction}", instruction) if instruction else prompts.SFT_USER_PROMPT,
                }, {
                    "role": "assistant",
                    "content": output,
                }],
                "images": ['../../' + os.path.relpath(origin_image_path, config.PROJECT_ROOT).replace('\\', '/')],
                "id": item_id + "_" + str(i),
                "extra_info": {
                    "origin_image_path": '../../' + os.path.relpath(origin_image_path, config.PROJECT_ROOT).replace('\\', '/'),
                    "reference_image_path": '../../' + os.path.relpath(reference_image_path, config.PROJECT_ROOT).replace('\\', '/'),
                    "label": best_params,
                    "instruction": instruction,
                    "inst_type": instruction_type,  # "expert", "amateur", "general", "none"
                    "task_type": "image_editing",
                }
            }
            
            if item_id in test_id_list and i != len(instruction_list) - 1:
                test_sft_data.append(data)
            else:
                train_sft_data.append(data)
            
            i += 1
    
    dataset_info = {
        "gier_sft_distill": {
            "file_name": "../" + os.path.relpath(train_json_path).replace('\\', '/'),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "images": "images"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system"
            }
        }
    }
    
    # randomize and split the data
    random.shuffle(train_sft_data)
    random.shuffle(test_sft_data)
    
    print_with_color(f"The length of train data is {len(train_sft_data)}", "green")
    print_with_color(f"The length of test data is {len(test_sft_data)}", "green")
    
    # dump jsons
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_sft_data, f, indent=4, ensure_ascii=False)
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_sft_data, f, indent=4, ensure_ascii=False)
    with open(train_info_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    dataset_path = r"../data/GIER_distill/Pick-20250714-174542/distill_data.json"
    prepare_sft_dataset_summary(dataset_path, 0.95)
    prepare_sft_dataset_edit(dataset_path, 0.95)
