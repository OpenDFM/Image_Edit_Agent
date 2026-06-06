import json
import os
import pdb
import random

from jupyterlab.semver import test_set
from tqdm import tqdm

import config
import prompts
from src.utils import print_with_color
from utils import merge_images

DEFAULT_TEMPLATE = ["Please simply make this image better.", "Edit this image to make it more beautiful", "What will you do to improve this image?", "Just refine this image, make it better.", "How to enhance this image?"]

number_of_scores = {
    10: 0,
    9: 0,
    8: 0,
    7: 0,
    6: 0,
    5: 0,
    4: 0,
    3: 0,
    2: 0,
    1: 0,
    0: 0,
    -1: 0,
    -2: 0,
    -3: 0,
    -4: 0,
    -5: 0,
    -6: 0,
    -7: 0,
    -8: 0,
    -9: 0,
    -10: 0
}

accept_possibility = {
    10: 0.1,
    9: 1,
    8: 1,
    7: 0.5,
    6: 1,
    5: 1,
    4: 1,
    3: 1,
    2: 1,
    1: 0.75,
    0: 1,
    -1: 1,
    -2: 1,
    -3: 1,
    -4: 1,
    -5: 1,
    -6: 1,
    -7: 1,
    -8: 1,
    -9: 1,
    -10: 0.2
}


def prepare_sft_dataset_reward(data_path, train_split=1.0):
    train_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_reward_train.json"
    test_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_reward_test.json"
    train_info_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_reward_train_info.json"
    
    data_ = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_.append(json.loads(line))
            
            """
            {"group_id": 1, "score": 10, "instruction": "somewhat reduce overall exposure, bring down the midtones as much as possible and tame the hot spots by reducing highlights softly; add a distinctly cooler color tone, neutralize a magenta cast by adding green somewhat and subtly boost overall saturation.", "params": {"exposure": -55, "brightness": -100, "tint": 53, "temperature": -79, "saturation": 27, "highlights": -25}, "source": "template"}
            """
    
    data = {}
    for item in data_:
        group_id = item['group_id']
        score = item['score']
        instruction = item['instruction']
        params = item['params']
        source = item['source']
        
        if group_id not in data:
            data[group_id] = {}
        if score not in data[group_id]:
            data[group_id][score] = []
        data[group_id][score].append(instruction)
    
    train_sft_data = []
    test_sft_data = []
    
    for group_id, group_data in tqdm(data.items(), desc="Processing data"):
        reference_instruction_list = group_data[10]
        predict_instruction_list = []
        for item in range(-10, 11):
            if item in group_data:
                for instruction in group_data[item]:
                    predict_instruction_list.append((instruction, item, accept_possibility[item]))
                predict_instruction_list.append((random.choice(DEFAULT_TEMPLATE), 0, 0.1))
        
        for reference_instruction in reference_instruction_list:
            for predict_instruction, score, possi in predict_instruction_list:
                if random.random() > possi:
                    continue
                if reference_instruction == predict_instruction:
                    if random.random() > 0.1:
                        continue
                    score = 10
                number_of_scores[score] += 1
                
                output = f'{score}'
                
                data = {
                    "conversations": [{
                        "role": "system",
                        "content": prompts.SUMMARY_REWARD_SYSTEM_PROMPT_ZERO_SHOT,
                    }, {
                        "role": "user",
                        "content": prompts.SUMMARY_REWARD_USER_PROMPT.replace("{prediction}", predict_instruction).replace("{reference}", reference_instruction),
                    }, {
                        "role": "assistant",
                        "content": output,
                    }],
                    "id": f"{group_id}_{hash(reference_instruction) % 100000}_{hash(predict_instruction) % 100000}",
                    "extra_info": {
                        "data_source": "Image-Summary-RM",
                        "prediction": predict_instruction,
                        "reference": reference_instruction,
                        "ground_truth": score
                    }
                }
                
                if random.random() > train_split:
                    test_sft_data.append(data)
                else:
                    train_sft_data.append(data)
    
    dataset_info = {
        "gier_sft_summary_reward": {
            "file_name": "../" + os.path.relpath(train_json_path).replace('\\', '/'),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
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
    print_with_color(f"The number of scores is {number_of_scores}", "green")
    
    # dump jsons
    with open(train_json_path, 'w', encoding='utf-8') as f:
        json.dump(train_sft_data, f, indent=4, ensure_ascii=False)
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_sft_data, f, indent=4, ensure_ascii=False)
    with open(train_info_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    dataset_path = r"../datasets/GIER_synthesis/instructions_2000.jsonl"
    prepare_sft_dataset_reward(dataset_path, 0.98)
