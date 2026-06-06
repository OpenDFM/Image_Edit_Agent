import json
import os
import pdb
import random

from tqdm import tqdm

import config
import prompts
from src.llm_caller import generate_second_instruction
from src.utils import print_with_color
from utils import merge_images
from image_editor import ImageEditor

image_pool = []

GIER_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER\images"
for file in os.listdir(GIER_path):
    if file.endswith(".jpg") or file.endswith(".png"):
        image_id = file.split(".")[0]
        image_id_1, image_id_2 = image_id.split("_")
        if image_id_1 != image_id_2:
            continue
        image_path = os.path.join(GIER_path, file)
        image_pool.append((image_id_1, image_path))

# Adobe5K_path=r"D:\Codes\Image_Edit_Agent\datasets\MIT-Adobe-FiveK\images\jpg"
# for file in os.listdir(Adobe5K_path):
#     if file.endswith(".jpg") or file.endswith(".png"):
#         image_id=file.split("-")[0]
#         image_path=os.path.join(Adobe5K_path, file)
#         image_pool.append((image_id,image_path))

print(f"Total images in pool: {len(image_pool)}")


def prepare_grpo_dataset_summary(data_path, project_root_server, train_split=0.98):
    train_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\grpo_synthesis_train_adobe5k.json"
    test_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\grpo_synthesis_test_adobe5k.json"
    
    data_ = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_.append(json.loads(line))
            
            """
            {"group_id": 1, "score": 10, "instruction": "somewhat reduce overall exposure, bring down the midtones as much as possible and tame the hot spots by reducing highlights softly; add a distinctly cooler color tone, neutralize a magenta cast by adding green somewhat and subtly boost overall saturation.", "params": {"exposure": -55, "brightness": -100, "tint": 53, "temperature": -79, "saturation": 27, "highlights": -25}, "source": "template"}
            """
    
    data = {}
    for item in data_:
        if item['source'] != "generated":
            continue
        group_id = item['group_id']
        instruction = item['instruction']
        params = item['params']
        instruction_id = f"{group_id}@{hash(instruction) % 100000}"
        
        data[instruction_id] = {
            "instruction": instruction,
            "params": params,
        }
    
    train_grpo_data = []
    test_grpo_data = []
    
    for instruction_id, item_data in tqdm(data.items(), desc="Processing data"):
        origin_image_id, origin_image_path = random.choice(image_pool)
        id = f"{origin_image_id}_{instruction_id}"
        
        params = item_data.get('params')
        instruction = item_data.get('instruction', '')
        i = 1
        output = f'<answer>\n{instruction}\n</answer>'
        reference_image_path = os.path.join(config.PROJECT_ROOT, 'datasets', 'GIER-Synthesis', "processed", f'{id}.jpg')
        merged_image_path = os.path.join(config.PROJECT_ROOT, 'datasets', 'GIER-Synthesis', "merge", f'{id}_merged.jpg')
        
        # process image with params
        editor = ImageEditor(str(origin_image_path))
        editor.process_image(params=params, save_path=str(reference_image_path))
        merge_images(origin_image_path, reference_image_path, merged_image_path)
        
        data = {
            "data_source": "Image-Summary-Synthesis",
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
                "id": id,
                "instruction": instruction,
                "task_type": "inst_summary",
            },
        }
        
        if random.random() > train_split:
            test_grpo_data.append(data)
        else:
            train_grpo_data.append(data)
    
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


def prepare_grpo_dataset_edit(data_path, project_root_server, train_split=0.98):
    train_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\grpo_synthesis_train_adobe5k.json"
    test_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\grpo_synthesis_test_adobe5k.json"
    
    data_ = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_.append(json.loads(line))
            
            """
            {"group_id": 1, "score": 10, "instruction": "somewhat reduce overall exposure, bring down the midtones as much as possible and tame the hot spots by reducing highlights softly; add a distinctly cooler color tone, neutralize a magenta cast by adding green somewhat and subtly boost overall saturation.", "params": {"exposure": -55, "brightness": -100, "tint": 53, "temperature": -79, "saturation": 27, "highlights": -25}, "source": "template"}
            """
    
    data = {}
    for item in data_:
        if item['source'] not in ["generated", "template"]:
            continue
        group_id = item['group_id']
        instruction = item['instruction']
        params = item['params']
        instruction_id = f"{group_id}@{hash(instruction) % 100000}"
        
        data[instruction_id] = {
            "instruction": instruction,
            "params": params,
            "source": item['source'],
        }
    
    train_grpo_data = []
    test_grpo_data = []
    
    REPEAT_TIMES = 5
    
    for instruction_id, item_data in tqdm(data.items(), desc="Processing data"):
        for _ in range(REPEAT_TIMES):
            origin_image_id, origin_image_path = random.choice(image_pool)
            id = f"{origin_image_id}_{instruction_id}"
            
            params = item_data['params']
            instruction = item_data['instruction']
            source = item_data['source']
            
            reference_image_path = os.path.join(config.PROJECT_ROOT, 'datasets', 'GIER-Synthesis', "processed", f'{id}.jpg')
            
            # process image with params
            editor = ImageEditor(str(origin_image_path))
            editor.process_image(params=params, save_path=str(reference_image_path))
            
            output = f'<answer>\n```json\n{json.dumps(params, indent=4)}\n```\n</answer>'
            
            data = {
                "data_source": "Image-Edit-Synthesis",
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
                    "id": id,
                    "params": params,
                    "instruction": instruction,
                    "origin_image_path": str(project_root_server + "/" + os.path.relpath(origin_image_path, config.PROJECT_ROOT).replace('\\', '/')),
                    "reference_image_path": str(project_root_server + "/" + os.path.relpath(reference_image_path, config.PROJECT_ROOT).replace('\\', '/')),
                    "inst_type": source,  # "expert", "amateur", "general", "none"
                    "task_type": "image_editing",
                },
            }
            
            if random.random() > train_split:
                test_grpo_data.append(data)
            else:
                train_grpo_data.append(data)
    
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


def prepare_grpo_dataset_refine(data_path, project_root_server, train_split=0.95):
    train_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\grpo_synthesis_refine_train.json"
    test_json_path = r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\grpo_synthesis_refine_test.json"
    
    data_ = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_.append(json.loads(line))
            
            """
            {"group_id": 1, "score": 10, "instruction": "somewhat reduce overall exposure, bring down the midtones as much as possible and tame the hot spots by reducing highlights softly; add a distinctly cooler color tone, neutralize a magenta cast by adding green somewhat and subtly boost overall saturation.", "params": {"exposure": -55, "brightness": -100, "tint": 53, "temperature": -79, "saturation": 27, "highlights": -25}, "source": "template"}
            """
    cnt = 0
    data = {}
    for item in data_:
        if item['source'] not in ["generated"]:
            continue
        group_id = item['group_id']
        instruction = item['instruction']
        params = item['params']
        score = item['score']
        
        if group_id not in data:
            data[group_id] = {
                "origin_edit": [],
                "refine_edit": []
            }
        
        if score >= 9:
            data[group_id]["origin_edit"].append((instruction, params))
        elif score >= 5:
            data[group_id]["refine_edit"].append((instruction, params))
            cnt += 1
    
    train_grpo_data = []
    test_grpo_data = []
    
    with tqdm(range(cnt), desc="Processing data") as pbar:
        
        for group_id, group_data in data.items():
            
            for item in group_data["refine_edit"]:
                pbar.update(1)
                refined_instruction, refined_params = item
                refined_instruction_id = hash(refined_instruction) % 100000
                origin_instruction, origin_params = random.choice(group_data["origin_edit"])
                origin_instruction_id = hash(origin_instruction) % 100000
                origin_image_id, origin_image_path = random.choice(image_pool)
                id = f"{origin_image_id}_{group_id}@{refined_instruction_id}@{origin_instruction_id}"
                
                reference_image_path = os.path.join(config.PROJECT_ROOT, 'datasets', 'GIER-Synthesis', "processed", f'{origin_image_id}_{group_id}@{origin_instruction_id}.jpg')
                refined_image_path = os.path.join(config.PROJECT_ROOT, 'datasets', 'GIER-Synthesis', "processed", f'{origin_image_id}_{group_id}@{refined_instruction_id}.jpg')
                merged_image_path = os.path.join(config.PROJECT_ROOT, 'datasets', 'GIER-Synthesis', "merge", f'{origin_image_id}_{group_id}@{origin_instruction_id}_merged.jpg')
                
                # process image with params
                editor = ImageEditor(str(origin_image_path))
                editor.process_image(params=refined_params, save_path=str(refined_image_path).replace("processed", "processed-2"))
                editor.process_image(params=origin_params, save_path=str(reference_image_path).replace("processed", "processed-2"))
                # merge_images(origin_image_path, reference_image_path, merged_image_path)
                merge_images(origin_image_path, str(reference_image_path).replace("processed", "processed-2"), str(merged_image_path).replace("merge\\", "merge-2\\"))
                
                # distill user edit command
                refine_instruction = generate_second_instruction(old_params=origin_params, new_params=refined_params, old_instruction=origin_instruction, new_instruction=refined_instruction)
                
                output = f'<answer>\n```json\n{json.dumps(refined_params, indent=4)}\n```\n</answer>'
                
                data = {
                    "data_source": "Image-Edit-Refine-Synthesis",
                    "prompt": [{
                        "role": "system",
                        "content": prompts.SFT_REFINE_SYSTEM_PROMPT,
                    }, {
                        "role": "user",
                        "content": prompts.SFT_REFINE_USER_PROMPT.replace("{origin_instruction}", origin_instruction).replace("{refine_instruction}", refine_instruction).replace("{old_params}", json.dumps(origin_params, indent=4)),
                    }],
                    "images": [{
                        "image_url": "file://" + str(project_root_server + "/" + os.path.relpath(merged_image_path, config.PROJECT_ROOT)).replace("\\", "/"),
                    }],
                    "ability": "image editing refine",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": output
                    },
                    "extra_info": {
                        "id": id,
                        "origin_params": origin_params,
                        "refined_params": refined_params,
                        "origin_instruction": origin_instruction,
                        "refine_instruction": refine_instruction,
                        "refined_instruction": refined_instruction,
                        "origin_image_path": str(project_root_server + "/" + os.path.relpath(origin_image_path, config.PROJECT_ROOT).replace('\\', '/')),
                        "reference_image_path": str(project_root_server + "/" + os.path.relpath(reference_image_path, config.PROJECT_ROOT).replace('\\', '/')),
                        "refined_image_path": str(project_root_server + "/" + os.path.relpath(refined_image_path, config.PROJECT_ROOT).replace('\\', '/')),
                        "task_type": "image_editing_refine",
                        "inst_type": "refine",
                    },
                }
                
                data_ = {
                    "data_source": "Image-Edit-Refine-Synthesis",
                    "prompt": [{
                        "role": "system",
                        "content": prompts.SFT_REFINE_SYSTEM_PROMPT,
                    }, {
                        "role": "user",
                        "content": prompts.SFT_REFINE_USER_PROMPT.replace("{origin_instruction}", origin_instruction).replace("{refine_instruction}", refined_instruction).replace("{old_params}", json.dumps(origin_params, indent=4)),
                    }],
                    "images": [{
                        "image_url": "file://" + str(project_root_server + "/" + os.path.relpath(merged_image_path, config.PROJECT_ROOT)).replace("\\", "/"),
                    }],
                    "ability": "image editing refine",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": output
                    },
                    "extra_info": {
                        "id": id,
                        "origin_params": origin_params,
                        "refined_params": refined_params,
                        "origin_instruction": origin_instruction,
                        "refine_instruction": refine_instruction,
                        "refined_instruction": refined_instruction,
                        "origin_image_path": str(project_root_server + "/" + os.path.relpath(origin_image_path, config.PROJECT_ROOT).replace('\\', '/')),
                        "reference_image_path": str(project_root_server + "/" + os.path.relpath(reference_image_path, config.PROJECT_ROOT).replace('\\', '/')),
                        "refined_image_path": str(project_root_server + "/" + os.path.relpath(refined_image_path, config.PROJECT_ROOT).replace('\\', '/')),
                        "task_type": "image_editing_refine",
                        "inst_type": "refined",
                    },
                }
                
                if random.random() > train_split:
                    test_grpo_data.append(data)
                else:
                    train_grpo_data.append(data)
                
                if random.random() > train_split:
                    test_grpo_data.append(data_)
                else:
                    train_grpo_data.append(data_)
    
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
    dataset_path = r"../datasets/GIER-Synthesis/instructions_2000.jsonl"
    project_root_server = config.PROJECT_ROOT
    # prepare_grpo_dataset_summary(dataset_path, project_root_server)
    # prepare_grpo_dataset_edit(dataset_path, project_root_server)
    prepare_grpo_dataset_refine(dataset_path, project_root_server)
