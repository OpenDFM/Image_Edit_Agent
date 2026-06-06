import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GIER_EDIT_DIR = PROJECT_ROOT / "datasets" / "GIER-Edit"
GIER_SUMMARY_DIR = PROJECT_ROOT / "datasets" / "GIER-Summary"
grpo_data_path = [
    GIER_EDIT_DIR / "grpo_synthesis_refine_test.json",
    GIER_EDIT_DIR / "grpo_synthesis_refine_test_adobe5k.json",
    GIER_EDIT_DIR / "grpo_synthesis_refine_train.json",
    GIER_EDIT_DIR / "grpo_synthesis_refine_train_adobe5k.json",
    GIER_EDIT_DIR / "grpo_synthesis_test.json",
    GIER_EDIT_DIR / "grpo_synthesis_test_adobe5k.json",
    GIER_EDIT_DIR / "grpo_synthesis_train.json",
    GIER_EDIT_DIR / "grpo_synthesis_train_adobe5k.json",
    GIER_SUMMARY_DIR / "grpo_synthesis_test.json",
    GIER_SUMMARY_DIR / "grpo_synthesis_test_adobe5k.json",
    GIER_SUMMARY_DIR / "grpo_synthesis_train.json",
    GIER_SUMMARY_DIR / "grpo_synthesis_train_adobe5k.json",
]

replace_template = [("First, observe and provide your thoughts about what was modified to optimize the input image. Then, please summarize", "Directly summarize"), ("First, provide your thoughts about how to optimize the original image based on current edit and new refine instruction. Then, output", "Directly output"),
    ("First, provide your thoughts about how to optimize the input image. Then, output", "Directly output")]


def make_repo_relative(value):
    if not isinstance(value, str):
        return value

    project_root = str(PROJECT_ROOT)
    project_uri = PROJECT_ROOT.as_uri()
    return value.replace(project_uri, "../..").replace(project_root, "../..")


def convert_grpo_data(item):
    new_data = {}
    new_data['images'] = [make_repo_relative(item['images'][0]['image_url'])]
    new_data['conversations'] = item['prompt']
    for old, new in replace_template:
        new_data['conversations'][1]['content'] = new_data['conversations'][1]['content'].replace(old, new)
    new_data['conversations'].append({
        "role": "assistant",
        "content": item['reward_model']['ground_truth']
    })
    new_data['id'] = item['extra_info'].pop('id')
    
    new_data['extra_info'] = item['extra_info']
    
    for key, value in new_data['extra_info'].items():
        new_data['extra_info'][key] = make_repo_relative(value)
    
    new_data['extra_info']['task_type'] = item['data_source']
    return new_data


for data_path in grpo_data_path:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    new_data_list = []
    for item in data:
        new_data_list.append(convert_grpo_data(item))
    save_path = Path(str(data_path).replace('grpo_synthesis', 'sft_synthesis'))
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(new_data_list, f, ensure_ascii=False, indent=4)
    print(f"Converted data saved to {save_path}")
