import argparse
import json
import os
from pathlib import Path

from PIL import Image
import torch
from tqdm import tqdm

from diffusers import QwenImageEditPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.getenv("QWEN_IMAGE_EDIT_MODEL", "Qwen/Qwen-Image-Edit"),
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=PROJECT_ROOT / "user_study" / "data_selected_new_for_vlm.json",
    )
    parser.add_argument(
        "--origin-image-dir",
        type=Path,
        default=PROJECT_ROOT / "user_study" / "images" / "origin",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "user_study" / "images" / "qwen_image_edit",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = QwenImageEditPipeline.from_pretrained(args.model)
    print("pipeline loaded")
    pipeline.to(torch.bfloat16)
    pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=None)

    with args.input_json.open('r', encoding='utf-8') as f:
        data = json.load(f)

    with torch.inference_mode():
        for item in tqdm(data):
            item_id = item['id']
            origin_image_path = args.origin_image_dir / Path(item['origin_image_path']).name
            prompt = item['prompt']

            image = Image.open(origin_image_path).convert("RGB")
            inputs = {
                "image": image,
                "prompt": prompt,
                "generator": torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 50,
            }

            print("Processing image", item_id)
            print("Prompt:", prompt)
            print("Origin image path:", origin_image_path)
            output = pipeline(**inputs)
            output_image = output.images[0]
            save_path = args.output_dir / f"{item_id.split('_')[0]}_qwen-image-edit.jpg"
            output_image.save(save_path)
            print(f"image {item_id} saved at", save_path.resolve())


if __name__ == "__main__":
    main()
