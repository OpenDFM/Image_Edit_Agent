dataset_list = [[r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_refine_test.json", "test", 0.2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_refine_test_adobe5k.json", "test", 0.2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_test.json", "test", 0.02],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_test_adobe5k.json", "test", 0.02],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_synthesis_test.json", "test", 0.2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_synthesis_test_adobe5k.json", "test", 0.2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_refine_train.json", "train", 2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_refine_train_adobe5k.json", "train", 2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_train.json", "train", 0.2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_train_adobe5k.json", "train", 0.2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_synthesis_train.json", "train", 1],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_synthesis_train_adobe5k.json", "train", 1],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_train.json", "train", 2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_train.json", "train", 5],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_test.json", "test", 0.04],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_test.json", "test", 0.2], ]

dataset_list = [[r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_refine_test.json", "test", 0.2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_refine_test_adobe5k.json", "test", 0.2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_test.json", "test", 0.02],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_test_adobe5k.json", "test", 0.02],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_synthesis_test.json", "test", 0.2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_synthesis_test_adobe5k.json", "test", 0.2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_refine_train.json", "train", 2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_refine_train_adobe5k.json", "train", 2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_train.json", "train", 0.2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_train_adobe5k.json", "train", 0.2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_synthesis_train.json", "train", 1],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_synthesis_train_adobe5k.json", "train", 1],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_train.json", "train", 2],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_train.json", "train", 5],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_test.json", "test", 0.04],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_test.json", "test", 0.2], ]

dataset_list = [[r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_refine_train.json", "train", 1],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_refine_train_adobe5k.json", "train", 1],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_train.json", "train", 1],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_synthesis_train_adobe5k.json", "train", 1],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_synthesis_train.json", "train", 1],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_synthesis_train_adobe5k.json", "train", 1],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Edit\sft_train.json", "train", 1],
                [r"D:\Codes\Image_Edit_Agent\datasets\GIER-Summary\sft_train.json", "train", 1]]

test_output = r"D:\Codes\Image_Edit_Agent\datasets\GIER_SFT\test_sft_synthesis_20250908.json"
train_output = r"D:\Codes\Image_Edit_Agent\datasets\GIER_SFT\train_sft_synthesis_20250908.json"

import pandas as pd
import os
import json

test_data = []
train_data = []

for dataset in dataset_list:
    dataset_path, dataset_type, copy_ratio = dataset
    
    df = pd.read_json(dataset_path)
    
    df_copy = df.sample(frac=copy_ratio, replace=True, random_state=42)
    
    print(f"Loaded {len(df_copy)} records from {dataset_path} for {dataset_type} dataset.")

#     if dataset_type == "test":
#         test_data.append(df_copy)
#     elif dataset_type == "train":
#         train_data.append(df_copy)
#
# # Concatenate all test and train data
# test_df = pd.concat(test_data, ignore_index=True)
# train_df = pd.concat(train_data, ignore_index=True)
#
# # Shuffle the data
# test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
# train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # Save to json files
# test_df.to_json(test_output, orient='records', lines=False, force_ascii=False, indent=4)
# train_df.to_json(train_output, orient='records', lines=False, force_ascii=False, indent=4)
# print(f"Test data saved to {test_output} with {len(test_df)} records.")
# print(f"Train data saved to {train_output} with {len(train_df)} records.")
