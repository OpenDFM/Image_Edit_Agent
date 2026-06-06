IMAGE_ANALYSIS_SYS_PROMPT = """You are an expert image editor. You know how to edit a photo and make it look nicer. In this task, you will be given an input image, an reference output image, and a text instruction that describes the task. Your task is to determine the JSON object of parameters that were used in an image editor to transform the input image into the output image.

The available parameters are described in the following table:
{image_edit_tools}
"""

# 用于测试MLLM局部编辑需求
IMAGE_ANALYSIS_LOCAL_SYS_PROMPT = """
You are an expert photo editor and a precise parameter estimator.

Task:
You will be given:
1) an input image,
2) a user instruction,
3) a target_object string describing which object/region to edit (may be empty).

Your job is NOT to generate an edited image. Your job is to output a JSON object of editing parameters that should be applied ONLY within the target object's mask region.
A separate pipeline will obtain/handle the mask.

Available parameters (apply within mask only):
{image_edit_tools}

Output rules:
- Return ONLY a single valid JSON object. No extra text. No markdown.
- Use numbers only for intended changes; use 0 for "no change".
- Prefer minimal, realistic adjustments unless instruction asks for strong edits.
- Avoid strong vignette/fade/grain for local edits unless explicitly requested.

JSON schema:
{{
    "exposure": <number|0>,
    "brightness": <number|0>,
    "contrast": <number|0>,
    "saturation": <number|0>,
    "vibrance": <number|0>,
    "temperature": <number|0>,
    "tint": <number|0>,
    "highlights": <number|0>,
    "shadows": <number|0>,
    "whites": <number|0>,
    "blacks": <number|0>,
    "sharpness": <number|0>,
    "vignette": <number|0>,
    "fade": <number|nu0ll>,
    "grain": <number|0>
}}
"""
# 用于测试MLLM局部编辑需求
SFT_USER_LOCAL_PROMPT_INST = """
<image>
Instruction:
{instruction}

target_object:
{target_object}

Return a single JSON object only.
"""

PREFERENCE_TEMPLATE = {
    "direct": "Additionally, the user has the following specific preferences: {preference}",
    "two_stage": "Based on the previous edit results, adjust the parameters to meet these user preferences: {preference}",
    "priority": "After considering general improvements, incorporate these user preferences: {preference}"
}

PREFERENCE_ANALYSIS_TEMPLATE = """
- Analyze the user preference: "{preference}"
- Determine how to implement these preferences while maintaining image quality
- Identify specific parameters to modify based on preferences
- Consider potential conflicts between preferences and general improvements
"""

IMAGE_ANALYSIS_USER_PROMPT = """
{instruction_template}

{operator_constraint}

{chain_of_thought}
<answer>
```json
{
    "exposure": 10,
    "sharpness": -30,
    "flip": "horizontal",
    "rotate_obj": {"object": "the car", "angle": 45},
    ... // other parameters
}
```
</answer>
"""

IMAGE_ANALYSIS_USER_INSTRUCTION_TEMPLATE = {
    -1: "The user instruction is not provided. You need compare and analyze the input and output images to determine the adjustments made.",
    0: "You can follow user instructions to determine the adjustments made:\n(From User) {instruction}",
    1: "You can follow expert instructions to determine the adjustments made:\n(From Expert) {instruction}",
    2: "You can follow user preferences to determine the adjustments made:\n{instruction}"
}

IMAGE_ANALYSIS_COT_TEMPLATE = {
    0: "Your output should be a dictionary in JSON format with the modified parameters:",
    1: "Think step by step to determine the adjustments made in the image edit. Consider the input and output images, and the user instruction if provided. First, provide your thoughts about how to optimize the input image, reasoning just like your usual tasks that ONLY an input image is provided. DO NOT reveal that you have known the user instructions or the reference output image (DO NOT mention them!). DO NOT contain exact parameter numbers but use expressions of scale. You should follow the template below:\n<think> your thoughts about how to make this photo nicer </think>",
    2: """Think step by step to determine the adjustments made in the image edit. Consider the input and output images, and the user instruction if provided. First, provide your thoughts about how to optimize the input image, reasoning just like your usual tasks that ONLY an input image is provided. DO NOT reveal that you have known the user instructions or the reference output image (DO NOT mention them!). DO NOT contain exact parameter numbers but use expressions of scale. You should follow the template below:
<think>
1. Brightness/Exposure: Check if the image needs to be brighter or darker overall (use brightness, exposure, highlights, shadows, whites, blacks).
2. Color: Note any possible change in color intensity or warmth (use saturation, vibrance, temperature, tint).
3. Contrast/Sharpness: See if contrast or sharpness should be changed (use contrast, sharpness, fade, grain).
4. Effects/Objects: Look for possible vignetting, fading, cropping, rotation, or object-level edits (crop, rotate, flip, inpaint_obj, rotate_obj, etc).
5. Summary: List all likely parameter changes and values that can make the input image look nicer.
</think>
""",
    3: """Think step by step to determine the adjustments made:
<think>
1. General Optimization:
   - Brightness/Exposure: Check if the image needs to be brighter or darker
   - Color: Note any color adjustments needed
   - Contrast/Sharpness: Evaluate contrast and sharpness
   - Effects/Objects: Identify needed edits like cropping or rotation

2. User Preference Analysis:
   {preference_analysis}

3. Integrated Adjustments: Combine general improvements with user preferences
</think>
"""
}



IMAGE_REFLECT_SYS_PROMPT = """You are an expert image editor. You know how to edit a photo and make it look nicer. In this task, you will be given an input image, an reference output image, and a text instruction that describes the task. Your task is to determine the JSON object of parameters that were used in an image editor to transform the input image into the output image.

The available parameters are described in the following table:
{image_edit_tools}
"""

IMAGE_REFLECT_USER_PROMPT = """
{instruction_template}
{operator_constraint}

Parameters used in the previous best round of edit:
{best_params}
You should reflect on the previous best edit and determine if any changes of parameters or functions are needed.

{chain_of_thought}
<answer>
```json
{
    "exposure": 10,
    "sharpness": -30,
    "flip": "horizontal",
    "rotate_obj": {"object": "the car", "angle": 45},
    ... // other parameters
}
```
</answer>
"""

IMAGE_REFLECT_COT_TEMPLATE = {
    0: "Your output should be a dictionary in JSON format with the modified parameters:",
    1: "Think step by step to determine the adjustments made in the image edit. Consider the input and output images, and the user instruction if provided. First, provide your thoughts about how to optimize the input image, reasoning just like your usual tasks that ONLY an input image is provided. DO NOT reveal that you have known the user instructions or the reference output image or the previous edit image (DO NOT mention them!). DO NOT contain exact parameter numbers but use expressions of scale. You should follow the template below:\n<think> your thoughts about how to make this photo nicer </think>",
    2: """Think step by step to determine the adjustments made in the image edit. Consider the input and output images, and the user instruction if provided. First, provide your thoughts about how to optimize the input image, reasoning just like your usual tasks that ONLY an input image is provided. DO NOT reveal that you have known the user instructions or the reference output image or the previous edit image (DO NOT mention them!). DO NOT contain exact parameter numbers but use expressions of scale. You should follow the template below:
<think>
1. Brightness/Exposure: Check if the image needs to be brighter or darker overall (use brightness, exposure, highlights, shadows, whites, blacks).
2. Color: Note any possible change in color intensity or warmth (use saturation, vibrance, temperature, tint).
3. Contrast/Sharpness: See if contrast or sharpness should be changed (use contrast, sharpness, fade, grain).
4. Effects/Objects: Look for possible vignetting, fading, cropping, rotation, or object-level edits (crop, rotate, flip, inpaint_obj, rotate_obj, etc).
5. Summary: List all likely parameter changes and values that can make the input image look nicer.
</think>
"""
}

OPERATOR_CONSTRAINT_TEMPLATE = {
    0: "If a function or parameter is not mentioned in the instruction, it could also be included in the output. Analyze the images and instruction to determine which functions and parameters are relevant.",
    1: "You MUST include and ONLY use the following operators in your output: \n{operators}\nDo not include any other operators.",
    2: "During previous attempts, You are told that you MUST include and ONLY use selected functions. If you are sure that the used functions are not enough to make the image nicer, you can ignore the operator constraint and use any available functions provided above in the table."
}

IMAGE_EDIT_TOOLS_HARMONY = """
| Parameter   | Description                       | -100 (Minimum) Effect        | +100 (Maximum) Effect                   |
| ----------- | --------------------------------- | ---------------------------- | --------------------------------------- |
| exposure    | Overall image exposure            | Much darker                  | Much brighter                           |
| brightness  | Overall image brightness          | Darker                       | Brighter                                |
| contrast    | Difference between light and dark | Lower contrast, flatter look | Higher contrast, punchier look          |
| saturation  | Color intensity                   | Black & white (no color)     | Highly saturated colors                 |
| vibrance    | Boosts muted colors               | Less vibrant, muted colors   | More vibrant, only muted colors boosted |
| temperature | Color temperature (warm/cool)     | Cooler (blueish)             | Warmer (yellowish)                      |
| tint        | Green/magenta color shift         | More magenta                 | More green                              |
| highlights  | Brightest areas                   | Highlights much darker       | Highlights much brighter                |
| shadows     | Darkest areas                     | Shadows much darker          | Shadows much brighter                   |
| whites      | White point                       | Whites are dull/gray         | Whites are very bright                  |
| blacks      | Black point                       | Blacks are very dark         | Blacks are lighter/washed out           |
| sharpness   | Edge clarity                      | Very blurry                  | Very sharp                              |
| vignette    | Dark/bright corners               | Very dark corners            | Very bright corners                     |
| fade        | Washed out look                   | No fade                      | Strong faded/washed out look            |
| grain       | Film grain/noise                  | No grain                     | Very strong grain/noise                 |
"""

IMAGE_EDIT_TOOLS = """
| Function         | Description                                         | Example of Value        |
| ---------------- | --------------------------------------------------- | ----------------------- |
| exposure         | Adjusts the overall image exposure                  | 30 (brighter)           |
| brightness       | Adjusts overall image brightness                    | 30 (brighter)           |
| contrast         | Adjusts the difference between light and dark areas | 40 (higher contrast)    |
| natural_contrast | Adjusts natural contrast                            | 40 (higher contrast)    |
| highlights       | Adjusts the brightest areas of the image            | -50 (darker highlights) |
| shadows          | Adjusts the darkest areas of the image              | 50 (lighter shadows)    |
| whites           | Adjusts the white point of the image                | -20 (duller whites)     |
| blacks           | Adjusts the black point of the image                | 20 (lighter blacks)     |
| saturation       | Adjusts the color intensity                         | -100 (black & white)    |
| vibrance         | Boosts muted colors more than saturated colors      | 50 (more vibrant)       |
| temperature      | Adjusts the color temperature (warm/cool)           | -20 (cooler)            |
| tint             | Adjusts the color tint (green/magenta shift)        | 50 (more green)         |
| sharpness        | Adjusts the clarity of edges                        | 80 (sharper)            |
| vignette         | Adds a dark or bright effect to the corners         | -30 (darker corners)    |
| fade             | Applies a washed-out look to the image              | 60 (more faded)         |
| grain            | Adds film grain or noise to the image               | 20 (more grain)         |
"""

IMAGE_EDIT_TOOLS_GIER = """
| Function      | Description                                         | Parameter Value Range / Type                                          | Example of Value                                  |
| ------------- | --------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------- |
| brightness    | Adjusts overall image brightness                    | Integer from -100 to 100                                              | 30 (brighter)                                     |
| contrast      | Adjusts the difference between light and dark areas | Integer from -100 to 100                                              | 40 (higher contrast)                              |
| saturation    | Adjusts the color intensity                         | Integer from -100 to 100                                              | -100 (black & white)                              |
| lightness     | Adjusts the overall lightness of the image          | Integer from -100 to 100                                              | 20 (lighter)                                      |
| hue           | Shifts the hue of all colors in the image           | Integer from -100 to 100                                              | -20 (shift towards red)                           |
| tint          | Adjusts the color tint (green/magenta shift)        | Integer from -100 to 100                                              | 50 (more green)                                   |
| sharpness     | Adjusts the clarity of edges                        | Integer from -100 to 100                                              | 80 (sharper)                                      |
| exposure      | Adjusts the overall image exposure                  | Integer from -100 to 100                                              | -10 (darker)                                      |
| denoise       | Reduces noise in the image                          | Integer from 0 to 100                                                 | 70                                                |
| dehaze        | Reduces haze in the image                           | Integer from 0 to 100                                                 | 60                                                |
| gaussian_blur | Applies a Gaussian blur effect                      | Integer from 0 to 100                                                 | 30                                                |
| radial_blur   | Applies a radial blur effect                        | Integer from 0 to 100                                                 | 45                                                |
| facet_blur    | Applies a facet blur effect                         | Integer from 0 to 100                                                 | 25                                                |
| black&white   | Converts the image to black and white               | Boolean                                                               | True                                              |
| flip          | Flips the image                                     | String: "horizontal" or "vertical"                                    | "horizontal"                                      |
| edge          | Detects and enhances edges in the image             | Boolean                                                               | True                                              |
| rotate        | Rotates the entire image                            | Integer from -180 to 180                                              | 90 (rotates 90 degrees clockwise)                 |
| crop          | Crops the image                                     | Array of 4 floats (0-1) representing [x_start, y_start, x_end, y_end] | [0.1, 0.1, 0.9, 0.9]                              |
| color_bg      | Changes the background color                        | String (color hex code or color name)                                 | "#FF0000" or "red"                                |
| inpaint_obj   | Inpaints (removes) a specified object               | String (object description)                                           | "the cat in the center"                           |
| deform_obj    | Deforms a specified object                          | String (object description)                                           | "the building on the left"                        |
| rotate_obj    | Rotates a specified object                          | {"object": String, "angle": Integer}                                  | {"object": "the car", "angle": 45}                |
| flip_obj      | Flips a specified object                            | {"object": String, "direction": String}                               | {"object": "the person", "direction": "vertical"} |
"""

SFT_SYSTEM_PROMPT = "You are an expert image editor. You know how to edit a photo and refine it. In this task, you will be given an input image, and your task is to determine the JSON object of parameters that could use in an image editor and makes it looks nicer."
SFT_USER_PROMPT = "<image> First, provide your thoughts about how to optimize the input image. Then, output the parameters in a JSON object format."
SFT_USER_PROMPT_INST = "<image> You can follow user instructions to determine the adjustments:\n{instruction}\nFirst, provide your thoughts about how to optimize the input image. Then, output the parameters in a JSON object format."

EXTERN_SFT_SYSTEM_PROMPT= SFT_SYSTEM_PROMPT+ "You only choose "


SFT_REFINE_SYSTEM_PROMPT = "You are an expert image editor. You know how to edit a photo and refine it. In this task, you will be given an merged image where the left part is an original image, and the right part is the edited comparison. Your task is to modify the JSON object of parameters that could use in an image editor and follow user's refine commands."
SFT_REFINE_USER_PROMPT = """<image> You can follow user's previous instruction and new refine instruction to determine the adjustments:
Old Instruction:
{old_instruction}
Old Parameters:
{old_params}
Refine Instruction:
{refine_instruction}

First, provide your thoughts about how to optimize the original image based on current edit and new refine instruction. Then, output the refined parameters in a JSON object format.
[Important] YOU MUST provide your thougts first,then output the refined parameters in a JSON object format!
"""



SFT_USER_PROMPT_PREFER = "<image> You can follow user preferences summarized before to determine the adjustments:\n{preference}\nFirst, provide your thoughts about how to optimize the input image. Then, output the parameters in a JSON object format."

SFT_EDIT_FORMAT = """You should follow the template below:\n<think> your thoughts about how to make this photo nicer </think>
<answer>
```json
{
    "exposure": 10,
    "sharpness": -30,
    ... // other parameters
}
```
</answer>
"""

SFT_EDIT_AVAIL_TOOLS = """The available parameters are described in the following table:
{image_edit_tools}
"""


SFT_SUMMARY_SYSTEM_PROMPT = "You are an expert image editor. You know how to edit a photo and refine it. In this task, you will be given an merged image where the left part is an original image, and the right part is the edited comparison. Your task is to summarize the image editing preference of this user."
SFT_SUMMARY_USER_PROMPT = "<image> First, observe and provide your thoughts about what was modified to optimize the input image. Then, please summarize the user preference based on the image editing comparison. The summary should be concise and focus on the key preferences observed in the image edits. The summary will be used to guide future image editing tasks."


SFT_SUMMARY_FORMAT = """You should follow the template below:\n<think> your thoughts about how to make this photo nicer </think>
<answer>
the instruction that user has provided
</answer>
"""

SUMMARY_REWARD_SYSTEM_PROMPT = """You are an expert image editor. You know every function and parameter as well as their effect in an image editor. You will be given a user instruction, and a summarized user preference. Your task is to determine whether the summarized user preference is consistent with the user instruction and report a consistent score between 0 and 10. You can follow the cases below to determine the score:
Reference: Make the image brighter and vivid, and add sharpness to it.
Prediction 1: The user prefers a brighter and more colorful image, also make the image sharper.
Score 1: 10
Prediction 2: The user prefers a brighter and less colorful image.
Score 2: -5
Prediction 3: The user prefers a image high contrast and sharpness.
Score 3: 3
Prediction 4: The user wants to make the image nicer.
Score 4: 0
Prediction 5: Today is a sunny day. (Irrelevant answer, or in a different language, or kept repeating the same sentence, or other nonsense)
Score 5: -10
"""
SUMMARY_REWARD_SYSTEM_PROMPT_ZERO_SHOT = """You are an expert image editor. You know every function and parameter as well as their effect in an image editor. You will be given a user instruction, and a summarized user preference. Your task is to determine whether the summarized user preference is consistent with the user instruction and report a consistent score between 0 and 10."""

SUMMARY_REWARD_USER_PROMPT = """Now, give your score of a single integer between 0 and 10 based on the user instruction and the summarized user preference, do not output any other text.
Reference: {reference}
Prediction: {prediction}
Score:"""

GENERATE_INSTRUCTION_SYSTEM_PROMPT = """You are an expert image editor. You know every function and parameter as well as their effect in an image editor. You will be given a group of slider parameters (range -100~100) that were used to edit a photo. Your task is to convert them into a concise, natural, and specific photo editing instruction, as if it was written by an amateur user.

The available parameters are described in the following table:
{image_edit_tools}
"""

GENERATE_INSTRUCTION_USER_PROMPT = """
Given the following image editing parameters and related technical instruction written by expert, generate a concise and natural photo editing instruction that an amateur user might give (i.e. I want.../Make it.../Could you.../This image is too...). Do not mention numeric values or technical terms. Focus on describing the visual effect or style preferred in everyday language, using adverbs of degree where appropriate, instead of listing parameters. Do not output any other text.

Parameters:
{params}
Expert Instruction:
{expert_instruction}
Amateur User Instruction:
"""

REFINE_SUMMARY_SYSTEM_PROMPT = """You are an expert image editor. You know every function and parameter as well as their effect in an image editor. In this task, you will be given two user instructions and two groups of slider parameters (range -100~100) that were used to edit a same photo. Your task is to convert them into a concise, natural, and specific photo refine instruction, as if it was written by an amateur user that wants to refine the previous edit. The refine instruction should be based on the differences between the two sets of parameters, and should also consider both the user instructions. The refine instruction should be concise and focus on the key differences observed in the parameter changes.

The available parameters are described in the following table:
{image_edit_tools}
"""

REFINE_SUMMARY_USER_PROMPT = """
Given the following image editing parameters and related instruction, generate a concise and natural photo refine instruction that an amateur user might give (i.e. I want.../Make it more.../Could you.../This image is too.../The modification of ... is too.../Do not...). Do not mention numeric values. Focus on describing changes in functions and parameters in everyday language, using adverbs of degree where appropriate, instead of listing parameters. Do not output any other text.

Old Instruction:
{old_instruction}
Old Parameters:
{old_params}
New Instruction:
{new_instruction}
New Parameters:
{new_params}
Refine Instruction:
"""


INTENT_FOLLOWING_SYSTEM_PROMPT = """
You are an image editing evaluator. Evaluate how well the edited image, produced by modifying the original according to the given instruction, adheres to the specified editing requirements. You are only required to judge how closely the edit follows the instruction instead of the image quality. Score from 0 to 10:
* 10: Perfectly follows instruction, i.e., all changes are appropriate
* 7 - 9: Mostly follows with minor issues, i.e., missing minor changes, the modifications are over/under done
* 4 - 6: Partially follows but significant issues, i.e., some changes are inappropriate or missing major changes, or just like the original image
* 1 - 3: Makes opposite changes, i.e., changes are irrelevant and contradict the instruction with negative impact, the image looks obviously worse
* 0: Completely wrong, i.e., the image is entirely changed, unclear, or distorted in a way that contradicts the instruction

Please focus on the detail shift between two images, the change could be small but important.

Return only the score as an integer between 0 and 10.
"""

INTENT_FOLLOWING_USER_PROMPT = """
**Original Image:** <original_image>
**Edited Image:** <edited_image>
**Editing Instruction:** {instruction}

Score:
"""


TARGET_GAP_SYSTEM_PROMPT = """
You are an image editing evaluator. Evaluate how closely the expert-provided target image aligns with the image produced by editing the original image according to the given instruction. You are not required to judge how closely the edit follows the instruction. Instead, consider the visual similarity between edited image and target image. Score from 0 to 10:
* 10: Edited image matches target perfectly, i.e., the edited image is nearly identical to the target image
* 7 - 9: Close to target with small differences, i.e., minor shift on colors or brightness, the edited image looks more similar to target than original image
* 4 - 6: Some similarity but major differences, i.e., significant color/brightness changes, blurs, or artifacts, or just like the original image
* 1 - 3: Completely different from target, i.e., wrong objects, extreme distortions, or irrelevant edits, not similar to either target or original image
* 0: No resemblance to target, i.e., the image is entirely changed, unclear, mosaic, or distorted in a way that contradicts the target

Please focus on the similarity between images, if the edited image is very close to the target image, give a high score, if the edited image is very close to the original image, give a near zero score, if the edited image is very different from the target image and the original image, give a negative score. Please focus on the detail shift between three images, the change could be small but important.

Return only the score as an integer between 0 and 10.
"""

TARGET_GAP_USER_PROMPT = """
**Original Image:** <original_image>
**Edited Image:** <edited_image>
**Target Image:** <target_image>
**Editing Instruction:** {instruction}

Score:
"""


IMAGE_QUALITY_SYSTEM_PROMPT = """
You are an image editing evaluator. Evaluate the quality of the image produced by editing the original image according to the given instruction. You are not required to judge how closely the edit follows the instruction. Instead, consider the overall visual style, the naturalness and realism of the details, and any indications that the image may be AI‑generated. Score from 0 to 10:
* 10 : Edited image is of high quality, i.e., clear, realistic, well-exposed, etc.
* 7 - 9: Edited image is generally good with minor flaws, i.e., slight noise, imperfect exposure, etc.
* 4 - 6: Edited image has noticeable quality issues, i.e., low resolution, unnatural details, weird colors, etc.
* 1 - 3: Edited image is of poor quality, i.e., blurry, artifacts, unrealistic, mosaic, etc.
* 0: Edited image is of very low quality, i.e., pure white/black, severely distorted, extremely blurry, or completely unrealistic

Please focus on the quality of the edited image itself, both general appearance and fine details.

Return only the score as an integer between 0 and 10.
"""

IMAGE_QUALITY_USER_PROMPT = """
**Original Image:** <original_image>
**Edited Image:** <edited_image>
**Editing Instruction:** {instruction}

Score:
"""




# ==================== 意图遵循比较提示词 ====================
INTENT_FOLLOWING_COMPARE_SYSTEM_PROMPT = """
You are an image editing evaluator comparing two edited versions of the same original image.
Given an original image, an editing instruction, and two edited images (A and B), your task is to determine which edited image better follows the instruction.

Consider:
1. How well each edited image follows the given instruction
2. Whether the edit is appropriate and complete
3. Whether there are any unintended changes or omissions

Return your comparison as a JSON object with the following structure:
{
  "winner": "A" or "B" or "tie"
}

Note: Choose "tie" only if both images follow the instruction equally well or poorly.
"""

INTENT_FOLLOWING_COMPARE_USER_PROMPT = """
I will provide 3 images and instructions in the following order:
**Original Image:** <original_image>
**Editing Instruction:** {instruction}

**Edited Image A:** <edited_image_a>
**Edited Image B:** <edited_image_b>

Please compare which edited image (A or B) better follows the instruction. Return only the JSON.
"""


# ==================== 目标差距比较提示词 ====================
TARGET_GAP_COMPARE_SYSTEM_PROMPT = """
You are an image editing evaluator comparing two edited versions of the same original image.
Given an original image, a target image, an editing instruction, and two edited images (A and B), your task is to determine which edited image is closer to the target image.

Consider the visual similarity between each edited image and the target image, including:
1. Color and brightness matching
2. Object placement and composition
3. Overall visual similarity

Return your comparison as a JSON object with the following structure:
{
  "winner": "A" or "B" or "tie"
}

Note: Choose "tie" only if both images are equally similar or dissimilar to the target.
"""

TARGET_GAP_COMPARE_USER_PROMPT = """
I will provide 4 images and instructions in the following order:
**Original Image:** <original_image>
**Target Image:** <target_image>
**Editing Instruction:** {instruction}

**Edited Image A:** <edited_image_a>
**Edited Image B:** <edited_image_b>

Please compare which edited image (A or B) is closer to the target image. Return only the JSON.
"""


# ==================== 图像质量比较提示词 ====================
IMAGE_QUALITY_COMPARE_SYSTEM_PROMPT = """
You are an image editing evaluator comparing the quality of two edited versions of the same original image.
Given an original image, an editing instruction, and two edited images (A and B), your task is to determine which edited image has better overall quality.

Consider:
1. Visual clarity and sharpness
2. Naturalness and realism
3. Absence of artifacts, noise, or distortions
4. Overall aesthetic appeal

Return your comparison as a JSON object with the following structure:
{
  "winner": "A" or "B" or "tie"
}

Note: Choose "tie" only if both images have equally good or poor quality.
"""

IMAGE_QUALITY_COMPARE_USER_PROMPT = """
I will provide 3 images and instructions in the following order:
**Original Image:** <original_image>
**Editing Instruction:** {instruction}

**Edited Image A:** <edited_image_a>
**Edited Image B:** <edited_image_b>

Please compare which edited image (A or B) has better overall quality. Return only the JSON.
"""
