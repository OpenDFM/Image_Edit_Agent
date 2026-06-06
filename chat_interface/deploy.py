# chat_interface/deploy.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import argparse
import os
import uuid
from werkzeug.utils import secure_filename
import json
import traceback
import cv2
import numpy as np
import re
import requests
import sys
# add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.image_editor import ImageEditor
from src import config
from src.eval_all import predict_operations_gier_edit_eval, predict_operations_gier_refine_eval, \
    predict_operations_gier_summary_eval
from src.external_caller import task_analysis, reflection_summary, call_external_image_edit
from src.llm_caller import call_vlm_api
import src.prompts as prompts

# 设置不压缩图片
config.IMAGE_RESIZE = -1

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'tmp'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def extract_json_from_response(response_text):
    """
    从IEA响应中提取JSON参数，支持多种格式
    """
    # 方法1: 尝试提取```json```包裹的内容
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 方法2: 尝试提取整个响应中的JSON对象
    # 查找第一个{和最后一个}之间的内容
    start = response_text.find('{')
    end = response_text.rfind('}')

    if start != -1 and end != -1 and start < end:
        json_str = response_text[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # 方法3: 尝试提取可能的多行JSON
    lines = response_text.split('\n')
    json_lines = []
    in_json = False

    for line in lines:
        if line.strip().startswith('{'):
            in_json = True
        if in_json:
            json_lines.append(line.strip())
        if line.strip().endswith('}'):
            break

    if json_lines:
        json_str = ''.join(json_lines)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return None


def extract_cot_from_response(response_text):
    """
    从IEA响应中提取思维链内容（JSON之前的部分）
    """
    # 找到JSON开始的位置
    json_start = response_text.find('{')
    if json_start > 0:
        return response_text[:json_start].strip()
    return response_text.strip()

# 添加翻译功能
def translate_text(text, target_language):
    """翻译文本到目标语言，所有参数已固定"""
    try:
        external_base_url = os.getenv("IEA_EXTERNAL_BASE_URL", "")
        external_api_key = os.getenv("IEA_EXTERNAL_API_KEY", "")
        external_model = os.getenv("IEA_EXTERNAL_MODEL", "")
        if not external_base_url or not external_api_key or not external_model:
            return text

        # 检测文本语言并构建提示
        if contains_chinese(text):
            if target_language == 'en':
                # 中文翻译成英文
                system_prompt = "请将以下中文翻译成英文，要求翻译准确且简洁，不要添加多余内容"
                user_prompt = text
            else:
                return text  # 已经是中文，不需要翻译
        else:
            if target_language == 'zh':
                # 英文翻译成中文
                system_prompt = "请将以下英文翻译成中文，要求翻译准确且简洁，不要添加多余内容"
                user_prompt = text
            else:
                return text  # 已经是英文，不需要翻译

        # 构建请求参数（所有参数直接固定）
        json_dict = {
            "model": external_model,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            "max_tokens": 3000,
            "temperature": 0,
            "top_p": 1,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": config.authorization_header(external_api_key),
        }

        # 调用LLM API并处理重试（参数直接固定）
        retry_count = 0
        response_text = ""
        while retry_count < 3:  # 直接使用固定的重试次数
            try:
                response = requests.post(
                    external_base_url,
                    json=json_dict,
                    headers=headers,
                    timeout=60
                )
                response.raise_for_status()  # 抛出HTTP错误
                response_data = response.json()
                response_text = response_data["choices"][0]["message"]["content"]
                break  # 成功获取响应，退出重试循环
            except KeyboardInterrupt:
                raise Exception("用户终止了操作")
            except Exception as e:
                print(f"API调用错误: {str(e)}")
                retry_count += 1
                if retry_count < 3:
                    print(f"重试中...({retry_count}/3)")

        if not response_text:
            print("达到最大重试次数（3次），翻译失败")
            return text
        return response_text.strip()

    except Exception as e:
        print(f"翻译过程出错: {e}")
        return text  # 翻译失败时返回原文本


def contains_chinese(text):
    """检查文本是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def translate_instruction(instruction):
    """翻译用户指令"""
    if contains_chinese(instruction):
        # 中文指令翻译成英文传给后续prompt
        english_instruction = translate_text(instruction, 'en')
        return {
            'original': instruction,
            'translated': english_instruction,
            'is_chinese': True
        }
    else:
        # 英文指令翻译成中文用于显示
        chinese_instruction = translate_text(instruction, 'zh')
        return {
            'original': instruction,
            'translated': chinese_instruction,
            'is_chinese': False
        }


def translate_cot(cot_text):
    """翻译思维链内容"""
    if cot_text and contains_chinese(cot_text):
        # 如果思维链已经是中文，不需要翻译
        return {
            'original': cot_text,
            'translated': None,
            'is_chinese': True
        }
    else:
        # 英文思维链翻译成中文
        chinese_cot = translate_text(cot_text, 'zh')
        return {
            'original': cot_text,
            'translated': chinese_cot,
            'is_chinese': False
        }


# 拼接两张图片（保持不变）
def concatenate_images(image1_path, image2_path):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # 调整图片大小使其相同
    height = max(img1.shape[0], img2.shape[0])
    width = max(img1.shape[1], img2.shape[1])

    img1_resized = cv2.resize(img1, (width, height))
    img2_resized = cv2.resize(img2, (width, height))

    # 水平拼接
    concatenated = np.hstack((img1_resized, img2_resized))
    return concatenated


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/tmp/<filename>')
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/send_message', methods=['POST'])
def send_message():
    input_image_path = None
    if 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            # 保存原始图片（后续所有轮次都基于此）
            filename = f"original_{uuid.uuid4().hex}.{secure_filename(file.filename).split('.')[-1]}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            input_image_path = save_path

    message = request.form.get('message', '').strip()
    if not message or not input_image_path:
        return jsonify({'error': 'Message and image are required'}), 400

    try:
        # 1. 任务分析 - 调用外部大模型分析用户需求
        analysis_result = task_analysis(message, [input_image_path])

        # 2. 根据分析结果选择处理方式
        if not analysis_result['need_editing']:
            # 非修图任务，直接返回反馈
            return jsonify({
                'response': analysis_result['clear_instruction'],
                'user_message_original': message,
                'operations': {},
                'image': None
            })

        # 获取优化后的清晰指令
        clear_instruction = analysis_result['clear_instruction']
        model_to_use = analysis_result['model_to_use']

        # 翻译用户指令（用于显示）
        instruction_translation = translate_instruction(message)
        clear_instruction_translation = translate_instruction(clear_instruction)

        if model_to_use == "IEA":
            # 3. 使用IEA处理 - 获取原始响应（保持原有逻辑）
            processing_instruction = clear_instruction_translation['translated'] if clear_instruction_translation[
                'is_chinese'] else clear_instruction_translation['original']

            response, model, usage = call_vlm_api(
                system_prompt=prompts.SFT_SYSTEM_PROMPT,
                user_prompt=prompts.SFT_USER_PROMPT_INST.replace("{instruction}",
                                                                 processing_instruction) if processing_instruction else prompts.SFT_USER_PROMPT,
                model=config.VLM_DEFAULT_MODEL,
                image_list=[input_image_path]
            )

            # 4. 提取原始思维链并优化（核心修改）
            output_image_path = None
            if response:
                extracted_json = extract_json_from_response(response)
                extracted_cot = extract_cot_from_response(response)  # 原始思维链

                if extracted_json:
                    validated_params = extracted_json
                    editor = ImageEditor(input_image_path)
                    output_filename = f"edited_{uuid.uuid4().hex}.png"
                    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

                    try:
                        editor.process_image(validated_params, output_image_path)

                        # 直接调用 reflection_summary 优化思维链（无需新增提示词）
                        # 该函数内部已通过 call_external_vlm_api 调用外部模型
                        optimized_cot = reflection_summary(
                            original_instruction=clear_instruction,
                            reasoning_chain=extracted_cot,  # 传入原始思维链
                            tool_operations=str(validated_params),
                            image_paths=[input_image_path, output_image_path]
                        )
                    except Exception as e:
                        print(f"图片处理失败: {e}")
                        optimized_cot = f"图片处理失败: {str(e)}"
                        output_image_path = None
                else:
                    print("无法从响应中提取JSON参数")
                    optimized_cot = "无法生成有效的编辑参数"  # 兜底逻辑
                    validated_params = {}
            else:
                optimized_cot = "IEA无响应"
                validated_params = {}

            # 翻译优化后的思维链
            cot_translation = translate_cot(optimized_cot)
            refined_cot = optimized_cot

        else:
            # 使用外部大模型处理复杂编辑
            edited_image_path, external_reasoning = call_external_image_edit(
                input_image_path,
                clear_instruction
            )

            output_filename = f"external_edited_{uuid.uuid4().hex}.png"
            output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

            if edited_image_path != input_image_path and os.path.exists(edited_image_path):
                import shutil
                shutil.copy2(edited_image_path, output_image_path)
            else:
                output_image_path = None

            validated_params = {"external_model": model_to_use, "instruction": clear_instruction}
            refined_cot = external_reasoning
            model = "External_Model"
            usage = {}

        # 翻译思维链
        cot_translation = translate_cot(refined_cot)

        # 生成会话ID并保存
        session_id = str(uuid.uuid4())
        session_data = {
            'original_image_path': input_image_path,
            'origin_instruction': clear_instruction,
            'origin_instruction_original': message,
            'operation_history': [validated_params],  # 保存解析后的参数
            'current_image_path': output_image_path,
            'round': 1,
            'model_used': model_to_use
        }

        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f'session_{session_id}.json')
        with open(session_file, 'w') as f:
            json.dump(session_data, f)

        return jsonify({
            'response': refined_cot ,  # 优化后的思维链
            'response_translated': cot_translation['translated'],
            'user_message_original': message,
            'user_message_translated': instruction_translation['translated'],
            'clear_instruction': clear_instruction,
            'clear_instruction_translated': clear_instruction_translation['translated'],
            'operations': validated_params,  # 返回解析后的参数用于前端展示
            'image': f"/tmp/{os.path.basename(output_image_path)}" if output_image_path else None,
            'model': model,
            'usage': usage,
            'session_id': session_id,
            'model_used': model_to_use
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Processing failed', 'detail': str(e)}), 500


@app.route('/api/edit_image', methods=['POST'])
def edit_image():
    try:
        # 从请求中获取会话ID和新指令
        session_id = request.form.get('session_id', '').strip()
        refine_instruction = request.form.get('preference', '').strip()

        print(f"[DEBUG] edit_image called with session_id: {session_id}, preference: {refine_instruction}")

        if not session_id:
            return jsonify({'error': 'session_id is required'}), 400
        if not refine_instruction:
            return jsonify({'error': 'preference (refine instruction) is required'}), 400

        # 1. 任务分析 - 调用外部大模型分析用户需求
        analysis_result = task_analysis(refine_instruction, [])

        # 获取优化后的清晰指令
        clear_instruction = analysis_result['clear_instruction']
        model_to_use = analysis_result['model_to_use']

        # 翻译用户指令
        instruction_translation = translate_instruction(refine_instruction)
        clear_instruction_translation = translate_instruction(clear_instruction)

        # 加载会话数据
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f'session_{session_id}.json')
        if not os.path.exists(session_file):
            return jsonify({'error': 'Session not found (invalid session_id)'}), 404
        with open(session_file, 'r') as f:
            session_data = json.load(f)

        print(f"[DEBUG] Session data: {session_data}")

        # 从会话中提取关键信息
        original_image_path = session_data['original_image_path']
        origin_instruction = session_data['origin_instruction']
        last_operation = session_data['operation_history'][-1]
        current_round = session_data['round'] + 1

        # 获取上一轮编辑结果图片路径
        previous_image_path = session_data['current_image_path']

        # 创建拼接图像
        concatenated_image = concatenate_images(original_image_path, previous_image_path)
        concatenated_filename = f"concatenated_{uuid.uuid4().hex}.png"
        concatenated_path = os.path.join(app.config['UPLOAD_FOLDER'], concatenated_filename)
        cv2.imwrite(concatenated_path, concatenated_image)

        processing_instruction = clear_instruction_translation['translated'] if clear_instruction_translation[
            'is_chinese'] else clear_instruction_translation['original']

        print(
            f"[DEBUG] Calling refine with: origin_instruction={origin_instruction}, origin_params={last_operation}, refine_instruction={processing_instruction}")
        extracted_json, extracted_cot, model, usage = predict_operations_gier_refine_eval(
            merged_image_path=concatenated_path,
            origin_instruction=origin_instruction,
            origin_params=last_operation,
            refine_instruction=processing_instruction,
            log_file=None
        )

        print(f"[DEBUG] Refine returned: extracted_json={extracted_json}, extracted_cot={extracted_cot}")

        # 翻译思维链
        cot_translation = translate_cot(extracted_cot)

        # 清理临时拼接图
        if os.path.exists(concatenated_path):
            os.remove(concatenated_path)

        # 基于原始图片和新参数生成编辑结果
        output_image_path = None
        if extracted_json:
            # 解析并验证JSON参数
            validated_params = parse_and_validate_parameters(extracted_json)

            editor = ImageEditor(original_image_path)
            output_filename = f"edited_{uuid.uuid4().hex}.png"
            output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

            try:
                editor.process_image(validated_params, output_image_path)

                # 反思总结 - 只优化思维链部分
                refined_cot = reflection_summary(
                    original_instruction=clear_instruction,
                    reasoning_chain=extracted_cot,
                    tool_operations=str(validated_params),
                    image_paths=[original_image_path, output_image_path]
                )
            except Exception as e:
                print(f"图片处理失败: {e}")
                refined_cot = f"图片处理失败: {str(e)}"
                output_image_path = None
        else:
            refined_cot = "无法生成有效的编辑参数"
            validated_params = {}

        # 更新会话数据
        session_data['operation_history'].append(validated_params)
        session_data['current_image_path'] = output_image_path
        session_data['round'] = current_round
        with open(session_file, 'w') as f:
            json.dump(session_data, f)

        # 处理响应文本
        response_text = refined_cot if refined_cot else cot_translation['original']
        if not response_text.strip():
            response_text = "According to your instructions, we made the following modifications to the parameters:"
        else:
            response_text = f"According to your instructions, we have made the following modifications to the parameters:\n\n{response_text}"

        return jsonify({
            'response': response_text.replace('```json', '').replace('```', '').strip(),
            'response_translated': cot_translation['translated'],
            'user_message_original': instruction_translation['original'],
            'user_message_translated': instruction_translation['translated'],
            'clear_instruction': clear_instruction,
            'clear_instruction_translated': clear_instruction_translation['translated'],
            'operations': validated_params,
            'image': f"/tmp/{os.path.basename(output_image_path)}" if output_image_path else None,
            'model': model,
            'usage': usage,
            'round': current_round
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# 比较图片并生成偏好总结
@app.route('/api/compare_images', methods=['POST'])
def compare_images():
    try:
        before_file = request.files['before_image']
        after_file = request.files['after_image']

        if before_file.filename == '' or after_file.filename == '':
            return jsonify({'error': 'Both images must be selected'}), 400

        before_filename = f"before_{uuid.uuid4().hex}.{secure_filename(before_file.filename).split('.')[-1]}"
        after_filename = f"after_{uuid.uuid4().hex}.{secure_filename(after_file.filename).split('.')[-1]}"

        before_path = os.path.join(app.config['UPLOAD_FOLDER'], before_filename)
        after_path = os.path.join(app.config['UPLOAD_FOLDER'], after_filename)

        before_file.save(before_path)
        after_file.save(after_path)

        concatenated_image = concatenate_images(before_path, after_path)
        concatenated_filename = f"concatenated_{uuid.uuid4().hex}.png"
        concatenated_path = os.path.join(app.config['UPLOAD_FOLDER'], concatenated_filename)
        cv2.imwrite(concatenated_path, concatenated_image)

        preference, extracted_cot, model, usage = predict_operations_gier_summary_eval(
            compare_image_path=concatenated_path,
            log_file=None
        )

        # 翻译偏好总结和推理过程
        preference_translation = translate_text(preference, 'zh') if not contains_chinese(preference) else preference
        cot_translation = translate_cot(extracted_cot)

        os.remove(before_path)
        os.remove(after_path)
        os.remove(concatenated_path)

        return jsonify({
            'preference': preference,
            'preference_translated': preference_translation,
            'reasoning': cot_translation['original'],
            'reasoning_translated': cot_translation['translated'],
            'model': model,
            'usage': usage
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'Image comparison failed', 'detail': str(e)}), 500


def parse_and_validate_parameters(extracted_json):
    """
    解析和验证IEA返回的JSON参数，使用config.py中的配置
    """
    validated_params = {}

    if not extracted_json or not isinstance(extracted_json, dict):
        return validated_params

    # 使用config.py中定义的参数范围
    for param, value in extracted_json.items():
        # 参数名称规范化（处理可能的同义词）
        normalized_param = param.lower().strip()

        # 检查参数是否在数值操作符中
        if normalized_param in config.NUMERICAL_OPERATORS:
            min_val, max_val = config.NUMERICAL_OPERATORS[normalized_param]

            # 验证数值范围
            try:
                numeric_value = float(value)
                if min_val <= numeric_value <= max_val:
                    validated_params[normalized_param] = int(numeric_value)
                else:
                    print(f"参数 {param} 的值 {value} 超出范围 [{min_val}, {max_val}]")
            except (ValueError, TypeError):
                print(f"参数 {param} 的值 {value} 不是有效数值")

        # 检查参数是否在非数值操作符中
        elif normalized_param in config.NON_NUMERICAL_OPERATORS:
            validated_params[normalized_param] = value

        # 检查参数名称是否需要替换
        elif normalized_param in config.REPLACE_TABLE:
            normalized_param = config.REPLACE_TABLE[normalized_param]
            if normalized_param in config.NUMERICAL_OPERATORS:
                min_val, max_val = config.NUMERICAL_OPERATORS[normalized_param]
                try:
                    numeric_value = float(value)
                    if min_val <= numeric_value <= max_val:
                        validated_params[normalized_param] = int(numeric_value)
                    else:
                        print(f"参数 {param} 的值 {value} 超出范围 [{min_val}, {max_val}]")
                except (ValueError, TypeError):
                    print(f"参数 {param} 的值 {value} 不是有效数值")
        else:
            print(f"忽略无效参数: {param}")

    return validated_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start Image Edit Agent server')
    parser.add_argument('--port', type=int, default=12345, help='Server port number')
    args = parser.parse_args()
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host='0.0.0.0', port=args.port)
