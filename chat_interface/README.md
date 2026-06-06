# AI对话与图像生成界面

这是一个基于Web的客户端界面，类似于ChatGPT的网站页面。右侧是用户输入区域，左侧是模型的输出和基于输出生成的图像。

## 功能特点

- 类似ChatGPT的用户界面
- 支持文本对话和图像生成
- 对话历史记录保存
- 响应式设计
- 简洁明了的用户体验

## 安装与使用

1. 安装依赖项：

```bash
pip install -r requirements.txt
```

2. 启动服务器：

```bash
python deploy.py --port 12345
```

3. 在浏览器中访问：`http://localhost:12345`

## 集成您自己的模型

在`deploy.py`文件中，找到`send_message`函数，并将示例实现替换为您自己的模型API调用代码：

```python
@app.route('/api/send_message', methods=['POST'])
def send_message():
    data = request.json
    user_message = data.get('message', '')
    
    # 替换为您的模型API调用
    model_response = your_model_api.generate_response(user_message)
    
    # 替换为您的图像生成API调用
    image_url = your_image_generator.generate_image(model_response)
    
    # 将对话添加到历史记录
    conversation_history.append({
        'user': user_message,
        'model': model_response,
        'image': image_url
    })
    
    return jsonify({
        'response': model_response,
        'image': image_url
    })
```

## 文件结构

- `deploy.py` - 主应用程序文件
- `templates/index.html` - HTML模板
- `static/style.css` - CSS样式
- `static/script.js` - JavaScript交互逻辑
- `requirements.txt` - 依赖项列表

