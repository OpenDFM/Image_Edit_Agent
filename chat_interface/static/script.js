document.addEventListener('DOMContentLoaded', function () {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const compareBtn = document.getElementById('compare-btn');
    const imageUpload = document.getElementById('image-upload');
    const compareUpload = document.getElementById('compare-upload');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const uploadSuccess = document.getElementById('upload-success');

    const compareModal = document.getElementById('compare-modal');
    const closeModal = document.querySelector('.close-modal');
    const beforeUpload = document.getElementById('before-image-upload');
    const afterUpload = document.getElementById('after-image-upload');
    const beforePreview = document.getElementById('before-preview');
    const afterPreview = document.getElementById('after-preview');
    const confirmCompare = document.getElementById('confirm-compare');
    let beforeFile = null;
    let afterFile = null;

    let uploadedFile = null;
    let currentSessionId = null;

    sendBtn.addEventListener('click', sendMessage);

    userInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    if (uploadBtn) {
        uploadBtn.addEventListener('click', () => imageUpload.click());
    }

    // 比较按钮事件
    if (compareBtn) {
        compareBtn.addEventListener('click', () => compareUpload.click());
    }

    if (imageUpload) {
        imageUpload.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                uploadedFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreviewContainer.innerHTML = `
                        <div class="image-preview">
                            <img src="${e.target.result}" alt="Image preview" class="thumbnail">
                            <button id="remove-image-btn">&times;</button>
                        </div>
                    `;
                    uploadSuccess.style.display = 'inline-block';

                    document.getElementById('remove-image-btn').addEventListener('click', () => {
                        uploadedFile = null;
                        imagePreviewContainer.innerHTML = '';
                        imageUpload.value = '';
                        uploadSuccess.style.display = 'none';
                    });

                    document.querySelector('.image-preview img').addEventListener('click', function() {
                        showEnlargedImage(this.src);
                    });
                };
                reader.readAsDataURL(file);
            }
        });
    }

    // 比较图片上传处理
    // 打开模态框
    compareBtn.addEventListener('click', () => {
        compareModal.style.display = 'block';
    });

    // 关闭模态框
    closeModal.addEventListener('click', () => {
        compareModal.style.display = 'none';
    });

    // 点击模态框外部关闭
    window.addEventListener('click', (e) => {
        if (e.target === compareModal) {
            compareModal.style.display = 'none';
        }
    });

    // 预览P之前的图片
    beforeUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            beforeFile = file;
            const reader = new FileReader();
            reader.onload = (event) => {
                beforePreview.innerHTML = `<img src="${event.target.result}" alt="Before preview">`;
            };
            reader.readAsDataURL(file);
        }
    });

    // 预览P之后的图片
    afterUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            afterFile = file;
            const reader = new FileReader();
            reader.onload = (event) => {
                afterPreview.innerHTML = `<img src="${event.target.result}" alt="After preview">`;
            };
            reader.readAsDataURL(file);
        }
    });

    // 确认对比
    confirmCompare.addEventListener('click', () => {
        if (!beforeFile || !afterFile) {
            alert('Please upload historical edited images');
            return;
        }

        // 关闭模态框并执行对比
        compareModal.style.display = 'none';
        compareImages(beforeFile, afterFile);

        // 重置上传状态
        beforeFile = null;
        afterFile = null;
        beforePreview.innerHTML = '';
        afterPreview.innerHTML = '';
        beforeUpload.value = '';
        afterUpload.value = '';
    });


    function detectLanguage(text) {
        const chineseRegex = /[\u4e00-\u9fff]/;
        return chineseRegex.test(text) ? 'zh' : 'en';
    }

    function renderTranslatedContent(originalText, translatedText, isUserMessage = false) {
        if (!translatedText || originalText === translatedText) {
            return `<p>${originalText}</p>`;
        }

        const originalLang = detectLanguage(originalText);
        const translatedLang = originalLang === 'zh' ? 'en' : 'zh';

        const originalLabel = originalLang === 'zh' ? '中文' : 'English';
        const translatedLabel = translatedLang === 'zh' ? '中文' : 'English';

        return `
            <div class="translation-container">
                <div class="original-text">
                    <span class="language-label">${originalLabel}:</span>
                    <p>${originalText}</p>
                </div>
                <div class="translated-text">
                    <span class="language-label">${translatedLabel}:</span>
                    <p>${translatedText}</p>
                </div>
            </div>
        `;
    }

    // 新增比较图片函数
    function compareImages(beforeImage, afterImage) {
        const loadingElement = document.createElement('div');
        loadingElement.className = 'message model-message';
        loadingElement.innerHTML = '<div class="message-header"><i class="fas fa-robot"></i> Analyzing historical edited images... 正在分析修改历史... </div>';
        loadingElement.id = 'loading-comparison';
        chatMessages.appendChild(loadingElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        const formData = new FormData();
        formData.append('before_image', beforeImage);
        formData.append('after_image', afterImage);

        fetch('/api/compare_images', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Image comparison failed.');
            }
            return response.json();
        })
        .then(data => {
            const loadingMessage = document.getElementById('loading-comparison');
            if (loadingMessage) loadingMessage.remove();

            // 显示比较结果
            addComparisonResultToChat(beforeImage, afterImage, data);

            // 将偏好总结设置为输入框内容
            userInput.value = data.preference;

            // 自动发送编辑请求（如果已有会话）
            if (currentSessionId) {
                sendMessage();
            }
        })
        .catch(error => {
            console.error('Comparison error:', error);
            const loadingMessage = document.getElementById('loading-comparison');
            if (loadingMessage) {
                loadingMessage.innerHTML = `<div class="message-header"><i class="fas fa-robot"></i> Error</div><p>${error.message}</p>`;
            }
        });
    }

    // 显示比较结果
    function addComparisonResultToChat(beforeImage, afterImage, preferenceData) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';

        const headerElement = document.createElement('div');
        headerElement.className = 'message-header';
        headerElement.innerHTML = '<i class="fas fa-exchange-alt"></i> Image Comparison';
        messageElement.appendChild(headerElement);

        const imagesContainer = document.createElement('div');
        imagesContainer.className = 'comparison-images-container';

        // 创建before图片预览
        const beforeContainer = document.createElement('div');
        beforeContainer.className = 'comparison-image-item';
        beforeContainer.innerHTML = '<div class="comparison-label">Before</div>';

        const beforeImg = document.createElement('img');
        beforeImg.className = 'message-image thumbnail';
        beforeImg.alt = 'Before image';

        const beforeReader = new FileReader();
        beforeReader.onload = (e) => {
            beforeImg.src = e.target.result;
            beforeImg.addEventListener('click', () => showEnlargedImage(e.target.result));
        };
        beforeReader.readAsDataURL(beforeImage);

        beforeContainer.appendChild(beforeImg);
        imagesContainer.appendChild(beforeContainer);

        // 创建after图片预览
        const afterContainer = document.createElement('div');
        afterContainer.className = 'comparison-image-item';
        afterContainer.innerHTML = '<div class="comparison-label">After</div>';

        const afterImg = document.createElement('img');
        afterImg.className = 'message-image thumbnail';
        afterImg.alt = 'After image';

        const afterReader = new FileReader();
        afterReader.onload = (e) => {
            afterImg.src = e.target.result;
            afterImg.addEventListener('click', () => showEnlargedImage(e.target.result));
        };
        afterReader.readAsDataURL(afterImage);

        afterContainer.appendChild(afterImg);
        imagesContainer.appendChild(afterContainer);

        messageElement.appendChild(imagesContainer);

        // 显示偏好总结（支持翻译）
        const textElement = document.createElement('div');
        textElement.className = 'response-text preference-summary';

        let preferenceHtml = `<p><strong>Preference Summary:</strong> ${preferenceData && preferenceData.preference ? preferenceData.preference : 'No preference data available'}</p>`;
        if (preferenceData.preference_translated && preferenceData.preference !== preferenceData.preference_translated) {
            preferenceHtml += `<p><strong>偏好总结:</strong> ${preferenceData.preference_translated}</p>`;
        }

        textElement.innerHTML = preferenceHtml;
        messageElement.appendChild(textElement);

        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function sendMessage() {
        message = userInput.value.trim();
        const hasImage = uploadedFile !== null;

        if (!message) {
            if (hasImage) {
                message = "Make it look nicer."; // 默认指令
                userInput.value = message
            } else {
                alert('Please enter a message or upload an image');
                return;
            }
        }

        let userImagePreviewUrl = null;
        if (uploadedFile) {
            userImagePreviewUrl = URL.createObjectURL(uploadedFile);
        }

        // 先添加用户消息（只显示原始内容）
        addMessageToChat('user', message, userImagePreviewUrl);
        userInput.value = '';

        const loadingElement = document.createElement('div');
        loadingElement.className = 'message model-message';
        loadingElement.innerHTML = '<div class="message-header"><i class="fas fa-robot"></i> IEA is thinking... IEA 正在分析...</div>';
        loadingElement.id = 'loading-message';
        chatMessages.appendChild(loadingElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        const formData = new FormData();
        formData.append('message', message || '');

        const isEditRequest = !!currentSessionId;
        const apiUrl = isEditRequest ? '/api/edit_image' : '/api/send_message';

        if (isEditRequest) {
            formData.append('session_id', currentSessionId);
            formData.append('preference', message || '');
        } else if (uploadedFile) {
            formData.append('image', uploadedFile);
        }

        fetch(apiUrl, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                if (response.status === 400) {
                    throw new Error('请求参数错误，请检查输入内容或图片格式');
                }
                throw new Error(`请求失败: ${response.status}`);
            }
            return response.text().then(text => {
                try {
                    return JSON.parse(text);
                } catch (e) {
                    throw new Error('后端返回格式错误，无法解析响应');
                }
            });
        })
        .then(data => {
            const loadingMessage = document.getElementById('loading-message');
            if (loadingMessage) loadingMessage.remove();

            // 更新最后一条用户消息以包含翻译
            const userMessages = document.querySelectorAll('.user-message');
            const lastUserMessage = userMessages[userMessages.length - 1];
            if (lastUserMessage && data.user_message_original) {
                const textElement = lastUserMessage.querySelector('.response-text');
                if (textElement) {
                    textElement.innerHTML = renderTranslatedContent(
                        data.user_message_original,
                        data.user_message_translated,
                        true
                    );
                }
            }

            addMessageToChat('model', data);

            if (data.session_id) {
                currentSessionId = data.session_id;
            }

            if (uploadedFile && !isEditRequest) {
                uploadedFile = null;
                imagePreviewContainer.innerHTML = '';
                imageUpload.value = '';
                uploadSuccess.style.display = 'none';
            }

            if (userImagePreviewUrl) {
                URL.revokeObjectURL(userImagePreviewUrl);
            }
        })
        .catch(error => {
            console.error('请求错误:', error);
            const loadingMessage = document.getElementById('loading-message');
            if (loadingMessage) {
                loadingMessage.innerHTML = `<div class="message-header"><i class="fas fa-robot"></i> 错误</div><p>${error.message}</p>`;
            }
            if (userImagePreviewUrl) URL.revokeObjectURL(userImagePreviewUrl);
        });
    }

    function renderMarkdown(text) {
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    function addMessageToChat(sender, data, imageUrl = null) {
        const messageElement = document.createElement('div');
        messageElement.className = sender === 'user' ? 'message user-message' : 'message model-message';

        const headerElement = document.createElement('div');
        headerElement.className = 'message-header';
        headerElement.innerHTML = sender === 'user'
            ? '<i class="fas fa-user"></i> User'
            : '<i class="fas fa-robot"></i> IEA';
        messageElement.appendChild(headerElement);

        if (sender === 'user') {
            if (data && data.trim() !== '') {
                const textElement = document.createElement('div');
                textElement.className = 'response-text';

                let messageHtml = '';

                // 显示原始输入和翻译
                if (typeof data === 'object' && data.user_message_original) {
                    messageHtml += `<div class="original-instruction">
                        <strong>原始输入:</strong> ${data.user_message_original}
                    </div>`;

                    if (data.user_message_translated && data.user_message_original !== data.user_message_translated) {
                        messageHtml += `<div class="translated-instruction">
                            <strong>翻译:</strong> ${data.user_message_translated}
                        </div>`;
                    }

                    // 显示优化后的指令
                    if (data.clear_instruction) {
                        messageHtml += `<div class="optimized-instruction">
                            <strong>优化指令:</strong> ${data.clear_instruction}
                        </div>`;

                        if (data.clear_instruction_translated && data.clear_instruction !== data.clear_instruction_translated) {
                            messageHtml += `<div class="optimized-translated">
                                <strong>优化指令翻译:</strong> ${data.clear_instruction_translated}
                            </div>`;
                        }
                    }
                } else {
                    messageHtml = `<p>${data}</p>`;
                }

                textElement.innerHTML = messageHtml;
                messageElement.appendChild(textElement);
            }
            }
            if (imageUrl) {
                const imgContainer = document.createElement('div');
                imgContainer.className = 'image-container';

                const imgElement = document.createElement('img');
                imgElement.src = imageUrl;
                imgElement.className = 'message-image thumbnail';
                imgElement.alt = 'Uploaded image';
                imgElement.addEventListener('click', function () {
                    showEnlargedImage(this.src);
                });

                imgContainer.appendChild(imgElement);
                messageElement.appendChild(imgContainer);
            }
        else {
            if (data.response) {
                const responseElement = document.createElement('div');
                responseElement.className = 'response-text';

                // 显示翻译后的内容
                responseElement.innerHTML = renderTranslatedContent(
                    data.response,
                    data.response_translated
                );
                messageElement.appendChild(responseElement);
            }

            if (data.operations) {
                const operationsElement = document.createElement('div');
                operationsElement.className = 'operations-info';
                operationsElement.innerHTML = `
                    <h4><i class="fas fa-tools"></i> Tool Calls:</h4>
                    <pre>${JSON.stringify(data.operations, null, 2)}</pre>
                `;
                messageElement.appendChild(operationsElement);
            }

            if (data.model) {
                const modelElement = document.createElement('div');
                modelElement.className = 'model-info';
                modelElement.innerHTML = `<i class="fas fa-microchip"></i> <span>Model: ${data.model}</span>`;
                messageElement.appendChild(modelElement);
            }

            if (data.image) {
                const imgContainer = document.createElement('div');
                imgContainer.className = 'image-container';

                const imgElement = document.createElement('img');
                imgElement.src = data.image;
                imgElement.className = 'message-image thumbnail';
                imgElement.alt = 'Generated image';
                imgElement.addEventListener('click', function() {
                    showEnlargedImage(this.src);
                });

                imgContainer.appendChild(imgElement);
                messageElement.appendChild(imgContainer);
            }
        }

        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }


    function showEnlargedImage(src) {
        const modalOverlay = document.createElement('div');
        modalOverlay.className = 'modal-overlay';
        modalOverlay.innerHTML = `
            <div class="modal-content">
                <span class="close-modal">&times;</span>
                <img src="${src}" alt="Enlarged image">
            </div>
        `;

        document.body.appendChild(modalOverlay);

        modalOverlay.addEventListener('click', function(e) {
            if (e.target === modalOverlay || e.target.classList.contains('close-modal')) {
                document.body.removeChild(modalOverlay);
            }
        });
    }
});

