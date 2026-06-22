/**
 * E-commerce Multi-Agent System — Frontend Logic
 *
 * Handles chat interaction, SSE streaming, state visualization,
 * and session management.
 */

(function () {
    'use strict';

    // ---- DOM Elements ----
    const chatMessages = document.getElementById('chatMessages');
    const chatInput = document.getElementById('chatInput');
    const btnSend = document.getElementById('btnSend');
    const btnNewSession = document.getElementById('btnNewSession');
    const btnClearSession = document.getElementById('btnClearSession');
    const stateIndicator = document.getElementById('stateIndicator');
    const stateDot = stateIndicator.querySelector('.state-dot');
    const stateLabel = stateIndicator.querySelector('.state-label');
    const intentValue = document.getElementById('intentValue');
    const sessionValue = document.getElementById('sessionValue');
    const traceContainer = document.getElementById('traceContainer');

    // ---- State ----
    let sessionId = generateId();
    let isProcessing = false;

    // ---- Init ----
    sessionValue.textContent = sessionId;
    updateStateDot('idle');
    clearWelcomeOnFirstMessage();

    // ---- Event Listeners ----
    btnSend.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    btnNewSession.addEventListener('click', newSession);
    btnClearSession.addEventListener('click', clearSession);

    // ---- Send Message ----
    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text || isProcessing) return;

        // Add user message to chat
        appendMessage('user', text);
        chatInput.value = '';
        chatInput.focus();

        // Show typing indicator
        const typingEl = showTyping();
        isProcessing = true;
        btnSend.disabled = true;

        // Clear trace for new run
        traceContainer.innerHTML = '';

        try {
            // Use SSE streaming for real-time updates
            await streamChat(text);
        } catch (err) {
            console.error('Chat error:', err);
            appendMessage('assistant', '抱歉，处理您的请求时出现了错误。请稍后重试。');
            updateStateDot('idle');
        } finally {
            removeTyping(typingEl);
            isProcessing = false;
            btnSend.disabled = false;
            chatInput.focus();
        }
    }

    // ---- SSE Streaming ----
    async function streamChat(message) {
        const response = await fetch('/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: sessionId }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let assistantText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = JSON.parse(line.slice(6));

                switch (data.type) {
                    case 'state':
                        updateStateDot(data.state.toLowerCase());
                        if (data.intent) {
                            intentValue.textContent = data.intent;
                        }
                        if (data.trace) {
                            updateTrace(data.trace);
                        }
                        break;

                    case 'response':
                        assistantText = data.text;
                        updateStateDot(data.current_state.toLowerCase());
                        if (data.intent) {
                            intentValue.textContent = data.intent;
                        }
                        if (data.trace) {
                            updateTrace(data.trace);
                        }
                        break;

                    case 'error':
                        assistantText = '错误: ' + data.message;
                        break;

                    case 'done':
                        // Final message already captured
                        break;
                }
            }
        }

        if (assistantText) {
            appendMessage('assistant', assistantText);
        }
    }

    // ---- Fallback: non-streaming ----
    async function sendChatFallback(message) {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_id: sessionId }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        appendMessage('assistant', data.response);
        updateStateDot(data.current_state.toLowerCase());
        intentValue.textContent = data.intent || '-';
        if (data.agent_trace) {
            updateTrace(data.agent_trace);
        }
    }

    // ---- UI Helpers ----
    function appendMessage(role, text) {
        const el = document.createElement('div');
        el.className = `message ${role}`;
        el.textContent = text;
        chatMessages.appendChild(el);
        scrollToBottom();
    }

    function showTyping() {
        const el = document.createElement('div');
        el.className = 'typing-indicator';
        el.innerHTML = '<span></span><span></span><span></span>';
        chatMessages.appendChild(el);
        scrollToBottom();
        return el;
    }

    function removeTyping(el) {
        if (el && el.parentNode) {
            el.parentNode.removeChild(el);
        }
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function updateStateDot(state) {
        // Remove all state classes
        stateDot.className = 'state-dot';
        if (state) {
            stateDot.classList.add(state);
        }
        // Update label
        const stateNames = {
            idle: 'IDLE',
            inquiry: 'INQUIRY',
            order: 'ORDER',
            aftersales: 'AFTERSALES',
            cross_agent: 'CROSS_AGENT',
        };
        stateLabel.textContent = stateNames[state] || state.toUpperCase();
    }

    function updateTrace(trace) {
        const stateNames = {
            IDLE: { name: 'IDLE', cls: 'idle', desc: '就绪' },
            INQUIRY: { name: 'INQUIRY', cls: 'inquiry', desc: '商品咨询' },
            ORDER: { name: 'ORDER', cls: 'order', desc: '订单处理' },
            AFTERSALES: { name: 'AFTERSALES', cls: 'aftersales', desc: '售后服务' },
            CROSS_AGENT: { name: 'CROSS_AGENT', cls: 'cross_agent', desc: '跨Agent协作' },
        };

        traceContainer.innerHTML = '';

        trace.forEach((state, i) => {
            const info = stateNames[state] || { name: state, cls: 'idle', desc: '' };
            const el = document.createElement('div');
            el.className = 'trace-item';
            el.innerHTML = `
                <span class="trace-step ${info.cls}">${i + 1}</span>
                <span class="trace-name">
                    ${info.name}
                    <small>${info.desc}</small>
                </span>
            `;
            traceContainer.appendChild(el);
        });
    }

    function clearWelcomeOnFirstMessage() {
        // Remove welcome on first send
        const orig = sendMessage;
        // Already handled in sendMessage flow
    }

    // ---- Session Management ----
    function newSession() {
        sessionId = generateId();
        sessionValue.textContent = sessionId;
        chatMessages.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">🤖</div>
                <h3>新会话已创建</h3>
                <p>会话ID: <code>${sessionId}</code></p>
                <p>请开始您的对话...</p>
            </div>
        `;
        traceContainer.innerHTML = '<div class="trace-empty">等待用户输入...</div>';
        updateStateDot('idle');
        intentValue.textContent = '-';
        chatInput.focus();
    }

    async function clearSession() {
        try {
            await fetch(`/session/${sessionId}`, { method: 'DELETE' });
        } catch (e) {
            // Ignore
        }
        newSession();
    }

    function generateId() {
        return 'sess-' + Math.random().toString(36).substring(2, 10);
    }

})();
