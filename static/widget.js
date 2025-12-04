(function() {
    // Inject CSS
    const style = document.createElement('style');
    style.innerHTML = `
        .rag-chat-widget-btn {
            position: fixed;
            bottom: 22px;
            right: 22px;
            width: 72px;
            height: 72px;
            background: linear-gradient(135deg, #1b6bff, #5c9dff);
            border-radius: 20px;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 12px 30px rgba(25, 90, 255, 0.28);
            z-index: 9999;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .rag-chat-widget-btn:hover {
            transform: translateY(-2px) scale(1.04);
            box-shadow: 0 16px 32px rgba(25, 90, 255, 0.34);
        }
        .rag-chat-window {
            position: fixed;
            bottom: 110px;
            right: 22px;
            width: min(420px, 90vw);
            height: min(620px, 80vh);
            background: white;
            border-radius: 18px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.18);
            display: none;
            flex-direction: column;
            z-index: 9999;
            overflow: hidden;
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .rag-chat-header {
            background: linear-gradient(135deg, #1b6bff, #5c9dff);
            color: white;
            padding: 16px 18px;
            font-weight: 700;
            display: flex;
            justify-content: space-between;
            align-items: center;
            letter-spacing: 0.3px;
        }
        .rag-chat-close {
            cursor: pointer;
            font-size: 22px;
            opacity: 0.9;
        }
        .rag-chat-close:hover {
            opacity: 1;
        }
        .rag-chat-messages {
            flex: 1;
            padding: 18px;
            overflow-y: auto;
            background: radial-gradient(circle at top left, #f7fbff 0%, #f0f3f7 45%, #edf1f5 100%);
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .rag-message {
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 16px;
            font-size: 15px;
            line-height: 1.5;
            box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        }
        .rag-message.user {
            align-self: flex-end;
            background: linear-gradient(135deg, #1b6bff, #5c9dff);
            color: white;
            border-bottom-right-radius: 6px;
        }
        .rag-message.bot {
            align-self: flex-start;
            background: #ffffff;
            color: #1f2d3d;
            border: 1px solid #e4e8ef;
            border-bottom-left-radius: 6px;
        }
        .rag-chat-input-area {
            padding: 14px 16px;
            border-top: 1px solid #e4e8ef;
            display: flex;
            gap: 10px;
            background: #f9fbff;
        }
        .rag-chat-input {
            flex: 1;
            padding: 12px 14px;
            border: 1px solid #d6deeb;
            border-radius: 14px;
            outline: none;
            font-size: 14px;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .rag-chat-input:focus {
            border-color: #5c9dff;
            box-shadow: 0 0 0 3px rgba(92, 157, 255, 0.2);
        }
        .rag-chat-send {
            background: linear-gradient(135deg, #1b6bff, #5c9dff);
            color: white;
            border: none;
            padding: 12px 18px;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 700;
            letter-spacing: 0.2px;
            min-width: 88px;
            box-shadow: 0 8px 20px rgba(25, 90, 255, 0.22);
            transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.2s ease;
        }
        .rag-chat-send:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(25, 90, 255, 0.28);
        }
        .rag-chat-send:disabled {
            background: #c5d4f5;
            cursor: not-allowed;
            box-shadow: none;
            transform: none;
        }
        .rag-typing {
            font-size: 12px;
            color: #5d6b82;
            margin-left: 10px;
            display: none;
        }
    `;
    document.head.appendChild(style);

    // Create Widget Elements
    const widgetBtn = document.createElement('div');
    widgetBtn.className = 'rag-chat-widget-btn';
    widgetBtn.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>';
    
    const chatWindow = document.createElement('div');
    chatWindow.className = 'rag-chat-window';
    chatWindow.innerHTML = `
        <div class="rag-chat-header">
            <span>Asistente IA</span>
            <span class="rag-chat-close">&times;</span>
        </div>
        <div class="rag-chat-messages" id="rag-messages">
            <div class="rag-message bot">Hola! Soy tu asistente IA. ¿En qué puedo ayudarte hoy?</div>
        </div>
        <div class="rag-typing" id="rag-typing">Escribiendo...</div>
        <div class="rag-chat-input-area">
            <input type="text" class="rag-chat-input" placeholder="Escribe tu pregunta..." id="rag-input">
            <button class="rag-chat-send" id="rag-send">Enviar</button>
        </div>
    `;

    document.body.appendChild(widgetBtn);
    document.body.appendChild(chatWindow);

    // State
    let isOpen = false;
    let chatHistory = []; // Stores {role: 'user'|'assistant', content: '...'}

    // Elements
    const closeBtn = chatWindow.querySelector('.rag-chat-close');
    const messagesContainer = chatWindow.querySelector('#rag-messages');
    const input = chatWindow.querySelector('#rag-input');
    const sendBtn = chatWindow.querySelector('#rag-send');
    const typingIndicator = chatWindow.querySelector('#rag-typing');

    // Toggle Chat
    function toggleChat() {
        isOpen = !isOpen;
        chatWindow.style.display = isOpen ? 'flex' : 'none';
        widgetBtn.style.display = isOpen ? 'none' : 'flex';
        if (isOpen) input.focus();
    }

    widgetBtn.addEventListener('click', toggleChat);
    closeBtn.addEventListener('click', toggleChat);

    // Add Message to UI
    function addMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `rag-message ${sender}`;
        msgDiv.textContent = text;
        messagesContainer.appendChild(msgDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // Send Message
    async function sendMessage() {
        const text = input.value.trim();
        if (!text) return;

        // UI Update
        addMessage(text, 'user');
        input.value = '';
        input.disabled = true;
        sendBtn.disabled = true;
        typingIndicator.style.display = 'block';

        // Add to history
        // Note: We send the history *before* the current message is added to it in the backend usually, 
        // but here we want to maintain state. The backend expects 'history' as a list of [q, a] tuples or similar,
        // or we can send the full conversation. 
        // Let's adapt to what we will implement in python: a list of messages.
        
        // API URL - configurable via window.RAG_WIDGET_CONFIG.endpoint
        const API_URL = (window.RAG_WIDGET_CONFIG && window.RAG_WIDGET_CONFIG.endpoint) || 'http://localhost:9994/chat';
        const API_KEY = (window.RAG_WIDGET_CONFIG && window.RAG_WIDGET_CONFIG.apiKey) || '';

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    ...(API_KEY ? { 'x-api-key': API_KEY } : {})
                },
                body: JSON.stringify({ 
                    prompt: text,
                    history: chatHistory 
                })
            });

            if (!response.ok) {
                const errText = await response.text();
                addMessage(`Error ${response.status}: ${errText || 'sin respuesta'}`, 'bot');
                return;
            }

            // Evitar fallo si la respuesta no es JSON válido
            let data;
            const contentType = response.headers.get('content-type') || '';
            if (contentType.includes('application/json')) {
                data = await response.json();
            } else {
                const txt = await response.text();
                addMessage(`Respuesta inesperada del servidor: ${txt || 'vacío'}`, 'bot');
                return;
            }
            
            if (data.error) {
                addMessage('Error: ' + data.error, 'bot');
            } else {
                const botResponse = data.choices[0].message.content;
                addMessage(botResponse, 'bot');
                
                // Update history
                chatHistory.push({ role: 'user', content: text });
                chatHistory.push({ role: 'assistant', content: botResponse });
            }

        } catch (err) {
            addMessage('Error de conexión.', 'bot');
            console.error(err);
        } finally {
            input.disabled = false;
            sendBtn.disabled = false;
            typingIndicator.style.display = 'none';
            input.focus();
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

})();
