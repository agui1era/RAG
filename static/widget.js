(function() {
    // Inject CSS
    const style = document.createElement('style');
    style.innerHTML = `
        .rag-chat-widget-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            background-color: #007bff;
            border-radius: 50%;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 9999;
            transition: transform 0.3s ease;
        }
        .rag-chat-widget-btn:hover {
            transform: scale(1.1);
        }
        .rag-chat-window {
            position: fixed;
            bottom: 90px;
            right: 20px;
            width: 350px;
            height: 500px;
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
            display: none;
            flex-direction: column;
            z-index: 9999;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        .rag-chat-header {
            background-color: #007bff;
            color: white;
            padding: 15px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .rag-chat-close {
            cursor: pointer;
            font-size: 20px;
        }
        .rag-chat-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            background-color: #f8f9fa;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .rag-message {
            max-width: 80%;
            padding: 10px 14px;
            border-radius: 18px;
            font-size: 14px;
            line-height: 1.4;
        }
        .rag-message.user {
            align-self: flex-end;
            background-color: #007bff;
            color: white;
            border-bottom-right-radius: 4px;
        }
        .rag-message.bot {
            align-self: flex-start;
            background-color: #e9ecef;
            color: #333;
            border-bottom-left-radius: 4px;
        }
        .rag-chat-input-area {
            padding: 15px;
            border-top: 1px solid #eee;
            display: flex;
            gap: 10px;
            background-color: white;
        }
        .rag-chat-input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 20px;
            outline: none;
        }
        .rag-chat-send {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            font-weight: bold;
        }
        .rag-chat-send:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .rag-typing {
            font-size: 12px;
            color: #888;
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
            <span>PDF Assistant</span>
            <span class="rag-chat-close">&times;</span>
        </div>
        <div class="rag-chat-messages" id="rag-messages">
            <div class="rag-message bot">Hola! Soy tu asistente de documentos PDF. ¿En qué puedo ayudarte hoy?</div>
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
        
        // API URL - Change this if hosting the widget on a different domain
        const API_URL = 'http://localhost:9994/chat';

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    prompt: text,
                    history: chatHistory 
                })
            });

            const data = await response.json();
            
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
