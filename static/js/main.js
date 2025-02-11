// Server status check
async function checkServerStatus() {
    const statusDiv = document.getElementById('serverStatus');
    try {
        const response = await fetch('/api/version');
        if (response.ok) {
            const version = await response.json();
            statusDiv.className = 'alert alert-success mb-4';
            statusDiv.innerHTML = `
                <h4 class="alert-heading">✅ Ollama Server Connected</h4>
                <p>Successfully connected to Ollama server (Version: ${version.version})</p>
            `;
        } else {
            throw new Error('Failed to connect to Ollama server');
        }
    } catch (error) {
        statusDiv.className = 'alert alert-danger mb-4';
        statusDiv.innerHTML = `
            <h4 class="alert-heading">❌ Connection Error</h4>
            <p>Failed to connect to Ollama server. Please make sure:</p>
            <ul>
                <li>Ollama is installed on your system</li>
                <li>The Ollama server is running on port 11434</li>
                <li>No firewall is blocking the connection</li>
            </ul>
            <p><a href="https://ollama.ai/download" target="_blank">Click here for Ollama installation instructions</a></p>
        `;
    }
}

// Check server status on page load and every 30 seconds
document.addEventListener('DOMContentLoaded', () => {
    checkServerStatus();
    setInterval(checkServerStatus, 30000);
});

// Utility functions
function showError(elementId, error) {
    const element = document.getElementById(elementId);
    element.innerHTML = `<div class="error-message">
        <strong>Error:</strong> ${error}
        ${error.includes('Failed to connect to Ollama server') ? 
            '<br><br><em>Tip: Make sure Ollama is installed and running locally. Visit <a href="https://ollama.ai/download" target="_blank">https://ollama.ai/download</a> for installation instructions.</em>' 
            : ''}
    </div>`;
}

function showResponse(elementId, response) {
    const element = document.getElementById(elementId);
    element.innerHTML = `<div class="success-message">
        ${JSON.stringify(response, null, 2)}
    </div>`;
}

// Generate completion
document.getElementById('generateForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const responseArea = document.getElementById('generateResponse');

    try {
        const data = {
            model: form.model.value,
            prompt: form.prompt.value,
            options: {
                stream: form.stream.value === 'true'
            }
        };

        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to generate completion');
        }

        if (data.options.stream) {
            responseArea.innerHTML = ''; // Clear previous content
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n').filter(line => line.trim());

                for (const line of lines) {
                    try {
                        const data = JSON.parse(line);
                        responseArea.innerHTML += data.response;
                    } catch (e) {
                        console.error('Error parsing streaming response:', e);
                    }
                }
            }
        } else {
            const result = await response.json();
            showResponse('generateResponse', result);
        }
    } catch (error) {
        showError('generateResponse', error.message);
    }
});

// Chat completion
function addMessage() {
    const messagesDiv = document.getElementById('messages');
    const messageInput = document.createElement('div');
    messageInput.className = 'message-input mb-2';
    messageInput.innerHTML = `
        <select class="form-control mb-2" name="role">
            <option value="user">User</option>
            <option value="assistant">Assistant</option>
            <option value="system">System</option>
        </select>
        <textarea class="form-control" name="content" rows="2" required></textarea>
    `;
    messagesDiv.appendChild(messageInput);
}

document.getElementById('chatForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const responseArea = document.getElementById('chatResponse');

    try {
        const messageInputs = form.querySelectorAll('.message-input');
        const messages = Array.from(messageInputs).map(input => ({
            role: input.querySelector('[name="role"]').value,
            content: input.querySelector('[name="content"]').value
        }));

        const data = {
            model: form.model.value,
            messages: messages,
            stream: form.stream.value === 'true'
        };

        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to generate chat completion');
        }

        if (data.stream) {
            responseArea.innerHTML = '';
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n').filter(line => line.trim());

                for (const line of lines) {
                    try {
                        const data = JSON.parse(line);
                        if (data.message && data.message.content) {
                            responseArea.innerHTML += data.message.content;
                        }
                    } catch (e) {
                        console.error('Error parsing streaming response:', e);
                    }
                }
            }
        } else {
            const result = await response.json();
            showResponse('chatResponse', result);
        }
    } catch (error) {
        showError('chatResponse', error.message);
    }
});

// Model operations
async function listModels() {
    try {
        const response = await fetch('/api/models');
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to list models');
        }
        const result = await response.json();
        showResponse('modelsResponse', result);
    } catch (error) {
        showError('modelsResponse', error.message);
    }
}

async function listRunningModels() {
    try {
        const response = await fetch('/api/running');
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to list running models');
        }
        const result = await response.json();
        showResponse('modelsResponse', result);
    } catch (error) {
        showError('modelsResponse', error.message);
    }
}

async function getVersion() {
    try {
        const response = await fetch('/api/version');
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to get version');
        }
        const result = await response.json();
        showResponse('modelsResponse', result);
    } catch (error) {
        showError('modelsResponse', error.message);
    }
}

function updateModelForm() {
    const action = document.querySelector('#modelManageForm [name="action"]').value;
    const destinationField = document.getElementById('destinationField');

    if (action === 'copy') {
        destinationField.classList.remove('d-none');
        destinationField.querySelector('input').required = true;
    } else {
        destinationField.classList.add('d-none');
        destinationField.querySelector('input').required = false;
    }
}

document.getElementById('modelManageForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const action = form.action.value;
    const modelName = form.modelName.value;
    const destination = form.destination?.value;

    try {
        let url, method, data;

        switch (action) {
            case 'show':
                url = `/api/show/${modelName}`;
                method = 'GET';
                break;
            case 'pull':
                url = '/api/models/pull';
                method = 'POST';
                data = { name: modelName };
                break;
            case 'push':
                url = '/api/models/push';
                method = 'POST';
                data = { name: modelName };
                break;
            case 'delete':
                url = `/api/models/${modelName}`;
                method = 'DELETE';
                break;
            case 'copy':
                url = '/api/models/copy';
                method = 'POST';
                data = { source: modelName, destination };
                break;
        }

        const response = await fetch(url, {
            method,
            headers: data ? { 'Content-Type': 'application/json' } : undefined,
            body: data ? JSON.stringify(data) : undefined
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `Failed to ${action} model`);
        }

        const result = await response.json();
        showResponse('modelManageResponse', result);
    } catch (error) {
        showError('modelManageResponse', error.message);
    }
});

// Embeddings
document.getElementById('embeddingsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;

    try {
        const data = {
            model: form.model.value,
            prompt: form.prompt.value
        };

        const response = await fetch('/api/embeddings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to generate embeddings');
        }

        const result = await response.json();
        showResponse('embeddingsResponse', result);
    } catch (error) {
        showError('embeddingsResponse', error.message);
    }
});