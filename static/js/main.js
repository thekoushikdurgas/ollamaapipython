// Utility functions for spinner
function showSpinner() {
    document.querySelector('.spinner-container').classList.add('visible');
}

function hideSpinner() {
    document.querySelector('.spinner-container').classList.remove('visible');
}

// Server status check
async function checkServerStatus() {
    const statusDiv = document.getElementById('serverStatus');
    showSpinner();// Show spinner before request
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
        showError('serverStatus', error);
    } finally {
        hideSpinner();// Hide spinner after response
    }
}

// Check server status on page load and every 30 seconds
document.addEventListener('DOMContentLoaded', () => {
    checkServerStatus();
    setInterval(checkServerStatus, 30000);
});

function showError(elementId, error) {
    hideSpinner();  // Hide spinner on error
    const element = document.getElementById(elementId);
    element.innerHTML = `<div class="error-message">
        <strong>Error:</strong> ${error}
        ${error.includes('Failed to connect to Ollama server') ?
            '<br><br><em>Tip: Make sure Ollama is installed and running locally. Visit <a href="https://ollama.ai/download" target="_blank">https://ollama.ai/download</a> for installation instructions.</em>'
            : ''}
    </div>`;
}

function showResponse(elementId, response) {
    hideSpinner();  // Hide spinner after response
    const element = document.getElementById(elementId);
    element.innerHTML = `<div class="success-message">
        ${JSON.stringify(response, null, 2)}
    </div>`;
}

async function handleStreamingResponse(response, responseArea, contentExtractor) {
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
                const content = contentExtractor(data);
                if (content) {
                    responseArea.innerHTML += content;
                }
            } catch (e) {
                console.error('Error parsing streaming response:', e);
            }
        }
    }
}

// Add this function for model file handling
function addModelFile() {
    const filesDiv = document.getElementById('modelFiles');
    const fileInput = document.createElement('div');
    fileInput.className = 'col-md-6 mt-2';
    fileInput.innerHTML = `
        <input type="file" class="form-control" name="modelFile" accept=".gguf,.safetensors,application/json">
    `;
    filesDiv.appendChild(fileInput);
}

// Generate completion
document.getElementById('generateForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    showSpinner();  // Show spinner before request
    const form = e.target;
    const responseArea = document.getElementById('generateResponse');

    try {
        // Build request data with all supported options
        const data = {
            model: form.model.value,
            prompt: form.prompt.value,
            stream: form.stream.value === 'true',
            raw: form.raw.value === 'true'
        };

        // Add optional fields if provided
        if (form.system.value) {
            data.system = form.system.value;
        }
        if (form.template.value) {
            data.template = form.template.value;
        }
        if (form.format.value) {
            if (form.format.value === 'schema') {
                // Example schema for structured output
                data.format = {
                    type: "object",
                    properties: {
                        response: {
                            type: "string",
                            description: "The generated response"
                        }
                    }
                };
            } else {
                data.format = form.format.value; // 'json'
            }
        }

        // Handle model options
        const options = {};
        const numericFields = ['temperature', 'top_p', 'top_k', 'seed', 'num_predict'];
        numericFields.forEach(field => {
            const value = form[field].value;
            if (value !== '') {
                options[field] = parseFloat(value);
            }
        });

        if (form.keep_alive.value) {
            options.keep_alive = form.keep_alive.value;
        }

        if (Object.keys(options).length > 0) {
            data.options = options;
        }

        // Handle images for multimodal models
        const imageFiles = form.images.files;
        if (imageFiles.length > 0) {
            const images = [];
            for (const file of imageFiles) {
                const base64 = await readFileAsBase64(file);
                images.push(base64);
            }
            data.images = images;
        }

        // Make the API request
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to generate completion');
        }

        if (data.stream) {
            hideSpinner();  // Hide spinner before streaming starts
            await handleStreamingResponse(response, responseArea, data => {
                // For structured outputs (JSON/schema), format the response
                if (data.format === 'json' || data.format?.type === 'object') {
                    try {
                        return JSON.stringify(JSON.parse(data.response), null, 2) + '\n';
                    } catch {
                        return data.response;
                    }
                }
                return data.response;
            });
        } else {
            const result = await response.json();
            showResponse('generateResponse', result);
        }
    } catch (error) {
        showError('generateResponse', error.message);
    }
});

// Utility function to read file as base64
function readFileAsBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
            // Extract the base64 data from the data URL
            const base64 = reader.result.split(',')[1];
            resolve(base64);
        };
        reader.onerror = error => reject(error);
    });
}

// Chat
function addMessage() {
    const messagesDiv = document.getElementById('messages');
    const messageInput = document.createElement('div');
    messageInput.className = 'message-input mb-2';
    messageInput.innerHTML = `
        <select class="form-control mb-2" name="role">
            <option value="user">User</option>
            <option value="assistant">Assistant</option>
            <option value="system">System</option>
            <option value="tool">Tool</option>
        </select>
        <textarea class="form-control" name="content" rows="2" required></textarea>
        <div class="mt-2">
            <label class="form-label">Images (Optional, for multimodal models)</label>
            <input type="file" class="form-control" name="images" multiple accept="image/*">
        </div>
    `;
    messagesDiv.appendChild(messageInput);
}

document.getElementById('chatForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    showSpinner();  // Show spinner before request
    const form = e.target;
    const responseArea = document.getElementById('chatResponse');

    try {
        const messageInputs = form.querySelectorAll('.message-input');
        const messages = [];

        // Process each message with its images
        for (const input of messageInputs) {
            const message = {
                role: input.querySelector('[name="role"]').value,
                content: input.querySelector('[name="content"]').value
            };

            // Handle images for multimodal models
            const imageFiles = input.querySelector('[name="images"]').files;
            if (imageFiles.length > 0) {
                const images = [];
                for (const file of imageFiles) {
                    const base64 = await readFileAsBase64(file);
                    images.push(base64);
                }
                message.images = images;
            }

            messages.push(message);
        }

        // Build request data
        const data = {
            model: form.model.value,
            messages: messages,
            stream: form.stream.value === 'true'
        };

        // Add format if specified
        if (form.format.value) {
            if (form.format.value === 'schema') {
                data.format = {
                    type: "object",
                    properties: {
                        response: {
                            type: "string",
                            description: "The chat response"
                        }
                    }
                };
            } else {
                data.format = form.format.value;
            }
        }

        // Add model options if provided
        const options = {};
        const numericFields = ['temperature', 'top_p', 'top_k', 'seed'];
        numericFields.forEach(field => {
            const value = form[field].value;
            if (value !== '') {
                options[field] = parseFloat(value);
            }
        });

        if (form.keep_alive.value) {
            options.keep_alive = form.keep_alive.value;
        }

        if (Object.keys(options).length > 0) {
            data.options = options;
        }

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
            hideSpinner();  // Hide spinner before streaming starts
            await handleStreamingResponse(response, responseArea, data => {
                // For structured outputs (JSON/schema), format the response
                if (data.message?.content) {
                    if (data.format === 'json' || data.format?.type === 'object') {
                        try {
                            return JSON.stringify(JSON.parse(data.message.content), null, 2) + '\n';
                        } catch {
                            return data.message.content;
                        }
                    }
                    return data.message.content;
                }
                return '';
            });
        } else {
            const result = await response.json();
            showResponse('chatResponse', result);
        }
    } catch (error) {
        showError('chatResponse', error.message);
    }
});

// Model listing and information
async function listModels() {
    showSpinner();  // Show spinner before request
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
    showSpinner();  // Show spinner before request
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
    showSpinner();  // Show spinner before request
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

// Create Model
document.getElementById('createModelForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    showSpinner();  // Show spinner before request
    const form = e.target;

    try {
        const data = {
            model: form.model.value,
            stream: true
        };

        if (form.from_model.value) {
            data.from = form.from_model.value;
        }
        if (form.system.value) {
            data.system = form.system.value;
        }
        if (form.template.value) {
            data.template = form.template.value;
        }
        if (form.quantize.value) {
            data.quantize = form.quantize.value;
        }

        // Handle model files
        const fileInputs = form.querySelectorAll('[name="modelFile"]');
        if (fileInputs.length > 0) {
            const files = {};
            for (const input of fileInputs) {
                if (input.files.length > 0) {
                    const file = input.files[0];
                    const arrayBuffer = await file.arrayBuffer();
                    const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
                    const hashArray = Array.from(new Uint8Array(hashBuffer));
                    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');

                    // Upload blob first
                    const blobResponse = await fetch(`/api/blobs/sha256:${hashHex}`, {
                        method: 'POST',
                        body: file
                    });

                    if (blobResponse.ok) {
                        files[file.name] = `sha256:${hashHex}`;
                    }
                }
            }
            if (Object.keys(files).length > 0) {
                data.files = files;
            }
        }

        const response = await fetch('/api/models', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to create model');
        }

        await handleStreamingResponse(response,
            document.getElementById('createModelResponse'),
            data => data.status ? `${data.status}\n` : ''
        );
    } catch (error) {
        showError('createModelResponse', error.message);
    }
});

// Model Operations
function updateModelOperationsForm() {
    const action = document.querySelector('#modelOperationsForm [name="action"]').value;
    const destinationField = document.getElementById('destinationModelField');
    const showOptionsField = document.getElementById('showOptionsField');
    const streamField = document.querySelector('#modelOperationsForm [name="stream"]').parentElement;

    if (action === 'copy') {
        destinationField.classList.remove('d-none');
        destinationField.querySelector('input').required = true;
        showOptionsField.classList.add('d-none');
        streamField.classList.add('d-none');
    } else if (action === 'show') {
        destinationField.classList.add('d-none');
        destinationField.querySelector('input').required = false;
        showOptionsField.classList.remove('d-none');
        streamField.classList.add('d-none');
    } else if (action === 'delete') {
        destinationField.classList.add('d-none');
        destinationField.querySelector('input').required = false;
        showOptionsField.classList.add('d-none');
        streamField.classList.add('d-none');
    } else {
        destinationField.classList.add('d-none');
        destinationField.querySelector('input').required = false;
        showOptionsField.classList.add('d-none');
        streamField.classList.remove('d-none');
    }
}

document.getElementById('modelOperationsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    showSpinner();  // Show spinner before request
    const form = e.target;
    const action = form.action.value;
    const source = form.source.value;
    const destination = form.destination?.value;
    const stream = form.stream.value === 'true';
    const verbose = form.verbose?.checked;

    try {
        let url, method, data;

        switch (action) {
            case 'show':
                url = `/api/show/${encodeURIComponent(source)}`;
                method = 'GET';
                if (verbose) {
                    url += '?verbose=true';
                }
                break;
            case 'delete':
                url = `/api/models/${encodeURIComponent(source)}`;
                method = 'DELETE';
                break;
            case 'copy':
                url = '/api/models/copy';
                method = 'POST';
                data = { source, destination };
                break;
            case 'pull':
                url = '/api/models/pull';
                method = 'POST';
                data = { name: source, stream };
                break;
            case 'push':
                url = '/api/models/push';
                method = 'POST';
                data = { name: source, stream };
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

        if (stream && (action === 'pull' || action === 'push')) {
            await handleStreamingResponse(response,
                document.getElementById('modelOperationsResponse'),
                data => data.status ? `${data.status}\n` : ''
            );
        } else {
            const result = await response.json();
            showResponse('modelOperationsResponse', result);
        }
    } catch (error) {
        showError('modelOperationsResponse', error.message);
    }
});

// Embeddings
document.getElementById('embeddingsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    showSpinner();  // Show spinner before request
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