# Ollama API Wrapper and UI

A Flask-based web interface and Python wrapper for interacting with the Ollama API. This project provides both a user-friendly web UI and a programmatic way to interact with Ollama's language models.

## Features

- Web UI for testing Ollama API endpoints
- Python wrapper for Ollama API integration
- Support for all major Ollama operations:
  - Text generation
  - Chat completions
  - Model management
  - Embeddings generation
- Dark/Light mode toggle
- Response streaming
- Syntax highlighting for JSON responses

## Getting Started

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Run the Flask server:

```bash
python app.py
```

3. Access the web UI at `http://0.0.0.0:5000`

## API Endpoints

- `/api/generate` - Generate text completions
- `/api/chat` - Chat completions
- `/api/models` - List and manage models
- `/api/embeddings` - Generate embeddings
- `/api/version` - Get Ollama version

## Development Mode

The project includes a mock server for development. Enable it by setting:

```python
os.environ['USE_MOCK_OLLAMA'] = 'true'
```

## Project Structure

```
├── ollama_wrapper/     # Python wrapper package
├── static/            # Static assets (CSS, JS)
├── templates/         # HTML templates
├── app.py            # Flask application
└── test_api.py       # API tests
```

## License

MIT License
