from ollama_wrapper import OllamaClient
from ollama_wrapper.models import (
    GenerateRequest, ChatRequest, 
    CreateModelRequest, ModelOptions,
    Message
)
from typing import Union, Generator, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_generate():
    """Test the generate endpoint of Ollama API"""
    try:
        # Initialize client
        client = OllamaClient()

        # Create generate request with options
        request = GenerateRequest(
            model="llama2",
            prompt="What is the meaning of life?",
            options=ModelOptions(
                stream=False,
                temperature=0.7
            )
        )

        # Make request
        response = client.generate(request)

        # Handle both streaming and non-streaming responses
        if isinstance(response, Generator):
            responses = list(response)  # Convert generator to list
            if responses:
                final_response = responses[-1]  # Get the last response
                print_generate_response(final_response)
        else:
            print_generate_response(response)

        return True

    except Exception as e:
        print_error("Generate test", e)
        return False

def test_chat():
    """Test the chat endpoint of Ollama API"""
    try:
        client = OllamaClient()

        request = ChatRequest(
            model="llama2",
            messages=[
                Message(role="user", content="Hello, how are you?")
            ],
            stream=False
        )

        response = client.chat(request)

        # Handle both streaming and non-streaming responses
        if isinstance(response, Generator):
            responses = list(response)
            if responses:
                final_response = responses[-1]
                print_chat_response(final_response)
        else:
            print_chat_response(response)

        return True

    except Exception as e:
        print_error("Chat test", e)
        return False

def print_generate_response(response: Any) -> None:
    """Helper function to print generate response"""
    print("\nGenerate Test Results:")
    print("-" * 50)
    print(f"Model: {response.model}")
    print(f"Response: {response.response}")
    print(f"Created at: {response.created_at}")
    print(f"Total duration: {response.total_duration}ns")
    print("-" * 50)

def print_chat_response(response: Any) -> None:
    """Helper function to print chat response"""
    print("\nChat Test Results:")
    print("-" * 50)
    print(f"Model: {response.model}")
    if hasattr(response, 'message'):
        print(f"Message: {response.message.content}")
    print(f"Created at: {response.created_at}")
    print("-" * 50)

def print_error(test_name: str, error: Exception) -> None:
    """Helper function to print error information"""
    print(f"\nError in {test_name}:")
    print("-" * 50)
    print(f"Error type: {type(error).__name__}")
    print(f"Error message: {str(error)}")
    print("-" * 50)

if __name__ == "__main__":
    # Run all tests
    tests = [test_generate, test_chat]
    results = []

    for test in tests:
        print(f"\nRunning {test.__name__}...")
        result = test()
        results.append(result)

    # Print summary
    print("\nTest Summary:")
    print("-" * 50)
    for test, result in zip(tests, results):
        status = "PASSED" if result else "FAILED"
        print(f"{test.__name__}: {status}")
    print("-" * 50)