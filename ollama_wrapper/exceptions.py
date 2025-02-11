class OllamaError(Exception):
    """Base exception for Ollama API errors"""
    pass

class OllamaRequestError(OllamaError):
    """Exception raised for errors in the request to Ollama API"""
    def __init__(self, message: str, status_code: int = None):
        self.status_code = status_code
        super().__init__(message)

class OllamaResponseError(OllamaError):
    """Exception raised for errors in the response from Ollama API"""
    def __init__(self, message: str, response_data: dict = None):
        self.response_data = response_data
        super().__init__(message)

class OllamaValidationError(OllamaError):
    """Exception raised for validation errors"""
    pass

class OllamaTimeoutError(OllamaError):
    """Exception raised for timeout errors"""
    pass
