from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ModelOptions(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = None
    stream: Optional[bool] = True
    keep_alive: Optional[str] = "5m"

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    format: Optional[Union[str, Dict[str, Any]]] = None
    raw: Optional[bool] = False
    images: Optional[List[str]] = None
    options: Optional[ModelOptions] = None

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = True
    format: Optional[Union[str, Dict[str, Any]]] = None
    options: Optional[ModelOptions] = None
    tools: Optional[List[Dict[str, Any]]] = None

class CreateModelRequest(BaseModel):
    model: str
    from_model: Optional[str] = Field(None, alias="from")
    files: Optional[Dict[str, str]] = None
    adapters: Optional[Dict[str, str]] = None
    template: Optional[str] = None
    system: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    license: Optional[Union[str, List[str]]] = None
    messages: Optional[List[Message]] = None
    stream: Optional[bool] = True
    quantize: Optional[str] = None

class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_duration: Optional[int] = None

class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: Message
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_duration: Optional[int] = None

class ModelResponse(BaseModel):
    status: str
    error: Optional[str] = None

class EmbeddingRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[ModelOptions] = None

class EmbeddingResponse(BaseModel):
    embedding: List[float]
