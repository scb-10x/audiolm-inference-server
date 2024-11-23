from fastapi import Response
from entity.entity import ChatCompletionRequest
from abc import ABC, abstractmethod

class BaseHandler(ABC):
    
    def __init__(self, model_name):
        self.model_name = model_name
        pass
    
    @abstractmethod
    def generate_stream(self, request: ChatCompletionRequest) -> Response:
        raise NotImplementedError()
    
    @abstractmethod
    def generate(self, request: ChatCompletionRequest) -> Response:
        raise NotImplementedError()
        