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
    
    def get_base_model_name(self):
        if '/' in self.model_name:
            return self.model_name.split('/')[-1]
        return self.model_name
        