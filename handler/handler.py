from .huggingface_handler import HuggingfaceHandler
from .vllm_handler import VLLMHandler
from .asr_pipeline_handler import ASRPipelineHandler

def get_handler(model_name: str):
    if model_name in ['Qwen/Qwen2-Audio-7B-Instruct']:
        return VLLMHandler(model_name)
    elif 'pipeline/' in model_name:
        llm_model_name = model_name.replace('pipeline/', '')
        return ASRPipelineHandler(llm_model_name)
    return HuggingfaceHandler(model_name)