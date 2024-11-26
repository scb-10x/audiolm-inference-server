import os
import time
from typing import AsyncGenerator
from fastapi.responses import StreamingResponse
from utils import get_either, get_file_from_any
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from entity.entity import (
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    DeltaMessage,
    UsageInfo,
)
from handler.base_handler import BaseHandler
from transformers import AutoProcessor
from transformers import pipeline
import librosa

class ASRPipelineHandler(BaseHandler):

    def __init__(self, model_name: str):
        super().__init__(model_name)
        engine_args = AsyncEngineArgs(
            model=model_name,
            max_model_len=2048,
            gpu_memory_utilization=float(os.environ.get("GPU_MEMORY_UTILIZATION", 0.4)),
            enforce_eager=True,
        )
        llm = AsyncLLMEngine.from_engine_args(engine_args)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.llm = llm
        asr_pipe = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            chunk_length_s=30,
            device="cuda",
        )
        self.asr_pipe = asr_pipe

    def generate_stream(self, request):
        conversations = []
        for message in request.messages:
            if isinstance(message["content"], list):
                cur_audio_text = None
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audio, sr = librosa.load(get_file_from_any(ele["audio_url"]))
                        cur_audio_text = self.asr_pipe(
                            audio,
                            generate_kwargs={"task": "transcribe"},
                            return_timestamps=False,
                        )["text"]
                if cur_audio_text is not None:
                    text = None
                    for ele in message["content"]:
                        if ele["type"] == "text":
                            if '<audio>' in ele['text']:
                                text = ele["text"].replace("<audio>",  f'"{cur_audio_text}"')
                            else:
                                text = ele['text'] + cur_audio_text
                    if text is not None:
                        conversations.append({'role': message['role'], 'content': text})
            else:
                conversations.append({'role': message['role'], 'content': message['content']})
        engine_prompt = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        results_generator = self.llm.generate(
            engine_prompt,
            SamplingParams(
                top_p=request.top_p,
                temperature=request.temperature,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                max_tokens=get_either(
                    [request.max_completion_tokens, request.max_tokens]
                ),
            ),
            str(time.monotonic()),
        )

        # Streaming case
        async def stream_results() -> AsyncGenerator[bytes, None]:
            prev_output = ""
            async for request_output in results_generator:
                output = request_output.outputs[-1]
                delta_out = output.text[len(prev_output) :]
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(content=delta_out),
                    logprobs=None,
                    finish_reason=output.finish_reason,
                )

                chunk = ChatCompletionStreamResponse(
                    id=request.request_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    choices=[choice_data],
                    model=self.model_name,
                )
                chunk.usage = UsageInfo(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                )
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"
                prev_output = output.text
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=""),
                logprobs=None,
                finish_reason=output.finish_reason,
            )
            final_usage_chunk = ChatCompletionStreamResponse(
                id=request.request_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                choices=[choice_data],
                model=self.model_name,
                usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )
            final_usage_data = final_usage_chunk.model_dump_json(
                exclude_unset=True, exclude_none=True
            )
            yield f"data: {final_usage_data}\n\n"

        return StreamingResponse(stream_results())

    def generate(self, request):
        raise NotImplementedError
