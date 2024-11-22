
from fastapi import FastAPI
from fastapi.responses import Response
from handler.handler import get_handler
from entity.entity import ChatCompletionRequest
import os
app = FastAPI()

# handler = get_handler('Qwen/Qwen2-Audio-7B-Instruct')
handler = get_handler(os.environ['MODEL_NAME'])

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/v1/chat/completions")
async def generate(request: ChatCompletionRequest) -> Response:
    if request.stream:
        return handler.generate_stream(request)
    return handler.generate(request)