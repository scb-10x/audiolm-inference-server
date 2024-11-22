
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from handler.handler import get_handler
from entity.entity import ChatCompletionRequest

app = FastAPI()

# handler = get_handler('Qwen/Qwen2-Audio-7B-Instruct')
handler = get_handler("WillHeld/DiVA-llama-3-v0-8b")

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/v1/chat/completions")
async def generate(request: ChatCompletionRequest) -> Response:
    if request.stream:
        return handler.generate_stream(request)
    return handler.generate(request)