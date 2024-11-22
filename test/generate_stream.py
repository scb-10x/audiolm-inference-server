import base64
from openai import OpenAI
import requests

def encode_audio_base64_from_url(url_or_path: str) -> str:
    """Encode an audio retrieved from a remote url to base64 format."""
    if url_or_path.startswith("http"):
        with requests.get(url_or_path) as response:
            response.raise_for_status()
            result = base64.b64encode(response.content).decode("utf-8")
    else:
        with open(url_or_path, "rb") as wav_file:
            result = base64.b64encode(wav_file.read()).decode("utf-8")
    return result

def main(audio_url_or_path: str, model_name: str, stream=True):
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="aabb",
    )
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant. Respond conversationally to the speech provided.",
                    },
                    {
                        "type": "audio",
                        "audio_url": "data:audio/wav;base64,"
                        + encode_audio_base64_from_url(audio_url_or_path),
                    },
                ],
            }
        ],
        model=model_name,
        max_tokens=64,
        stream=stream,
    )
    for output in chat_completion_from_url:
        print(output)
        if len(output.choices) > 0:
            print(output.choices[0].delta.content)


if __name__ == "__main__":
    main(audio_url_or_path="test/tmp-20240801-035414.wav", model_name="qwen-audio")
