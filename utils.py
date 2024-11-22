import base64
import io
import uuid

import requests

def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def get_either(values):
    for v in values:
        if v:
            return v
        

def get_file_from_any(base64_encoded_or_url: str):
    if base64_encoded_or_url.startswith("http"):
        data = requests.get(base64_encoded_or_url).content
    else:
        if ";base64," in base64_encoded_or_url:
            base64_encoded_or_url = base64_encoded_or_url.split(";base64,")[-1]
        data = base64.b64decode(base64_encoded_or_url)
    wav_file_like = io.BytesIO(data)
    return wav_file_like