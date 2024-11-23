### Problem

```
If vllm model cannot be import in subprocess (runtime error) try re-install numpy==1.26.4
```

### Model specific problem
```
# Qwen audio
current implement version of qwen-audio is not work yet. there are need for custom-vllm version
```

### How to run
```
MODEL_NAME="WillHeld/DiVA-llama-3-v0-8b" uvicorn api_server:app --port 40021
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct" GPU_MEMORY_UTILIZATION=0.5 uvicorn api_server:app --port 40020
```