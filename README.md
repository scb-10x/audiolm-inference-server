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
MODEL_NAME="pretrained/DiVA-llama-3-v0-8b" uvicorn api_server:app --port 40021 --host 0.0.0.0
MODEL_NAME="Qwen/Qwen2-Audio-7B-Instruct" GPU_MEMORY_UTILIZATION=0.5 uvicorn api_server:app --port 40020 --host 0.0.0.0
```

### Step to setup on runpod
```
git clone this repo
git submodule init
git submodule update
cd thirdparty/vllm
export MAX_JOBS=18
pip install -e .
```