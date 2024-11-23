### Problem

```
If vllm model cannot be import in subprocess (runtime error) try re-install numpy==1.26.4
```

### Model specific problem
```
# Qwen audio
current implement version of qwen-audio is not work yet. there are need for custom-vllm version
```


### Step to start on runpod
```
git clone this repo
git submodule init
git submodule update
cd thirdparty/vllm
export MAX_JOBS=18
pip install -e .
```