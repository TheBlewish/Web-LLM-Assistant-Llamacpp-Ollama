# llm_config.py

MODEL_PATH = "/home/james/llama.cpp/models/Phi-3-medium-128k-instruct-Q6_K.gguf"

# LLM settings
LLM_CONFIG = {
    "model_path": MODEL_PATH,
    "n_ctx": 20000,  # context size
    "n_gpu_layers": 0,  # number of layers to offload to GPU (-1 for all, 0 for none)
    "n_threads": 8,  # number of threads to use
    "temp": 0.7,  # temperature for sampling
    "top_p": 0.9,  # top p for sampling
    "top_k": 40,  # top k for sampling
    "repeat_penalty": 1.1,  # repeat penalty
    "max_tokens": 1024,  # max tokens to generate
    "stop": ["User:", "\n\n"]  # stop sequences
}

def get_llm_config():
    return LLM_CONFIG
