# llm_config.py

from colorama import Fore, Style
import os

# Model path for llama.cpp
MODEL_PATH = "/filepath/to/your/llama.cpp/model" # Replace with your llama.cpp models filepath

# LLM settings for llama_cpp
LLM_CONFIG_LLAMA_CPP = {
    "llm_type": "llama_cpp",
    "model_path": MODEL_PATH,
    "n_ctx": 20000,  # context size
    "n_gpu_layers": 0,  # number of layers to offload to GPU (-1 for all, 0 for none)
    "n_threads": 8,  # number of threads to use
    "temperature": 0.7,  # temperature for sampling
    "top_p": 0.9,  # top p for sampling
    "top_k": 40,  # top k for sampling
    "repeat_penalty": 1.1,  # repeat penalty
    "max_tokens": 1024,  # max tokens to generate
    "stop": ["User:", "\n\n"]  # stop sequences
}

# LLM settings for Ollama
LLM_CONFIG_OLLAMA = {
    "llm_type": "ollama",
    "base_url": "http://localhost:11434",  # default Ollama server URL
    "model_name": "",  # Replace with your Ollama model name
    "temperature": 0.7,
    "top_p": 0.9,
    "n_ctx": 20000,  # context size
    "stop": ["User:", "\n\n"]
}

# LLM settings for Gemini
LLM_CONFIG_GEMINI = {
    "llm_type": "gemini",  # Keep this as "gemini" for the LLM type
    "model_name": "gemini-1.5-pro",  # Just the model name, prefix will be added during initialization
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 8192,  # Note: actual limits will be determined by the selected model
    "api_key": "",  # Add your Gemini API key here
    "stop": ["User:", "\n\n"]
}

def get_config_values():
    """Get all config values without requiring LLM type."""
    return {
        'model_path': MODEL_PATH,
        'gemini_api_key': LLM_CONFIG_GEMINI["api_key"]
    }

def get_llm_config(llm_type=None):
    """Get LLM configuration.
    If llm_type is None, returns basic config for availability checks.
    If llm_type is specified, returns full config for that LLM type."""
    
    if llm_type is None:
        return {
            'model_path': MODEL_PATH,
            'api_key': LLM_CONFIG_GEMINI["api_key"]
        }
    
    if llm_type == 'llama_cpp':
        if not MODEL_PATH or MODEL_PATH == "/filepath/to/your/llama.cpp/model":
            print(f"{Fore.RED}Error: Please set your MODEL_PATH in llm_config.py first{Style.RESET_ALL}")
            raise ValueError("Please set your MODEL_PATH in llm_config.py first")
        return LLM_CONFIG_LLAMA_CPP
    elif llm_type == 'ollama':
        return LLM_CONFIG_OLLAMA
    elif llm_type == 'gemini':
        if not LLM_CONFIG_GEMINI["api_key"]:
            print(f"{Fore.RED}Error: Please set your Gemini API key in llm_config.py first{Style.RESET_ALL}")
            raise ValueError("Please set your Gemini API key in llm_config.py first")
        return LLM_CONFIG_GEMINI
    else:
        raise ValueError("Invalid LLM type")
