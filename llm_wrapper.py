from llama_cpp import Llama
import requests
import json
from llm_config import get_llm_config

class LLMWrapper:
    def __init__(self):
        self.llm_config = get_llm_config()
        self.llm_type = self.llm_config.get('llm_type', 'llama_cpp')
        if self.llm_type == 'llama_cpp':
            self.llm = self._initialize_llama_cpp()
        elif self.llm_type == 'ollama':
            self.base_url = self.llm_config.get('base_url', 'http://localhost:11434')
            self.model_name = self.llm_config.get('model_name', 'your_model_name')
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")

    def _initialize_llama_cpp(self):
        return Llama(
            model_path=self.llm_config.get('model_path'),
            n_ctx=self.llm_config.get('n_ctx', 2048),
            n_gpu_layers=self.llm_config.get('n_gpu_layers', 0),
            n_threads=self.llm_config.get('n_threads', 8),
            verbose=False
        )

    def generate(self, prompt, **kwargs):
        if self.llm_type == 'llama_cpp':
            llama_kwargs = self._prepare_llama_kwargs(kwargs)
            response = self.llm(prompt, **llama_kwargs)
            return response['choices'][0]['text'].strip()
        elif self.llm_type == 'ollama':
            return self._ollama_generate(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")

    def _ollama_generate(self, prompt, **kwargs):
        url = f"{self.base_url}/api/generate"
        data = {
            'model': self.model_name,
            'prompt': prompt,
            'options': {
                'temperature': kwargs.get('temperature', self.llm_config.get('temperature', 0.7)),
                'top_p': kwargs.get('top_p', self.llm_config.get('top_p', 0.9)),
                'stop': kwargs.get('stop', self.llm_config.get('stop', [])),
                'num_predict': kwargs.get('max_tokens', self.llm_config.get('max_tokens', 1024)),
            }
        }
        response = requests.post(url, json=data, stream=True)
        if response.status_code != 200:
            raise Exception(f"Ollama API request failed with status {response.status_code}: {response.text}")
        text = ''.join(json.loads(line)['response'] for line in response.iter_lines() if line)
        return text.strip()

    def _prepare_llama_kwargs(self, kwargs):
        llama_kwargs = {
            'max_tokens': kwargs.get('max_tokens', self.llm_config.get('max_tokens', 1024)),
            'temperature': kwargs.get('temperature', self.llm_config.get('temperature', 0.7)),
            'top_p': kwargs.get('top_p', self.llm_config.get('top_p', 0.9)),
            'stop': kwargs.get('stop', self.llm_config.get('stop', [])),
            'echo': False,
        }
        return llama_kwargs
