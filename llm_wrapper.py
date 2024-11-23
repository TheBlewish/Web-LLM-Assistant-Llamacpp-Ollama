from llama_cpp import Llama
import requests
import json
from llm_config import get_llm_config
import google.generativeai as genai
from colorama import Fore, Style
import readchar
import time

class LLMWrapper:
    def __init__(self, llm_type=None, config=None):
        self.llm_type = llm_type
        self.llm_config = config if config else get_llm_config(llm_type)
        
        # Initialize the appropriate LLM based on type
        if self.llm_type == "llama_cpp":
            self._initialize_llama()
        elif self.llm_type == "ollama":
            self._initialize_ollama()
        elif self.llm_type == "gemini":
            self._initialize_gemini()
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")

    def _initialize_llama_cpp(self):
        return Llama(
            model_path=self.llm_config.get('model_path'),
            n_ctx=self.llm_config.get('n_ctx', 8192),
            n_gpu_layers=self.llm_config.get('n_gpu_layers', 0),
            n_threads=self.llm_config.get('n_threads', 8),
            verbose=False
        )

    def _initialize_ollama(self):
        self.base_url = self.llm_config.get('base_url', 'http://localhost:11434')
        
        # Get list of available models
        response = requests.get(f"{self.base_url}/api/tags")
        if response.status_code != 200:
            raise Exception(f"Failed to get Ollama models: {response.text}")
        
        models = response.json().get('models', [])
        if not models:
            raise Exception("No Ollama models found")
        
        # Let user choose model
        print(f"\n{Fore.CYAN}Available Ollama models:{Style.RESET_ALL}")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model.get('name')}")
        
        while True:
            print(f"\n{Fore.GREEN}Choose a model number (1-{len(models)}): {Style.RESET_ALL}")
            try:
                choice = int(input().strip())
                if 1 <= choice <= len(models):
                    self.model_name = models[choice-1].get('name')
                    print(f"{Fore.GREEN}Selected model: {self.model_name}{Style.RESET_ALL}")
                    break
                else:
                    print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")

    def _initialize_gemini(self):
        try:
            # Configure API
            api_key = self.llm_config.get('api_key')
            if not api_key:
                raise ValueError("Gemini API key not found in configuration")

            genai.configure(api_key=api_key)

            # Get list of available models that support text generation
            print(f"\n{Fore.CYAN}Available Gemini models:{Style.RESET_ALL}")
            available_models = []
            model_count = 0
            
            for model in genai.list_models():
                model_count += 1
                if "generateContent" in model.supported_generation_methods:
                    available_models.append(model)
                    print(f"\n{Fore.GREEN}{len(available_models)}. {model.name}{Style.RESET_ALL}")
                    print(f"   Display name: {model.display_name}")
                    print(f"   Description: {model.description}")
                    print(f"   Input token limit: {model.input_token_limit}")
                    print(f"   Output token limit: {model.output_token_limit}")

            if not available_models:
                raise ValueError("No Gemini models found that support text generation")

            print(f"\n{Fore.YELLOW}Found {model_count} total models, {len(available_models)} support text generation{Style.RESET_ALL}")

            # Let user choose a model
            while True:
                try:
                    choice = input(f"\n{Fore.GREEN}Choose a model number (1-{len(available_models)}): {Style.RESET_ALL}").strip()
                    if choice.lower() == 'q':
                        raise KeyboardInterrupt
                    choice = int(choice)
                    if 1 <= choice <= len(available_models):
                        selected_model = available_models[choice-1]
                        model_name = selected_model.name
                        print(f"\n{Fore.GREEN}Selected model: {selected_model.display_name}{Style.RESET_ALL}")
                        
                        # Initialize the model with generation config
                        generation_config = genai.types.GenerationConfig(
                            temperature=self.llm_config.get('temperature', 0.7),
                            top_p=self.llm_config.get('top_p', 0.9),
                            top_k=self.llm_config.get('top_k', 40),
                            max_output_tokens=min(
                                self.llm_config.get('max_tokens', 8192),
                                selected_model.output_token_limit
                            )
                        )

                        self.model = genai.GenerativeModel(
                            model_name=model_name,
                            generation_config=generation_config
                        )

                        print(f"{Fore.GREEN}Successfully initialized {model_name}{Style.RESET_ALL}")
                        print(f"Model capabilities: {selected_model.description}")
                        print(f"Input token limit: {selected_model.input_token_limit}")
                        print(f"Output token limit: {selected_model.output_token_limit}")
                        break
                    else:
                        print(f"{Fore.RED}Invalid choice. Please enter a number between 1 and {len(available_models)}.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}Error initializing Gemini: {str(e)}{Style.RESET_ALL}")
            raise

    def generate(self, prompt, **kwargs):
        if self.llm_type == 'llama_cpp':
            llama_kwargs = self._prepare_llama_kwargs(kwargs)
            response = self.llm(prompt, **llama_kwargs)
            return response['choices'][0]['text'].strip()
        elif self.llm_type == 'ollama':
            return self._ollama_generate(prompt, **kwargs)
        elif self.llm_type == 'gemini':
            return self._gemini_generate(prompt, **kwargs)
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
                'num_predict': kwargs.get('max_tokens', self.llm_config.get('max_tokens', 8192)),
            }
        }
        response = requests.post(url, json=data, stream=True)
        if response.status_code != 200:
            raise Exception(f"Ollama API request failed with status {response.status_code}: {response.text}")
        text = ''.join(json.loads(line)['response'] for line in response.iter_lines() if line)
        return text.strip()

    def _gemini_generate(self, prompt, **kwargs):
        try:
            # Create safety settings for this specific request
            safety_settings = []
            for category, level in self.llm_config.get('safety_settings', {}).items():
                safety_settings.append({
                    "category": getattr(genai.types.HarmCategory, category),
                    "threshold": getattr(genai.types.HarmBlockThreshold, level)
                })

            # Generate response with retry logic for rate limits
            max_retries = 3
            retry_delay = 1  # Start with 1 second delay
            
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(
                        prompt,
                        safety_settings=safety_settings
                    )
                    
                    # Check if the response was blocked
                    if hasattr(response, 'prompt_feedback'):
                        if response.prompt_feedback.block_reason:
                            raise ValueError(f"Response blocked: {response.prompt_feedback.block_reason}")
                    
                    # Get the response text
                    if hasattr(response, 'text'):
                        return response.text.strip()
                    elif hasattr(response, 'parts'):
                        return response.parts[0].text.strip()
                    else:
                        raise ValueError("Unexpected response format from Gemini")
                        
                except Exception as e:
                    if "RATE_LIMIT_EXCEEDED" in str(e) and attempt < max_retries - 1:
                        print(f"{Fore.YELLOW}Rate limit exceeded, retrying in {retry_delay} seconds...{Style.RESET_ALL}")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    raise
                    
        except Exception as e:
            error_msg = str(e)
            if "RATE_LIMIT_EXCEEDED" in error_msg:
                error_msg = "Rate limit exceeded. Please try again in a few seconds."
            elif "SAFETY" in error_msg.upper():
                error_msg = "The response was blocked due to safety concerns."
            print(f"{Fore.RED}Error generating response from Gemini: {error_msg}{Style.RESET_ALL}")
            raise

    def _prepare_llama_kwargs(self, kwargs):
        llama_kwargs = {
            'max_tokens': kwargs.get('max_tokens', self.llm_config.get('max_tokens', 8192)),
            'temperature': kwargs.get('temperature', self.llm_config.get('temperature', 0.7)),
            'top_p': kwargs.get('top_p', self.llm_config.get('top_p', 0.9)),
            'stop': kwargs.get('stop', self.llm_config.get('stop', [])),
            'echo': False,
        }
        return llama_kwargs
