import sys
import os
from colorama import init, Fore, Style
import logging
from io import StringIO
from Self_Improving_Search import EnhancedSelfImprovingSearch
from llm_config import get_llm_config, MODEL_PATH, LLM_CONFIG_GEMINI
from llm_response_parser import UltimateLLMResponseParser
from llm_wrapper import LLMWrapper
import requests

# Set console encoding to UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Initialize colorama for cross-platform color support
if os.name == 'nt':  # Windows-specific initialization
    init(convert=True, strip=False, wrap=True)
    os.system('chcp 65001 >nul 2>&1')  # Set Windows console to UTF-8 mode, suppress output
else:
    init()

# Set up logging
log_directory = 'logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = os.path.join(log_directory, 'web_llm.log')
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.handlers = []
logger.addHandler(file_handler)
logger.propagate = False

# Disable all other loggers to prevent console output
for name in logging.root.manager.loggerDict:
    if name != __name__:
        logging.getLogger(name).disabled = True

# Suppress root logger
root_logger = logging.getLogger()
root_logger.handlers = []
root_logger.propagate = False
root_logger.setLevel(logging.WARNING)

# Initialize the UltimateLLMResponseParser
parser = UltimateLLMResponseParser()

SYSTEM_PROMPT = """You are an AI assistant capable of web searching and providing informative responses.
When a user's query starts with '/', interpret it as a request to search the web and formulate an appropriate search query.

ALWAYS follow the prompts provided throughout the searching process EXACTLY as indicated.

NEVER assume new instructions for anywhere other than directly when prompted directly. DO NOT SELF PROMPT OR PROVIDE MULTIPLE ANSWERS OR ATTEMPT MULTIPLE RESPONSES FOR ONE PROMPT!
"""

class OutputRedirector:
    def __init__(self, stream=None):
        self.stream = stream or StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def __enter__(self):
        sys.stdout = self.stream
        sys.stderr = self.stream
        return self.stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

def print_header():
    print(Fore.CYAN + Style.BRIGHT + """
    ╔══════════════════════════════════════════════════════════╗
    ║             🌐 Web-LLM Assistant Llama-cpp 🤖            ║
    ╚══════════════════════════════════════════════════════════╝
    """ + Style.RESET_ALL)
    print(Fore.YELLOW + """
    Welcome to the Web-LLM Assistant!

    - For normal interaction, simply type your message and press CTRL+D (Linux/Mac) or CTRL+Z (Windows) to submit.
    - To request a web search, start your message with '/'.
      Example: "/latest news on AI advancements"

    The AI will process your input, perform a search if requested,
    and provide an informed response.

    Press CTRL+D (Linux/Mac) or CTRL+Z (Windows) to submit your input, and type 'quit' to exit.
    """ + Style.RESET_ALL)

def get_multiline_input():
    submit_key = "CTRL+Z" if os.name == 'nt' else "CTRL+D"
    print(Fore.GREEN + f"📝 Enter your message (Press {submit_key} to submit):" + Style.RESET_ALL)
    lines = []
    while True:
        try:
            line = input()
            lines.append(line)
        except EOFError:  # This catches both Ctrl+D on Unix and Ctrl+Z on Windows
            break
        except KeyboardInterrupt:
            print("\nInput cancelled")
            return ""
    return "\n".join(lines)

def print_thinking():
    print(Fore.MAGENTA + "🧠 Thinking..." + Style.RESET_ALL)

def check_llm_availability():
    config = get_llm_config()  # No llm_type for availability check
    status = {
        'llama_cpp': {'available': False, 'message': ''},
        'ollama': {'available': False, 'message': ''},
        'gemini': {'available': False, 'message': ''}
    }
    
    # Check Llama.cpp
    model_path = config.get('model_path')
    if model_path and os.path.exists(model_path) and model_path != "/filepath/to/your/llama.cpp/model":
        status['llama_cpp']['available'] = True
        status['llama_cpp']['message'] = '(Local model)'
    else:
        status['llama_cpp']['message'] = '(Model not found - add model_path in llm_config.py)'
    
    # Check Ollama
    try:
        import requests
        response = requests.get('http://localhost:11434/api/version')
        if response.status_code == 200:
            status['ollama']['available'] = True
            status['ollama']['message'] = '(Local server)'
        else:
            status['ollama']['message'] = '(Ollama server not running - start ollama service)'
    except:
        status['ollama']['message'] = '(Ollama server not running - start ollama service)'
    
    # Check Gemini
    api_key = config.get('api_key')
    if api_key and api_key != "your-api-key-here":
        status['gemini']['available'] = True
        status['gemini']['message'] = '(Cloud API)'
    else:
        status['gemini']['message'] = '(API key not found - add api_key in llm_config.py)'
    
    return status

def choose_llm_type():
    status = check_llm_availability()
    
    print(f"\n{Fore.CYAN}Choose your LLM backend:{Style.RESET_ALL}")
    
    # Llama.cpp
    color = Fore.GREEN if status['llama_cpp']['available'] else Fore.RED
    print(f"{color}1. Llama.cpp {status['llama_cpp']['message']}{Style.RESET_ALL}")
    
    # Ollama
    color = Fore.GREEN if status['ollama']['available'] else Fore.RED
    print(f"{color}2. Ollama {status['ollama']['message']}{Style.RESET_ALL}")
    
    # Gemini
    color = Fore.GREEN if status['gemini']['available'] else Fore.RED
    print(f"{color}3. Gemini {status['gemini']['message']}{Style.RESET_ALL}")
    
    while True:
        try:
            choice = input(f"\n{Fore.GREEN}Enter your choice (1-3): {Style.RESET_ALL}").strip()
            if choice.lower() == 'q':
                raise KeyboardInterrupt
            choice = int(choice)
            if 1 <= choice <= 3:
                llm_type = ["llama_cpp", "ollama", "gemini"][choice - 1]
                if not status[llm_type]['available']:
                    print(f"{Fore.RED}This LLM backend is not properly configured. Please resolve the setup issue and try again.{Style.RESET_ALL}")
                    continue
                return llm_type
            else:
                print(f"{Fore.RED}Please enter a number between 1 and 3.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")

def initialize_llm(llm_type):
    try:
        print(Fore.YELLOW + "\nInitializing LLM..." + Style.RESET_ALL)
        config = get_llm_config(llm_type=llm_type)
        llm_wrapper = LLMWrapper(llm_type=llm_type, config=config)
        if llm_wrapper is None:
            print(Fore.RED + "Failed to initialize LLM. Exiting." + Style.RESET_ALL)
            sys.exit(1)
        return llm_wrapper
    except Exception as e:
        print(Fore.RED + f"Error initializing LLM: {str(e)}" + Style.RESET_ALL)
        sys.exit(1)

def get_llm_response(llm, prompt):
    try:
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        response = llm.generate(full_prompt)
        logger.info(f"LLM Output:\n{response}")
        return parser.parse_llm_response(response)
    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}", exc_info=True)
        print(Fore.RED + f"Error getting response from LLM. Check the log file for details." + Style.RESET_ALL)
        return None

def print_assistant_response(response):
    print(Fore.GREEN + "\n🤖 Assistant:" + Style.RESET_ALL)
    print(response)

def print_footer():
    submit_key = "CTRL+Z" if os.name == 'nt' else "CTRL+D"
    print(Fore.CYAN + Style.BRIGHT + f"""
    ╔══════════════════════════════════════════════════════════╗
    ║  Type 'quit' to exit | {submit_key} to submit                  ║
    ╚══════════════════════════════════════════════════════════╝
    """ + Style.RESET_ALL)

def main():
    # Choose LLM type
    llm_type = choose_llm_type()

    # Initialize LLM
    llm = initialize_llm(llm_type)
    if llm is None:
        print(Fore.RED + "Failed to initialize LLM. Exiting." + Style.RESET_ALL)
        return

    # Show welcome message after LLM is initialized
    print_header()

    while True:
        user_input = get_multiline_input()
        if user_input.lower().strip() == 'quit':
            break

        if user_input.startswith('/'):
            search_query = user_input[1:].strip()
            print(Fore.CYAN + "Initiating web search..." + Style.RESET_ALL)
            search = EnhancedSelfImprovingSearch(llm=llm, parser=parser)
            try:
                answer = search.search_and_improve(search_query)
                print_assistant_response(answer)
            except Exception as e:
                logger.error(f"Error during web search: {str(e)}", exc_info=True)
                print_assistant_response(f"I encountered an error while performing the web search. Please check the log file for details.")
        else:
            print_thinking()
            llm_response = get_llm_response(llm, user_input)
            print_assistant_response(llm_response)

        print_footer()

if __name__ == "__main__":
    main()
