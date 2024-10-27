import sys
import os
from colorama import init, Fore, Style
import logging
from io import StringIO
from Self_Improving_Search import EnhancedSelfImprovingSearch
from llm_config import get_llm_config
from llm_response_parser import UltimateLLMResponseParser
from llm_wrapper import LLMWrapper

# Initialize colorama for cross-platform color support
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

def initialize_llm():
    try:
        print(Fore.YELLOW + "Initializing LLM..." + Style.RESET_ALL)
        with OutputRedirector() as output:
            llm_wrapper = LLMWrapper()
        initialization_output = output.getvalue()
        logger.info(f"LLM Initialization Output:\n{initialization_output}")
        print(Fore.GREEN + "LLM initialized successfully." + Style.RESET_ALL)
        return llm_wrapper
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}", exc_info=True)
        print(Fore.RED + f"Error initializing LLM. Check the log file for details." + Style.RESET_ALL)
        return None

def get_llm_response(llm, prompt):
    try:
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        llm_config = get_llm_config()
        generate_kwargs = {
            'max_tokens': llm_config.get('max_tokens', 1024),
            'stop': llm_config.get('stop', None),
            'temperature': llm_config.get('temperature', 0.7),
            'top_p': llm_config.get('top_p', 1.0),
            'top_k': llm_config.get('top_k', 0),
            'repeat_penalty': llm_config.get('repeat_penalty', 1.0),
        }
        with OutputRedirector() as output:
            response_text = llm.generate(full_prompt, **generate_kwargs)
        llm_output = output.getvalue()
        logger.info(f"LLM Output in get_llm_response:\n{llm_output}")
        return response_text
    except Exception as e:
        logger.error(f"Error getting LLM response: {str(e)}", exc_info=True)
        return f"Sorry, I encountered an error while processing your request. Please check the log file for details."

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
    print_header()
    llm = None

    while True:
        user_input = get_multiline_input()
        if user_input.lower().strip() == 'quit':
            break

        if llm is None:
            print(Fore.YELLOW + "Initializing LLM for the first time..." + Style.RESET_ALL)
            llm = initialize_llm()
            if llm is None:
                print(Fore.RED + "Failed to initialize LLM. Exiting." + Style.RESET_ALL)
                return

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
