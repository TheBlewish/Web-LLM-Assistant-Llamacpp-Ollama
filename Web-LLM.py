import sys
import os
import time
import readchar
from colorama import init, Fore, Style
import requests
from bs4 import BeautifulSoup
import trafilatura
from duckduckgo_search import DDGS
from llama_cpp import Llama
import logging
import json
import urllib.parse
import random
import socket
from Self_Improving_Search import SimpleSelfImprovingSearch, OutputRedirector

# Initialize colorama for cross-platform color support
init()

# Set up logging
logging.basicConfig(filename='web_llm.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = "/home/llama.cpp/models/Phi-3-medium-128k-instruct-Q6_K.gguf"

SYSTEM_PROMPT = """You are an AI assistant capable of web searching and providing informative responses.
When a user's query starts with '/', interpret it as a request to search the web and formulate an appropriate search query.
For regular inputs, respond based on your existing knowledge.
Aim to provide helpful, accurate, and concise responses, incorporating the most recent information from web searches when available.
When given search results, analyze and synthesize the information to provide a comprehensive answer.
Pay attention to the publication dates of search results and prioritize the most recent and relevant information in your responses when appropriate."""

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
]

def print_header():
    print(Fore.CYAN + Style.BRIGHT + """
    ╔══════════════════════════════════════════════════════════╗
    ║                 🌐 LLM Web Assistant 🤖                  ║
    ╚══════════════════════════════════════════════════════════╝
    """ + Style.RESET_ALL)
    print(Fore.YELLOW + """
    Welcome to the LLM Web Assistant!

    - For normal interaction, simply type your message and press CTRL+D to submit.
    - To request a web search, start your message with '/'.
      Example: "/latest news on AI advancements"

    The AI will process your input, perform a search if requested,
    and provide an informed response.

    Press CTRL+D to submit your input, and type 'quit' to exit.
    """ + Style.RESET_ALL)

def get_multiline_input():
    print(Fore.GREEN + "📝 Enter your message (Press CTRL+D to submit):" + Style.RESET_ALL)
    lines = [""]
    while True:
        char = readchar.readchar()
        if char == readchar.key.CTRL_D:
            print()  # Move to a new line after submission
            return "\n".join(lines).strip()
        elif char == readchar.key.ENTER:
            print()  # Move to a new line
            lines.append("")
        elif char == readchar.key.BACKSPACE:
            if lines[-1]:
                lines[-1] = lines[-1][:-1]
                print('\b \b', end='', flush=True)
            elif len(lines) > 1:
                lines.pop()
                print('\033[F\033[K', end='', flush=True)  # Move up and clear line
        else:
            lines[-1] += char
            print(char, end='', flush=True)

def print_thinking():
    print(Fore.MAGENTA + "\n🧠 Thinking", end="")
    for _ in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print("\n")

def check_network_connectivity():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def initialize_llm():
    print(Fore.YELLOW + "Initializing LLM..." + Style.RESET_ALL)
    try:
        with OutputRedirector():
            llm = Llama(model_path=MODEL_PATH, n_ctx=20000)  # Using 20000 for context size as requested
        print(Fore.GREEN + "LLM initialized successfully." + Style.RESET_ALL)
        return llm
    except Exception as e:
        print(Fore.RED + f"Error initializing LLM: {str(e)}" + Style.RESET_ALL)
        return None

def get_llm_response(llm, prompt):
    try:
        with OutputRedirector() as output:
            full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
            response = llm(full_prompt, max_tokens=1024, stop=["User:", "\n\n"])
        return response['choices'][0]['text'].strip()
    except Exception as e:
        return f"Sorry, I encountered an error while processing your request: {str(e)}"

def print_assistant_response(response):
    print(Fore.GREEN + "\n🤖 Assistant:" + Fore.WHITE)
    print("  " + response)

def print_footer():
    print(Fore.CYAN + Style.BRIGHT + """
    ╔══════════════════════════════════════════════════════════╗
    ║  Type 'quit' and press CTRL+D to exit | CTRL+D to submit ║
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

        print_thinking()

        try:
            if user_input.startswith('/'):
                search_query = user_input[1:].strip()
                self_improving_search = SimpleSelfImprovingSearch(llm)
                answer = self_improving_search.search_and_improve(search_query)
                if answer:
                    print_assistant_response(answer)
                else:
                    print(Fore.RED + "Sorry, I couldn't find a satisfactory answer after multiple attempts." + Style.RESET_ALL)
            else:
                llm_response = get_llm_response(llm, user_input)
                print_assistant_response(llm_response)
        except Exception as e:
            print(Fore.RED + f"An error occurred: {str(e)}" + Style.RESET_ALL)
            logging.error(f"Detailed error: {str(e)}", exc_info=True)

        print_footer()

if __name__ == "__main__":
    main()
