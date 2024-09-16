import time
import re
import os
from typing import List, Dict, Tuple, Union
from llama_cpp import Llama
from duckduckgo_search import DDGS
from colorama import Fore, Style
import logging
import sys
from io import StringIO
from web_scraper import get_web_content, can_fetch
from llm_config import get_llm_config
from llm_response_parser import UltimateLLMResponseParser
from urllib.parse import urlparse

# Set up logging
log_directory = 'logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Configure logger to write only to file and not to console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = os.path.join(log_directory, 'llama_output.log')
file_handler = logging.FileHandler(log_file)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.handlers = []  # Clear existing handlers
logger.addHandler(file_handler)
logger.propagate = False  # Prevent log messages from propagating to the root logger

# Suppress logging from root logger and other modules
root_logger = logging.getLogger()
root_logger.handlers = []  # Remove all handlers associated with the root logger
root_logger.propagate = False
root_logger.setLevel(logging.WARNING)  # Set root logger level to WARNING

# Suppress logging from duckduckgo_search and related modules
logging.getLogger('duckduckgo_search').setLevel(logging.WARNING)
logging.getLogger('duckduckgo_search').handlers = []
logging.getLogger('duckduckgo_search').propagate = False

logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('requests').handlers = []
logging.getLogger('requests').propagate = False

logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('urllib3').handlers = []
logging.getLogger('urllib3').propagate = False

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

class EnhancedSelfImprovingSearch:
    def __init__(self, llm: Llama, parser: UltimateLLMResponseParser, max_attempts: int = 5):
        self.llm = llm
        self.parser = parser
        self.max_attempts = max_attempts
        self.llm_config = get_llm_config()

    @staticmethod
    def initialize_llm():
        llm_config = get_llm_config()
        llm_config['verbose'] = False  # Suppress verbose output
        with OutputRedirector() as output:
            llm = Llama(**llm_config)
        initialization_output = output.getvalue()
        logger.info(f"LLM Initialization Output:\n{initialization_output}")
        return llm

    def print_thinking(self):
        print(Fore.MAGENTA + "🧠 Thinking..." + Style.RESET_ALL)

    def print_searching(self):
        print(Fore.MAGENTA + "📝 Searching..." + Style.RESET_ALL)

    def search_and_improve(self, user_query: str) -> str:
        attempt = 0
        while attempt < self.max_attempts:
            print(f"\n{Fore.CYAN}Search attempt {attempt + 1}:{Style.RESET_ALL}")
            self.print_searching()

            try:
                formulated_query, time_range = self.formulate_query(user_query, attempt)

                print(f"{Fore.YELLOW}Original query: {user_query}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Formulated query: {formulated_query}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Time range: {time_range}{Style.RESET_ALL}")

                if not formulated_query:
                    print(f"{Fore.RED}Error: Empty search query. Retrying...{Style.RESET_ALL}")
                    attempt += 1
                    continue

                search_results = self.perform_search(formulated_query, time_range)

                if not search_results:
                    print(f"{Fore.RED}No results found. Retrying with a different query...{Style.RESET_ALL}")
                    attempt += 1
                    continue

                self.display_search_results(search_results)

                selected_urls = self.select_relevant_pages(search_results, user_query)

                if not selected_urls:
                    print(f"{Fore.RED}No relevant URLs found. Retrying...{Style.RESET_ALL}")
                    attempt += 1
                    continue

                print(Fore.MAGENTA + "⚙️ Scraping selected pages..." + Style.RESET_ALL)
                scraped_content = self.scrape_content(selected_urls)

                if not scraped_content:
                    print(f"{Fore.RED}Failed to scrape content. Retrying...{Style.RESET_ALL}")
                    attempt += 1
                    continue

                self.display_scraped_content(scraped_content)

                self.print_thinking()

                should_refine, decision = self.synthesize_or_refine(user_query, scraped_content)

                if not should_refine:
                    return self.generate_final_answer(user_query, scraped_content)
                else:
                    print(f"{Fore.YELLOW}Refining search...{Style.RESET_ALL}")
                    attempt += 1

            except Exception as e:
                print(f"{Fore.RED}An error occurred during search attempt. Check the log file for details.{Style.RESET_ALL}")
                logger.error(f"An error occurred during search: {str(e)}", exc_info=True)
                attempt += 1

        return self.synthesize_final_answer(user_query)

    def formulate_query(self, user_query: str, attempt: int) -> Tuple[str, str]:
        user_query_short = user_query[:200]
        prompt = f"""
Based on the following user question, formulate a concise and effective search query:
"{user_query_short}"
Your task:
1. Create a search query of 2-5 words that will yield relevant results.
2. Determine if a specific time range is needed for the search.
Time range options:
- 'd': Limit results to the past day. Use for very recent events or rapidly changing information.
- 'w': Limit results to the past week. Use for recent events or topics with frequent updates.
- 'm': Limit results to the past month. Use for relatively recent information or ongoing events.
- 'y': Limit results to the past year. Use for annual events or information that changes yearly.
- 'none': No time limit. Use for historical information or topics not tied to a specific time frame.
Respond in the following format:
Search query: [Your 2-5 word query]
Time range: [d/w/m/y/none]
Do not provide any additional information or explanation.
"""
        max_retries = 3
        for retry in range(max_retries):
            with OutputRedirector() as output:
                response = self.llm(prompt, max_tokens=50, stop=None)
            llm_output = output.getvalue()
            logger.info(f"LLM Output in formulate_query:\n{llm_output}")
            response_text = response['choices'][0]['text'].strip()
            query, time_range = self.parse_query_response(response_text)
            if query and time_range:
                return query, time_range
        return self.fallback_query(user_query), "none"

    def parse_query_response(self, response: str) -> Tuple[str, str]:
        query = ""
        time_range = "none"
        for line in response.split('\n'):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                if "query" in key:
                    query = self.clean_query(value)
                elif "time" in key or "range" in key:
                    time_range = self.validate_time_range(value)
        return query, time_range

    def clean_query(self, query: str) -> str:
        query = re.sub(r'["\'\[\]]', '', query)  # Remove quotes and brackets
        query = re.sub(r'\s+', ' ', query)   # Replace multiple spaces with single space
        return query.strip()[:100]  # Limit to 100 characters

    def validate_time_range(self, time_range: str) -> str:
        valid_ranges = ['d', 'w', 'm', 'y', 'none']
        time_range = time_range.lower()
        return time_range if time_range in valid_ranges else 'none'

    def fallback_query(self, user_query: str) -> str:
        words = user_query.split()
        return " ".join(words[:5])  # Use the first 5 words of the user query

    def perform_search(self, query: str, time_range: str) -> List[Dict]:
        if not query:
            return []

        with DDGS() as ddgs:
            try:
                with OutputRedirector() as output:
                    if time_range and time_range != 'none':
                        results = list(ddgs.text(query, timelimit=time_range, max_results=10))
                    else:
                        results = list(ddgs.text(query, max_results=10))
                ddg_output = output.getvalue()
                logger.info(f"DDG Output in perform_search:\n{ddg_output}")
                print(f"{Fore.GREEN}Search query sent to DuckDuckGo: {query}{Style.RESET_ALL}")
                print(f"{Fore.GREEN}Time range sent to DuckDuckGo: {time_range}{Style.RESET_ALL}")
                print(f"{Fore.GREEN}Number of results: {len(results)}{Style.RESET_ALL}")
                return [{'number': i+1, **result} for i, result in enumerate(results)]
            except Exception as e:
                print(f"{Fore.RED}Search error: {str(e)}{Style.RESET_ALL}")
                return []

    def display_search_results(self, results: List[Dict]):
        print(f"\n{Fore.CYAN}Search Results:{Style.RESET_ALL}")
        for result in results:
            print(f"{Fore.GREEN}Result {result['number']}:{Style.RESET_ALL}")
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"Snippet: {result.get('body', 'N/A')[:200]}...")  # Limit to 200 characters
            print(f"URL: {result.get('href', 'N/A')}\n")

    def select_relevant_pages(self, search_results: List[Dict], user_query: str) -> List[str]:
        prompt = f"""
Given the following search results for the user's question: "{user_query}"
Select the 2 most relevant results to scrape and analyze. Explain your reasoning for each selection.

Search Results:
{self.format_results(search_results)}

Instructions:
1. You MUST select exactly 2 result numbers from the search results.
2. Choose the results that are most likely to contain comprehensive and relevant information to answer the user's question.
3. Provide a brief reason for each selection.

You MUST respond using EXACTLY this format and nothing else:

Selected Results: [Two numbers corresponding to the selected results]
Reasoning: [Your reasoning for the selections]
"""

        max_retries = 3
        for retry in range(max_retries):
            with OutputRedirector() as output:
                response = self.llm(prompt, max_tokens=200, stop=None)
            llm_output = output.getvalue()
            logger.info(f"LLM Output in select_relevant_pages:\n{llm_output}")
            response_text = response['choices'][0]['text'].strip()

            parsed_response = self.parse_page_selection_response(response_text)
            if parsed_response and self.validate_page_selection_response(parsed_response, len(search_results)):
                selected_urls = [result['href'] for result in search_results if result['number'] in parsed_response['selected_results']]

                # Check robots.txt for each selected URL
                allowed_urls = []
                for url in selected_urls:
                    if can_fetch(url):
                        allowed_urls.append(url)
                    else:
                        print(f"{Fore.YELLOW}Warning: Robots.txt disallows scraping of {url}{Style.RESET_ALL}")

                if allowed_urls:
                    return allowed_urls
                else:
                    print(f"{Fore.YELLOW}Warning: All selected URLs are disallowed by robots.txt. Retrying selection.{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Warning: Invalid page selection. Retrying.{Style.RESET_ALL}")

        # If all retries fail, fall back to top 2 results that are allowed by robots.txt
        print(f"{Fore.YELLOW}Warning: All attempts to select relevant pages failed. Falling back to top allowed results.{Style.RESET_ALL}")
        allowed_urls = [result['href'] for result in search_results if can_fetch(result['href'])][:2]
        return allowed_urls

    def parse_page_selection_response(self, response: str) -> Dict[str, Union[List[int], str]]:
        lines = response.split('\n')
        parsed = {}
        for line in lines:
            if line.startswith('Selected Results:'):
                parsed['selected_results'] = [int(num.strip()) for num in re.findall(r'\d+', line)]
            elif line.startswith('Reasoning:'):
                parsed['reasoning'] = line.split(':', 1)[1].strip()
        return parsed if 'selected_results' in parsed and 'reasoning' in parsed else None

    def validate_page_selection_response(self, parsed_response: Dict[str, Union[List[int], str]], num_results: int) -> bool:
        if len(parsed_response['selected_results']) != 2:
            return False
        if any(num < 1 or num > num_results for num in parsed_response['selected_results']):
            return False
        return True

    def format_results(self, results: List[Dict]) -> str:
        formatted_results = []
        for result in results:
            formatted_result = f"{result['number']}. Title: {result.get('title', 'N/A')}\n"
            formatted_result += f"   Snippet: {result.get('body', 'N/A')[:200]}...\n"  # Limit to 200 characters
            formatted_result += f"   URL: {result.get('href', 'N/A')}\n"
            formatted_results.append(formatted_result)
        return "\n".join(formatted_results)

    def scrape_content(self, urls: List[str]) -> Dict[str, str]:
        scraped_content = {}
        for url in urls:
            if can_fetch(url):
                content = get_web_content([url])
                if content:
                    scraped_content.update(content)
                    print(Fore.YELLOW + f"Successfully scraped: {url}" + Style.RESET_ALL)
                else:
                    print(f"{Fore.YELLOW}Warning: Failed to scrape content from {url}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Warning: Robots.txt disallows scraping of {url}{Style.RESET_ALL}")
        print(Fore.YELLOW + f"Scraped content received for {len(scraped_content)} URLs" + Style.RESET_ALL)
        return scraped_content

    def display_scraped_content(self, scraped_content: Dict[str, str]):
        print(f"\n{Fore.CYAN}Scraped Content:{Style.RESET_ALL}")
        for url, content in scraped_content.items():
            print(f"{Fore.GREEN}URL: {url}{Style.RESET_ALL}")
            print(f"Content: {content[:1000]}...\n")  # Display first 1000 characters

    def synthesize_or_refine(self, user_query: str, scraped_content: Dict[str, str]) -> Tuple[bool, str]:
        user_query_short = user_query[:200]
        prompt = f"""
Based on the user's question: "{user_query_short}"
And the following scraped content:
{self.format_scraped_content(scraped_content)}
Determine if the scraped content contains enough relevant information to answer the user's question comprehensively.
Respond with ONLY ONE of the following two options:
1. Refine: If the information is insufficient or unclear to answer the question completely.
2. Answer: If there is enough information to provide a comprehensive answer to the user's question.
Your response MUST be ONLY either "Refine" or "Answer".
"""
        max_retries = 3
        for attempt in range(max_retries):
            with OutputRedirector() as output:
                response = self.llm(prompt, max_tokens=10, stop=None)
            llm_output = output.getvalue()
            logger.info(f"LLM Output in synthesize_or_refine:\n{llm_output}")
            response_text = response['choices'][0]['text'].strip()
            if re.search(r'\b(Refine|refine)\b', response_text):
                return True, "refine"
            elif re.search(r'\b(Answer|answer)\b', response_text):
                return False, "answer"
        return True, "refine"  # Default to refine if we can't get a valid response

    def generate_final_answer(self, user_query: str, scraped_content: Dict[str, str]) -> str:
        user_query_short = user_query[:200]
        prompt = f"""
You are an AI assistant. Provide a comprehensive and detailed answer to the following question using ONLY the information provided in the scraped content. Do not include any references or mention any sources. Answer directly and thoroughly.

Question: "{user_query_short}"

Scraped Content:
{self.format_scraped_content(scraped_content)}

Answer:
"""
        max_retries = 3
        for attempt in range(max_retries):
            with OutputRedirector() as output:
                response = self.llm(prompt, max_tokens=1024, stop=None)
            llm_output = output.getvalue()
            logger.info(f"LLM Output in generate_final_answer:\n{llm_output}")
            response_text = response['choices'][0]['text'].strip()
            if self.is_valid_response(response_text):
                return response_text
        return "I apologize, but I couldn't generate a satisfactory answer based on the available information."

    def is_valid_response(self, response_text: str) -> bool:
        # Basic validation to ensure response is meaningful
        return len(response_text) > 50  # Adjust the threshold as needed

    def format_scraped_content(self, scraped_content: Dict[str, str]) -> str:
        formatted_content = []
        for url, content in scraped_content.items():
            content = re.sub(r'\s+', ' ', content)
            formatted_content.append(f"Content from {url}:\n{content}\n")  # No character limit
        return "\n".join(formatted_content)

    def synthesize_final_answer(self, user_query: str) -> str:
        prompt = f"""
After multiple search attempts, we couldn't find a fully satisfactory answer to the user's question: "{user_query}"

Please provide the best possible answer you can, acknowledging any limitations or uncertainties.
If appropriate, suggest ways the user might refine their question or where they might find more information.

Respond in a clear, concise, and informative manner.
"""
        try:
            with OutputRedirector() as output:
                response = self.llm(prompt, max_tokens=self.llm_config['max_tokens'], stop=self.llm_config['stop'])
            llm_output = output.getvalue()
            logger.info(f"LLM Output in synthesize_final_answer:\n{llm_output}")
            final_answer = response['choices'][0]['text'].strip()
            return final_answer
        except Exception as e:
            logger.error(f"Error in synthesize_final_answer: {str(e)}", exc_info=True)
            return "I apologize, but after multiple attempts, I wasn't able to find a satisfactory answer to your question. Please try rephrasing your question or breaking it down into smaller, more specific queries."

# End of EnhancedSelfImprovingSearch class
