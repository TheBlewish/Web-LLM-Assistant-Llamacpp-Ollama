import time
import re
from typing import List, Dict, Tuple
from llama_cpp import Llama
from duckduckgo_search import DDGS
from colorama import Fore, Style
import logging
import sys
from io import StringIO

# Set up logging
logging.basicConfig(filename='llama_output.log', level=logging.INFO, format='%(asctime)s - %(message)s')

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

class SimpleSelfImprovingSearch:
    def __init__(self, llm: Llama, max_attempts: int = 5):
        self.llm = llm
        self.max_attempts = max_attempts

    def print_thinking(self):
        print(Fore.MAGENTA + "\n🧠 Thinking", end="")
        for _ in range(3):
            time.sleep(0.5)
            print(".", end="", flush=True)
        print("\n" + Style.RESET_ALL)

    def print_searching(self):
        print(Fore.MAGENTA + "📝 Searching", end="")
        for _ in range(3):
            time.sleep(0.5)
            print(".", end="", flush=True)
        print("\n" + Style.RESET_ALL)

    def search_and_improve(self, user_query: str) -> str:
        attempt = 0
        while attempt < self.max_attempts:
            print(f"\n{Fore.CYAN}Search attempt {attempt + 1}:{Style.RESET_ALL}")
            self.print_searching()

            try:
                # Formulate search query
                formulated_query, time_range, reasoning = self.formulate_query(user_query, attempt)

                # Display information to the user
                print(f"{Fore.YELLOW}Original query: {user_query}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Formulated query: {formulated_query}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Time range: {time_range}{Style.RESET_ALL}")
                print(f"{Fore.MAGENTA}LLM Reasoning: {reasoning}{Style.RESET_ALL}")

                if not formulated_query:
                    print(f"{Fore.RED}Error: Empty search query. Retrying...{Style.RESET_ALL}")
                    attempt += 1
                    continue

                # Perform search
                results = self.perform_search(formulated_query, time_range)

                if not results:
                    print(f"{Fore.RED}No results found. Retrying with a different query...{Style.RESET_ALL}")
                    attempt += 1
                    continue

                # Display search results
                self.display_search_results(results)

                # Add thinking indicator after displaying results
                self.print_thinking()

                # Evaluate results and decide next action
                evaluation, next_action = self.evaluate_results_and_decide(results, user_query)
                print(f"{Fore.MAGENTA}Evaluation: {evaluation}{Style.RESET_ALL}")
                print(f"{Fore.MAGENTA}Next action: {next_action}{Style.RESET_ALL}")

                if next_action == "answer":
                    return self.synthesize_answer(results, user_query)
                elif next_action == "refine":
                    attempt += 1
                else:
                    print(f"{Fore.YELLOW}Unexpected next action '{next_action}'. Defaulting to 'answer'.{Style.RESET_ALL}")
                    return self.synthesize_answer(results, user_query)

            except Exception as e:
                print(f"{Fore.RED}An error occurred during search attempt: {str(e)}{Style.RESET_ALL}")
                attempt += 1

        return self.synthesize_final_answer(user_query)

    def formulate_query(self, user_query: str, attempt: int) -> Tuple[str, str, str]:
        prompt = f"""
        Your task is to formulate a search query to get relevant information to specifically answer the user's question.
        User's question: "{user_query}"
        Current attempt number: {attempt + 1}

        Based on the user's question, formulate a search query that is likely to yield relevant results.
        Consider specifying a time range if the query requires recent information.

        Respond using EXACTLY this format:

        Reasoning: [Your reasoning for the formulated query]
        Query: [ONLY the search query goes here]
        Time Range: [ONLY 'd' for past day, 'w' for past week, 'm' for past month, 'y' for past year, or 'none' for all time]
        """

        with OutputRedirector() as output:
            response = self.llm(prompt, max_tokens=300)
        response_text = response['choices'][0]['text'].strip()

        reasoning, query, time_range = self.advanced_parse(response_text)

        query = self.clean_query(query or user_query)
        time_range = self.validate_time_range(time_range)

        return query, time_range, reasoning

    def advanced_parse(self, text: str) -> Tuple[str, str, str]:
        reasoning, query, time_range = "", "", "none"

        reasoning_match = re.search(r'Reasoning:(.+?)(?:Query:|Time Range:|$)', text, re.DOTALL | re.IGNORECASE)
        query_match = re.search(r'Query:(.+?)(?:Reasoning:|Time Range:|$)', text, re.DOTALL | re.IGNORECASE)
        time_range_match = re.search(r'Time Range:(.+?)(?:Reasoning:|Query:|$)', text, re.DOTALL | re.IGNORECASE)

        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        if query_match:
            query = query_match.group(1).strip()
        if time_range_match:
            time_range = time_range_match.group(1).strip().lower()

        return reasoning, query, time_range

    def clean_query(self, query: str) -> str:
        query = query.replace('"', '').replace('\n', ' ').strip()
        query = re.sub(r'\s+', ' ', query)
        return query[:100]

    def validate_time_range(self, time_range: str) -> str:
        valid_ranges = ['d', 'w', 'm', 'y', 'none']
        time_range = time_range.lower()
        return time_range if time_range in valid_ranges else 'none'

    def perform_search(self, query: str, time_range: str) -> List[Dict]:
        if not query:
            return []

        with DDGS() as ddgs:
            try:
                if time_range and time_range != 'none':
                    results = list(ddgs.text(query, timelimit=time_range, max_results=4))
                else:
                    results = list(ddgs.text(query, max_results=4))
                print(f"{Fore.GREEN}Search query sent to DuckDuckGo: {query}{Style.RESET_ALL}")
                print(f"{Fore.GREEN}Time range sent to DuckDuckGo: {time_range}{Style.RESET_ALL}")
                print(f"{Fore.GREEN}Number of results: {len(results)}{Style.RESET_ALL}")
                return results
            except Exception as e:
                print(f"{Fore.RED}Search error: {str(e)}{Style.RESET_ALL}")
                return []

    def display_search_results(self, results: List[Dict]):
        print(f"\n{Fore.CYAN}Search Results:{Style.RESET_ALL}")
        for i, result in enumerate(results, 1):
            print(f"{Fore.GREEN}Result {i}:{Style.RESET_ALL}")
            print(f"Title: {result.get('title', 'N/A')}")

            # Try to get a longer snippet
            body = result.get('body', '')
            sentences = body.split('.')
            snippet = '. '.join(sentences[:3]) + ('...' if len(sentences) > 3 else '')

            print(f"Snippet: {snippet}")
            print(f"URL: {result.get('href', 'N/A')}\n")

    def evaluate_results_and_decide(self, results: List[Dict], user_query: str) -> Tuple[str, str]:
        prompt = f"""
        Given the user's question: "{user_query}"
        And the following search results:
        {self.format_results(results)}

        Your task is to evaluate if these results contain enough relevant and detailed information to comprehensively answer the user's question.

        Respond using EXACTLY this format:

        Evaluation: [Your evaluation of the search results]
        Next Action: [ONLY 'answer' if results are sufficient for a comprehensive answer, or 'refine' if more detailed information is needed]
        """

        try:
            with OutputRedirector() as output:
                response = self.llm(prompt, max_tokens=300)
            response_text = response['choices'][0]['text'].strip()

            evaluation, next_action = self.parse_evaluation_response(response_text)

            if not evaluation or not next_action:
                print(f"{Fore.YELLOW}Warning: Unable to parse evaluation response. Defaulting to 'refine'.{Style.RESET_ALL}")
                return "Unable to evaluate results.", "refine"

            return evaluation, next_action
        except Exception as e:
            print(f"{Fore.RED}Error in evaluating results: {str(e)}. Defaulting to 'refine'.{Style.RESET_ALL}")
            return "Error in evaluation.", "refine"

    def parse_evaluation_response(self, response_text: str) -> Tuple[str, str]:
        evaluation_match = re.search(r'Evaluation:\s*(.+?)(?:\n|$)', response_text, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r'Next Action:\s*(.+?)(?:\n|$)', response_text, re.DOTALL | re.IGNORECASE)

        evaluation = evaluation_match.group(1).strip() if evaluation_match else ""
        next_action = action_match.group(1).strip().lower() if action_match else ""

        if 'answer' in next_action:
            next_action = 'answer'
        elif 'refine' in next_action:
            next_action = 'refine'
        else:
            next_action = 'refine'

        return evaluation, next_action

    def format_results(self, results: List[Dict]) -> str:
        if not results:
            return "No results found."

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_result = f"Result {i}:\n"
            formatted_result += f"Title: {result.get('title', 'N/A')}\n"

            body = result.get('body', '')
            sentences = body.split('.')
            snippet = '. '.join(sentences[:3]) + ('...' if len(sentences) > 3 else '')

            formatted_result += f"Snippet: {snippet}\n"
            formatted_result += f"URL: {result.get('href', 'N/A')}\n"
            formatted_results.append(formatted_result)
        return "\n".join(formatted_results)

    def synthesize_answer(self, results: List[Dict], user_query: str) -> str:
        prompt = f"""
        Given the user's question: "{user_query}"
        And the following search results:
        {self.format_results(results)}

        Your task:
        1. Using ONLY the information from the search results, create a comprehensive and detailed answer to the user's question.
        2. Ensure your answer is directly relevant to the user's question.
        3. DO NOT include any follow-up questions, additional answers, or any content not directly answering the user's question.
        4. DO NOT ask any further questions or provide anything other than the answer to the user's question.
        5. DO NOT generate a conversation or a series of Q&As.

        Use EXACTLY this format to answer:
        Response: [your answer to the User's question]

        Remember: Provide ONLY the response to the user's original question. No additional content whatsoever.
        """

        try:
            with OutputRedirector() as output:
                response = self.llm(prompt, max_tokens=1000)

            # Extract only the content after "Response:" and before any new line starting with "User:" or a blank line
            answer = response['choices'][0]['text'].strip()
            match = re.search(r'Response:\s*(.*?)(?:\n\s*User:|$)', answer, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return "I apologize, but I couldn't generate a proper response to your question."
        except Exception as e:
            print(f"{Fore.RED}Error in synthesizing answer: {str(e)}{Style.RESET_ALL}")
            return "I apologize, but I encountered an error while trying to synthesize an answer to your question. Please try asking your question again or rephrase it."

    def synthesize_final_answer(self, user_query: str) -> str:
        prompt = f"""
        After multiple search attempts, we couldn't find a fully satisfactory answer to the user's question: "{user_query}"

        Please provide the best possible answer you can, acknowledging any limitations or uncertainties.
        If appropriate, suggest ways the user might refine their question or where they might find more information.
        """

        try:
            with OutputRedirector() as output:
                response = self.llm(prompt, max_tokens=1000)
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"{Fore.RED}Error in synthesizing final answer: {str(e)}{Style.RESET_ALL}")
            return "I apologize, but after multiple attempts, I wasn't able to find a satisfactory answer to your question. Please try rephrasing your question or breaking it down into smaller, more specific queries."
