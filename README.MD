# Web-LLM Assistant

## Description
Web-LLM Assistant is an simple web search assistant that leverages a large language model (LLM) running via Llama.cpp to provide informative and context-aware responses to user queries. This project combines the power of LLMs with real-time web searching capabilities, allowing it to access up-to-date information and synthesize comprehensive answers.

Here is how it works in practice:

You can ask the LLM a question, for example: "Is the boeing starliner still stuck on the international space station", then the LLM will decide on a search query and a time frame for which to get search results from, such as results from the last day or the last year, depending on the needs of your specific question.

Then it will perform a web search, and collect the first 10 results and the information contained within them, it then will select 2 most relevant results and web scrape them to acquire the information contained within those results, after reviewing the information it will decide whether or not the information is sufficient to answer the your question. If it is then the LLM will answer the question, if it isn't then the LLM will perform a new search, likely rephrasing the search terms and/or time-frame, to find more appropriate and relevant information to use to answer your question, it can continue to do multiple searches refining the search terms or time-frame until it either has enough information to actually answer the User's question, or until it has done 5 separate searches, retrieving information from the the LLMs decided top 2 relevant results of each search, at which time if it hasn't been able to find the information needed to answer the User's question it will try it's best to provide whatever information it has acquired from the searches at that point to answer your question the best it can.

Thus allowing you to ask it queries about recent events, or anything that may not actually be in it's training data. Which it can now, via this python program still determine the answer to your question, even if the answer is absent from the LLM's training data via web searching and retrieving information from those searches.

## Project Demonstration (outdated since recent update)

[![Web-LLM-Assistant Llama-cpp demonstration](https://img.youtube.com/vi/3t4_NdwmgUw/0.jpg)](https://www.youtube.com/watch?v=3t4_NdwmgUw "Web-LLM-Assistant Llama-cpp demonstration")

Click the image above to watch a demonstration of the Web-LLM-Assistant Llama-cpp in action.

## Features
- Local LLM usage via llama_cpp for reduced latency and better control
- Web searching using DuckDuckGo for privacy-focused, up-to-date information
- Self-improving search mechanism that refines queries based on initial results
- Rich console output with colorful and animated indicators for better user experience
- Multi-attempt searching with intelligent evaluation of search results
- Comprehensive answer synthesis using both LLM knowledge and web search results
- Web scraping of search results for full information for the LLM to utilise

## Installation

1. Clone the repository:
2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
3. Install the required dependencies:
pip install -r requirements.txt
4. Download the LLM model:
Download the Phi-3-medium-128k-instruct-Q6_K.gguf model (or another one this is just the one I have used with it) and place it in the appropriate directory (default: `/home/llama.cpp/models/` can be changed via editing model path in llm_config.py also you can adjust llama.cpp settings in that file including GPU layers if you want to use your GPU with it!).

## Usage

Run the main script:
python Web-LLM.py

- For normal interaction, simply type your message and press CTRL+D to submit.
- To request a web search, start your message with '/'.
  Example: "/latest news on AI advancements"

The AI will process your input, perform a search if requested, and provide an informed response.

## Configuration

You can modify various llama.cpp parameters in the `llm_config.py` file including model used, GPU layers or anything you can as with normal llama.cpp use!

## Dependencies

- llama-cpp-python
- see full list of dependencies in the requirements.txt
- Given that this is using an LLM running via Llama.cpp ensure you have that installed an appropriate model, I would reccommend gemma-2-9b-it-Q6_K.gguf but feel free to use any model you want.

## Recent Updates!

in a recent and extensive update, I have now implemented webscraping allowing the program to have the LLM make a search, and pick the 2 most relevant out of 10 results which are then scraped to retrieve information from those two with which to answer the User's question. I have also added an llm_config.py file with which you can alter your llama.cpp generation settings as you would like, including changing the GPU layers option if you have llama.cpp compiled with GPU support enabled to run this program with your GPU!

Also a significant bug has now been entirely fixed, the bug was causing the LLM to state "Due to the absence of results" rather then answer the User's question this has now been completely resolved please let me know if you have any other bugs and I will endeavour to fix it for you!

## Contributing

Contributions to improve Web-LLM Assistant are welcome and encouraged! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE] file for details.

## Acknowledgments

- Thanks to the creators of llama.cpp for providing the foundation for local LLM usage.
- DuckDuckGo for their search API.

## Disclaimer

This project is for educational purposes only. Ensure you comply with the terms of service of all APIs and services used.

## Personal Note

I tried my best to create something that allows for the use of local llama.cpp run LLM's for web-searching, always in the past being frustrated that while services like ChatGPT could do web searching while local models were never easily able to.

Web-LLM Assistant represents countless hours of learning, coding, and problem-solving and is actually my first ever attempt at anything coding related especially anything I built from scratch.

If anyone who knows a lot more then me wants to dive in and make this magnitudes better, that would be fantastic. I believe this is the best I can do at my current level of knowledge, it's still a work in proggress and has it's issues, but it's certainly better at searching the web then llama.cpp alone (which obviously cant at all). So at least in that respect I would say that my goal has been achieved although it certainly has it's problems, i'm pretty happy with what I made for my first ever coding project, would love to know what others think!
