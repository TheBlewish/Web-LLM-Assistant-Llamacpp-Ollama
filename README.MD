# Web-LLM Assistant

## Description
Web-LLM Assistant is an simple web search assistant that leverages a large language model (LLM) running via Llama.cpp to provide informative and context-aware responses to user queries. This project combines the power of LLMs with real-time web searching capabilities, allowing it to access up-to-date information and synthesize comprehensive answers.

## Features
- Local LLM usage via llama_cpp for reduced latency and better control
- Web searching using DuckDuckGo for privacy-focused, up-to-date information
- Self-improving search mechanism that refines queries based on initial results
- Rich console output with colorful and animated indicators for better user experience
- Multi-attempt searching with intelligent evaluation of search results
- Comprehensive answer synthesis using both LLM knowledge and web search results

## dependencies:
Of course given that this is using an LLM running via Llama.cpp ensure you have that installed and an appropriate model, for me I used Phi-3-medium-128k-instruct-Q6_K.gguf but feel free to use any model you want.

## Installation

1. Clone the repository:
2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
3. Install the required dependencies:
pip install -r requirements.txt
4. Download the LLM model:
Download the Phi-3-medium-128k-instruct-Q6_K.gguf model (or another one this is just the one I have used with it) and place it in the appropriate directory (default: `/home/llama.cpp/models/` can be changed via editing model path in Web-LLM.py).

## Usage

Run the main script:
python Web-LLM.py

- For normal interaction, simply type your message and press CTRL+D to submit.
- To request a web search, start your message with '/'.
  Example: "/latest news on AI advancements"

The AI will process your input, perform a search if requested, and provide an informed response.

## Configuration

You can modify the following parameters in the `Web-LLM.py` file:
- `MODEL_PATH`: Path to your LLM model file
- `max_attempts`: Maximum number of search attempts (default: 5)
- Or really anything you'd like it's only two python scripts.

## Dependencies

- llama-cpp-python
- see full list of depencies in the requirements.txt

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
