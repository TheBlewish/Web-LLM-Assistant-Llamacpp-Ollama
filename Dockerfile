FROM python:3.13-slim-bookworm

# Create a non-root user
RUN useradd --create-home appuser

# Set working directory
WORKDIR /home/appuser/app

# Copy project files
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Install Python dependencies - System-wide for all users
RUN python -m pip install --no-cache-dir -r requirements.txt

# Set permissions (this may not be strictly necessary anymore)
RUN chown -R appuser:appuser /home/appuser/app

# Switch to non-root user
USER appuser

# Set default environment variables
ENV LLM_TYPE=ollama
ENV OLLAMA_BASE_URL=http://localhost:11434

# Set entry point
CMD ["python", "Web-LLM.py"]