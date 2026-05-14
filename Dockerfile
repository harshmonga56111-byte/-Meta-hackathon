FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic openai requests

# Copy all files from your repo to the container
COPY . .

# Expose the port HF Spaces uses
EXPOSE 7860

# Run the app (Change "server.app:app" if your file is named differently)
# If app.py is in the root, use: "app:app"
# If app.py is inside the server folder, use: "server.app:app"
CMD ["uvicorn", "server.app:main", "--host", "0.0.0.0", "--port", "7860"]