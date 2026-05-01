FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

# Expose port
EXPOSE 8000

# Command to run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
