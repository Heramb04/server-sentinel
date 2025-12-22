# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy dependency list first (for caching)
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project files
COPY . .

# 6. Expose the port your app runs on (Flask defaults to 5000)
EXPOSE 5000
EXPOSE 80

# 7. Startup command
# Replace 'app.py' with the entrypoint of your project
CMD ["python", "app.py"]
