# Use Python 3.12.7 as the base image
FROM python:3.12.7

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt into the container
COPY src/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY /src .

# Expose port 8000 (if your FastAPI server listens on that port)
EXPOSE 5060

# Run the blogging agent script
CMD ["python", "data_collector.py"]
