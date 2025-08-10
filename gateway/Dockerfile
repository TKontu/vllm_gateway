# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code to the working directory
COPY . .

# Command to run the application
# Use 0.0.0.0 to make it accessible from outside the container
# The port 9000 matches what you have in docker-compose.yml
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]