# Use a base image that includes Python, as you're using pip
FROM python:3.11-slim

# Install imagemagick and ffmpeg
RUN apt-get update && \
    apt-get install -y imagemagick ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install moviepy via pip
RUN pip install moviepy

# Install dependencies from the requirements.txt file
RUN pip install -r server-bot/requirements.txt

# Modify the ImageMagick policy
RUN sed -i '/<policy domain="path" rights="none" pattern="@\*"/d' /etc/ImageMagick-6/policy.xml

# Copy your application code into the Docker container
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Specify the command to run your application
# This should be adjusted to how you run your Python application
CMD ["python", "your_app.py"]

# Expose the port your app runs on
EXPOSE 8080
