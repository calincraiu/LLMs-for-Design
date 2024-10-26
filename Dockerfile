# NOTE: WIP - This DOCKERFILE is not yet completed.

FROM python:3.9

# Copy only the files and directories specified into the container
COPY requirements.txt ./
COPY .env .env
COPY app/ app/
COPY resources/ resources/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

CMD ["fastapi", "run", "app/app.py", "--port", "8000"]