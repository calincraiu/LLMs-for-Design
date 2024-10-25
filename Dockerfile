# NOTE: WIP - This DOCKERFILE is not yet completed.

FROM python:3.9

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

# Copy the app.
COPY ./app.py /app