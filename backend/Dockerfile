# pull official base image
FROM python:3.8.0-alpine

# set work directory
WORKDIR /backend

# set environment variable
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apk update
RUN apk add gcc musl-dev python3-dev zlib-dev jpeg-dev

COPY . /backend/

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
