# pull official base image
FROM python:3.8

# set work directory
WORKDIR /backend

# set environment variable
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get -y update && apt-get -y upgrade
# ImportError: libGL.so.1: cannot open shared object file: No such file or directory를 해결하기 위해 아래 코드 추가
RUN apt-get install -y libgl1

COPY . /backend/

# install dependencies
RUN pip install --upgrade pip
RUN pip install cmake && pip install dlib
RUN pip install -r requirements.txt
