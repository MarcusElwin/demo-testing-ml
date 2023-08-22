# pull official base image
FROM python:3.10-slim-buster

LABEL maintainer="Marcus Elwin"

# set working directory
WORKDIR /usr/code/

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install python dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt --quiet
RUN pip freeze