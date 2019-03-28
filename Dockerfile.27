FROM python:2.7
MAINTAINER "Franklin Koch <franklin.koch@seequent.com>"

RUN apt-get update && apt-get -y install vim

# Make the destination directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy resources
COPY setup.py requirements.txt requirements_dev.txt Makefile README.rst /usr/src/app/

# Install requirements
RUN pip install -r requirements_dev.txt
