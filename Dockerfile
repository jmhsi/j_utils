# Base image
FROM ubuntu:16.04 
MAINTAINER justinm.hsi@gmail.com

# Temporary
# RUN uname

# Update repo sources
RUN apt-get update --fix-missing && apt-get install -y \ 
    python3-pip

RUN pip3 --no-cache-dir install --upgrade \
    hypothesis \
    numpy \
    pandas
