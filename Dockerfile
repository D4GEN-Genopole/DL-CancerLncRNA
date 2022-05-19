#FROM python:3.8
#
## create working directory and install pip dependencies
#WORKDIR /hackathon
#COPY requirements.txt requirements.txt
#RUN pip3 install -r requirements.txt
#
## copy python project files from local to /hello-py image working directory
#COPY . .
#
## run the flask server
#CMD [ "python3", "-m", "main_sequences"]

FROM ubuntu:20.04
#RUN apt-get -y update && apt-get -y upgrade
RUN apt-get update && apt-get install -y  \
    python3.8 \
    python3-pip \
    make

#RUN apt-get install -y python3-pip python3.8  make
COPY . ./hackathon/
RUN pip3 install -r hackathon/requirements.txt
WORKDIR /hackathon

