FROM ubuntu:20.04
RUN apt-get update && apt-get install -y  \
    python3.8 \
    python3-pip \
    make
COPY . ./hackathon/
RUN pip3 install -r hackathon/requirements.txt
WORKDIR /hackathon

