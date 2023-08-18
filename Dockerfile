FROM ubuntu:22.04

FROM ubuntu:22.04 as base
LABEL maintainer="Andrew Van <vanandrew@wustl.edu>"

# set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# install dependencies
RUN apt-get update && apt-get install -y build-essential ftp tcsh wget git jq \
    python3 python3-pip gfortran tcl wish unzip dc bc libglu1-mesa libglib2.0-0

WORKDIR /
