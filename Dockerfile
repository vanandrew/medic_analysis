FROM ubuntu:22.04

FROM ubuntu:22.04 as base
LABEL maintainer="Andrew Van <vanandrew@wustl.edu>"

# set noninteractive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# install dependencies
RUN apt-get update && apt-get install -y build-essential ftp tcsh wget curl git jq vim \
    python3 python3-pip gfortran tcl wish unzip dc bc libglu1-mesa libglib2.0-0 && \
    pip install --upgrade pip

WORKDIR /opt

# install julia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.3-linux-x86_64.tar.gz && \
    tar -xzf julia-1.8.3-linux-x86_64.tar.gz && \
    rm julia-1.8.3-linux-x86_64.tar.gz && mv /opt/julia-1.8.3/ /opt/julia/ && \
    echo "/opt/julia/lib" >> /etc/ld.so.conf.d/julia.conf && ldconfig
ENV PATH=/opt/julia/bin:${PATH}

WORKDIR /
