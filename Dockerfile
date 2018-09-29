FROM pytorch/pytorch:latest

RUN pip install ipdb

RUN pip install torchtext

ENV LC_ALL C.UTF-8

RUN pip install tensorboardX
#ENV LANG=C.UTF-8
