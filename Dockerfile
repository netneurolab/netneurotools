FROM python:3.9.7-slim

RUN apt-get update \
    && apt-get install -y \
        wget \
        unzip \
        libgl1-mesa-glx \
        libglu1-mesa \
        libgomp1 \
        libglib2.0-0

COPY . netneurotools

RUN cd netneurotools \
    && python3 -m pip install '.[numba]'

CMD python3