FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY ./src /app/src
COPY ./configs /app/configs
COPY ./train*.py /app
COPY ./requirements.txt /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install package
RUN pip install -r /app/requirements.txt


# Default entrypoint runs the stage1 script
ENTRYPOINT ["/bin/bash", "run.sh"]