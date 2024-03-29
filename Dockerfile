
# First working version at 24/12/2023:
FROM pytorch/pytorch:latest

# prevent apt from hanging
ARG DEBIAN_FRONTEND=noninteractive

ENV HOME /workspace

WORKDIR $HOME

# general system dependencies:
RUN apt update
RUN apt install -y git
RUN apt install libgl1-mesa-glx -y
RUN apt install libglib2.0-0 -y

# dependency: lang-segment-anything:
RUN git clone https://github.com/kauevestena/lang-segment-anything.git
WORKDIR $HOME/lang-segment-anything
RUN pip install groundingdino-py
RUN pip install segment-anything-py
RUN python running_test.py

# CLIP:
RUN pip install ftfy regex tqdm
RUN pip install git+https://github.com/openai/CLIP.git

# this repository stuff:
WORKDIR $HOME
ENV REPODIR $HOME/deep_pavements_sample_picker
COPY . $REPODIR
WORKDIR $REPODIR
RUN pip install -r requirements.txt
COPY mapillary_token configs/mapillary_token
RUN python build_tests/test_configs.py
# RUN python build_tests/test_clip.py
# RUN python build_tests/test_openclip.py

# saving frozen requirements:
RUN pip list --format=freeze > frozen_requirements.txt

