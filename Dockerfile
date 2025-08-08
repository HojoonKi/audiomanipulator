# Use Miniconda3 as the base image
FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy all files to the container
COPY . .

# Create conda env and install all dependencies
RUN conda env create -f environment.yml

# Activate conda env by default in all shells
RUN echo "conda activate audiotools" >> /root/.bashrc

# Set PATH and CONDA_DEFAULT_ENV
ENV PATH /opt/conda/envs/audiotools/bin:$PATH
ENV CONDA_DEFAULT_ENV audiotools

CMD ["bash"]

# Labels
LABEL maintainer="HojoonKi"
LABEL description="Audio Manipulator: Text-to-Audio Effect Generation"
LABEL version="1.0"
