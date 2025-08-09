FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# 기본 유틸 설치 (wget, ca-certificates 등)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
        bzip2 \
        git \
        bash && \
    rm -rf /var/lib/apt/lists/*

# Miniconda 설치
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# conda 바이너리 PATH 등록
ENV PATH="/opt/miniconda/bin:${PATH}"

WORKDIR /app

# conda env 캐시 최적화를 위해 우선 환경 파일만 복사
COPY environment.yml /tmp/environment.yml

# 채널 구성 및 mamba 설치 후 환경 생성 (Anaconda TOS 회피)
RUN conda config --system --set channel_priority flexible && \
    (conda config --system --remove channels defaults || true) && \
    conda config --system --add channels pytorch && \
    conda config --system --add channels nvidia && \
    conda config --system --add channels conda-forge && \
    conda update -n base -c conda-forge -y conda && \
    conda install -n base -c conda-forge -y mamba && \
    mamba env create -f /tmp/environment.yml --override-channels -c pytorch -c nvidia -c conda-forge

# 셸에서 기본 활성화 설정
RUN echo "source /opt/miniconda/etc/profile.d/conda.sh && conda activate audiotools" >> /root/.bashrc

# 런타임 PATH 및 기본 env 설정 (non-login/non-interactive 셸 대응)
ENV CONDA_DEFAULT_ENV=audiotools
ENV PATH="/opt/miniconda/envs/audiotools/bin:/opt/miniconda/bin:${PATH}"

# 나머지 프로젝트 파일 복사
COPY . .

CMD ["bash"]

# Labels
LABEL maintainer="HojoonKi"
LABEL description="Audio Manipulator: Text-to-Audio Effect Generation"
LABEL version="1.0"
