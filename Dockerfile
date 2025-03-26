FROM nvidia/cuda:12.4.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Set working directory
WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Set up environment activation
SHELL ["conda", "run", "-n", "burr-detection", "/bin/bash", "-c"]

# Copy the project files
COPY . .

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Default command
ENTRYPOINT ["conda", "run", "-n", "burr-detection", "python3"]
