# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set the working directory
WORKDIR /opencap-processing

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

# Install Anaconda with absolute paths
RUN wget  https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh --no-check-certificate -O /root/anaconda.sh && \
    chmod +x /root/anaconda.sh && \
    /root/anaconda.sh -b -p /opt/conda && \
    rm /root/anaconda.sh && \
    chown -R root:root /opt/conda

# Update PATH environment variable
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment with Python 3.11
RUN conda create -y -n opencap-processing python=3.11

# Activate the environment and install OpenSim
RUN /opt/conda/bin/conda run -n opencap-processing conda install -y -c opensim-org opensim=4.5=py311np123

# Clone the opencap-processing repository with SSL verification disabled
RUN git -c http.sslVerify=false clone https://github.com/shubhMaheshwari/opencap-processing.git .

# Install required Python packages
RUN /opt/conda/bin/conda run -n opencap-processing pip install --no-cache-dir -r requirements.txt

# Install OpenBLAS libraries (already installed above)
# RUN apt-get update && apt-get install -y libopenblas-base

# Expose ports if needed (uncomment if your application uses networking)
# EXPOSE 8000