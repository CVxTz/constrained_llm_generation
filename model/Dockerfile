ARG CUDA_IMAGE="12.2.0-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

WORKDIR /app

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

ENV CUDA_DOCKER_ARCH=all
ENV LLAMA_CUBLAS=1

RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==v0.2.75

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user and set up permissions
# RUN useradd -m myuser

# Copy all .py files into the container
COPY *.py .

# Change ownership of the /app directory to the new user
# RUN chown -R myuser:myuser /app

# Switch to the non-root user
# USER myuser

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH python3 model_llama_cpp.py
RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1

# Command to run on container start
CMD ["python3", "worker.py"]