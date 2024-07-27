# constrained_llm_generation

## Docker compose

### GPU
#### Requirements
You need an env with an nvidia GPU with the necessary drivers and docker compose.

* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
* https://docs.docker.com/compose/install/linux/

#### Run:

```commandline
docker compose up --build
```
### CPU

TODO: If you need this one, open an issue and I'll work on it


## Local install

### Install llama.cpp python

#### CPU
```bash
CMAKE_ARGS="-DLLAVA_BUILD=OFF" pip install llama-cpp-python==v0.2.75 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --force-reinstall --no-cache-dir
```

#### GPU
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==v0.2.75
```

Then install all the requirement files.