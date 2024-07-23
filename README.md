# constrained_llm_generation

https://hub.docker.com/r/huggingface/transformers-pytorch-gpu 

Multi-class, Multi-label sentence classification
Token classification
Dates and numbers
CoT
Tool use:  Two ways: Easy + Advanced


CMAKE_ARGS="-DLLAVA_BUILD=OFF" pip install llama-cpp-python   --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --force-reinstall --no-cache-dir

CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==v0.2.75 --upgrade --force-reinstall --no-cache-dir
