import time

import torch
from guidance import assistant, gen, role, select
from guidance.chat import ChatMLTemplate
from guidance.models import Transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

# checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"
# checkpoint = "HuggingFaceTB/SmolLM-360M-Instruct"
checkpoint = "HuggingFaceTB/SmolLM-1.7B-Instruct"

device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map=device)

if __name__ == "__main__":
    classes = ["positive", "negative", "neutral"]
    classes_ = " ".join(classes)
    messages = [
        {
            "role": "system",
            "content": f"Your role is to classify the input sentence into {classes_} classes. "
                       f"Answer with one of {classes_} values."
            # "Respond with negative or positive or neutral "
            # "* positive: for positive emotion "
            # "* negative: for negative emotion "
            # "* neutral: for neutral or no emotion. "
            # "Provide a rational or explanation.",
        },
        {"role": "user", "content": "I liked the movie"},
        {
            "role": "assistant",
            "content": "Rational: Lets think step by step, 'like' reflects a positive emotion so the answer is: "
                       "positive",
        },
        {"role": "user", "content": "I hated the movie"},
        {
            "role": "assistant",
            "content": "Rational: Lets think step by step, 'hate' reflects a negative emotion so the answer is: "
                       "negative",
        },
        # {"role": "user", "content": "I watched the movie"},
        # {
        #     "role": "assistant",
        #     "content": "Rational: Lets think step by step, no emotion was expressed so the answer is: neutral",
        # },
        # {"role": "user", "content": "This was not a fun experience"},
        # {
        #     "role": "assistant",
        #     "content": "Rational: Lets think step by step, 'not a fun' reflects a negative emotion so the answer is: "
        #                "negative",
        # },
        # {"role": "user", "content": "I watched the movie"},
        # {
        #     "role": "assistant",
        #     "content": "Rational: Lets think step by step, no emotion was expressed so the answer is: neutral",
        # },
        {"role": "user", "content": "Sentence: Sunny Coffee ☀️Get more positive energy with 🌼 Lofi Coffee ~ Lofi Hip Hop for relax/chill/study"},
        # {
        #     "role": "user",
        #     "content": "Sentence: This trip was the best experience of my life",
        # },
    ]

    start = time.time()
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    print(input_text)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=False,
        generation_config=GenerationConfig(use_cache=True),
    )
    print(tokenizer.decode(outputs[:, inputs.shape[1] :][0]))

    print("base", time.time() - start)

    g_model = Transformers(
        model=model, tokenizer=tokenizer, echo=False, chat_template=ChatMLTemplate
    )

    lm = g_model

    for message in messages:
        with role(role_name=message["role"]):
            lm += message["content"]

    with assistant():
        lm += (
            "Rational: Lets think step by step, "
            + gen(max_tokens=100, stop=[".", "so the"], name="rational")
            + " so the answer is: "
            + select(classes, name="answer")
        )

    print(lm)
    print(lm["rational"])
    print(lm["answer"])

    print("constrained", time.time() - start)
