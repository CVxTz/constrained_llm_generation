import time

from guidance import assistant, gen, role, select
from guidance.models import LlamaCpp
from llama_cpp import Llama

checkpoint = "microsoft/Phi-3-mini-4k-instruct-gguf"

model = Llama.from_pretrained(
    repo_id=checkpoint, n_gpu_layers=-1, filename="*q4.gguf", verbose=False
)
g_model = LlamaCpp(model=model, echo=False)


if __name__ == "__main__":
    classes = ["positive", "negative", "neutral"]
    classes_ = ", ".join(classes)
    messages = [
        {
            "role": "user",
            "content": f"Your role is to classify the input sentence into {classes_} classes. "
            f"Answer with one of {classes_} values.",
        },
        {"role": "user", "content": "I watched the movie"},
    ]

    start = time.time()

    outputs = model.create_chat_completion(messages=messages, max_tokens=120)

    print(outputs["choices"][0]["message"]["content"])

    print("base", time.time() - start)

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
