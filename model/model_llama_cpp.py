import time

from llama_cpp import Llama
from guidance import assistant, gen, role, select
from guidance.models import LlamaCpp

checkpoint = "microsoft/Phi-3-mini-4k-instruct-gguf"

model = Llama.from_pretrained(
    repo_id=checkpoint,
    n_gpu_layers=-1,
    filename="*q4.gguf",
    verbose=False
)
g_model = LlamaCpp(model=model, echo=False)


if __name__ == "__main__":
    classes = ["positive", "negative", "neutral"]
    classes_ = ", ".join(classes)
    messages = [
        {
            "role": "user",
            "content": f"Your role is to classify the input sentence into {classes_} classes. "
                       f"Answer with one of {classes_} values."
            # "Respond with negative or positive or neutral "
            # "* positive: for positive emotion "
            # "* negative: for negative emotion "
            # "* neutral: for neutral or no emotion. "
            # "Provide a rational or explanation.",
        },
        # {"role": "user", "content": "I liked the movie"},
        # {
        #     "role": "assistant",
        #     "content": "Rational: Lets think step by step, 'like' reflects a positive emotion so the answer is: "
        #                "positive",
        # },
        # {"role": "user", "content": "I hated the movie"},
        # {
        #     "role": "assistant",
        #     "content": "Rational: Lets think step by step, 'hate' reflects a negative emotion so the answer is: "
        #                "negative",
        # },
        {"role": "user", "content": "I watched the movie"},
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
        # {
        #     "role": "assistant",
        #     "content": "Rational: Lets think step by step, no emotion was expressed so the answer is: neutral",
        # },
        # {
        #     "role": "user",
        #     "content": "Sentence: This trip was the best experience of my life",
        # },
    ]

    start = time.time()

    outputs = model.create_chat_completion(
      messages=messages,
      max_tokens=120
    )

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
