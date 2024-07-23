from guidance import assistant, gen, role, select
import time

from model_llama_cpp import model, g_model


def classify_freeform(classes: list, context: str) -> dict:

    assert len(classes) > 0

    classes_ = ", ".join(classes)
    messages = [
        {
            "role": "user",
            "content": f"Your role is to classify the input sentence into {classes_} classes. "
                       f"Answer with one of {classes_} values."
        },
        {"role": "user", "content": context},
    ]

    outputs = model.create_chat_completion(
      messages=messages,
      max_tokens=120
    )

    return {"answer": outputs["choices"][0]["message"]["content"]}


def classify_guided(classes: list, context: str) -> dict:

    assert len(classes) > 0

    classes_ = ", ".join(classes)
    messages = [
        {
            "role": "user",
            "content": f"Your role is to classify the input sentence into {classes_} classes. "
                       f"Answer with one of {classes_} values."
        },
        {"role": "user", "content": context},
    ]

    lm = g_model

    for message in messages:
        with role(role_name=message["role"]):
            lm += message["content"]

    with assistant():
        lm += " Answer: " + select(classes, name="answer")

    return {"answer": lm["answer"]}


def classify_cot(classes: list, context: str) -> dict:

    assert len(classes) > 0

    classes_ = ", ".join(classes)
    messages = [
        {
            "role": "user",
            "content": f"Your role is to classify the input sentence into {classes_} classes. "
                       f"Answer with one of {classes_} values."
        },
        {"role": "user", "content": context},
    ]

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

    return {"answer": lm["answer"], "rational": lm["rational"]}


if __name__ == "__main__":

    sentence = "This trip was the best experience of my life"
    _classes = ["positive", "negative", "neutral"]

    for func in [classify_freeform, classify_guided, classify_cot]:

        start = time.time()

        result = func(classes=_classes, context=sentence)

        print(f"{func=}")
        print(f"{result}")

        print("processing time:", time.time() - start)
