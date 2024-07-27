import functools
import time
from datetime import datetime, timedelta
from typing import Union

import guidance
from guidance import Tool, assistant, gen, role, select
from model_llama_cpp import g_model, model


def classify_freeform(classes: Union[list, tuple], context: str) -> dict:
    assert len(classes) > 0

    classes_ = ", ".join(classes)
    messages = [
        {
            "role": "user",
            "content": f"Your role is to classify the input sentence into {classes_} classes. "
            f"Answer with one of {classes_} values.",
        },
        {"role": "user", "content": context},
    ]

    outputs = model.create_chat_completion(messages=messages, max_tokens=120)

    return {"answer": outputs["choices"][0]["message"]["content"]}


def classify_guided(classes: Union[list, tuple], context: str) -> dict:
    """
    Classifies a given context string into one of the provided classes.

    Args:
        classes (list): A list of possible classes to classify the context into.
        context (str): The input text to be classified.

    Returns:
        dict: A dictionary containing the classification result.
    """

    # Assert that there is at least one class provided
    assert len(classes) > 0

    # Join the classes into a comma-separated string for readability in the prompt
    classes_ = ", ".join(classes)

    # Create a list of messages for the language model.
    # The first message defines the task and the possible answers.
    # The second message provides the input context.
    messages = [
        {
            "role": "user",
            "content": f"Your role is to classify the input sentence into {classes_} classes. "
            f"Answer with one of {classes_} values.",
        },
        {"role": "user", "content": context},
    ]

    # Assign the language model to the variable 'lm'
    lm = g_model  # Assuming 'g_model' is a pre-defined language model

    # Iterate through the messages and add them to the language model.
    # This essentially creates the context for the language model.
    for message in messages:
        # Use the 'role' context manager to define the role of the message
        with role(role_name=message["role"]):
            lm += message["content"]

    # Add the prompt for the language model to generate an answer from the provided classes
    with assistant():
        lm += " Answer: " + select(classes, name="answer")

    # Return the classification result as a dictionary
    return {"answer": lm["answer"]}


@functools.lru_cache(maxsize=1024)
def classify_cot(classes: Union[list, tuple], context: str) -> dict:
    assert len(classes) > 0

    classes_ = ", ".join(classes)
    messages = [
        {
            "role": "user",
            "content": f"Your role is to classify the input sentence into {classes_} classes. "
            f"Answer with one of {classes_} values.",
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
            + gen(max_tokens=100, stop=[".", "so the"], name="rationale")
            + " so the answer is: "
            + select(classes, name="answer")
        )

    return {"answer": lm["answer"], "rationale": lm["rationale"]}


def question_cot(question: str) -> dict:
    messages = [
        {"role": "user", "content": "Your role is to answer the following question."},
        {"role": "user", "content": f"Question: {question}"},
    ]

    lm = g_model

    for message in messages:
        with role(role_name=message["role"]):
            lm += message["content"]

    with assistant():
        lm += (
            "Lets think step by step, "
            + gen(
                max_tokens=100, stop=[".", "so the"], name="rationale", temperature=0.0
            )
            + " so the answer is: "
            + gen(max_tokens=10, stop=["."], name="answer")
        )

    return {"answer": lm["answer"], "rationale": lm["rationale"]}


def extract_date_address(context: str) -> dict:
    messages = [
        {
            "role": "user",
            "content": "Your role is to extract the date in YYYY/MM/DD format and address. If any of those information"
            " is not found, respond with Not found",
        },
        {"role": "user", "content": f"Context: {context}"},
    ]

    lm = g_model

    regex = "\d{4}/\d{2}/\d{2}"

    for message in messages:
        with role(role_name=message["role"]):
            lm += message["content"]

    with assistant():
        lm += f"""\
        ```json
        {{
            "date": "{select(options=[gen(regex=regex, stop='"'), "Not found"], name="date")}",
            "address": "{select(options=[gen(stop='"'), "Not found"], name="address")}"
        }}```"""

    return {"date": lm["date"], "address": lm["address"]}


@guidance
def reverse_string(lm, string: str):
    lm += " = " + string[::-1]
    return lm.set("answer", string[::-1])


@guidance
def get_date(lm, delta):
    delta = int(delta)
    date = (datetime.today() + timedelta(days=delta)).strftime("%Y-%m-%d")
    lm += " = " + date
    return lm.set("answer", date)


reverse_string_tool = Tool(callable=reverse_string)
date_tool = Tool(callable=get_date)


def tool_use(question):
    messages = [
        {
            "role": "user",
            "content": """You are tasked with answering user's questions.
            You have access to two tools:
            reverse_string which can be used like reverse_string("thg") = "ght"
            get_date which can be used like get_date(delta=x) = "YYYY-MM-DD""",
        },
        {"role": "user", "content": "What is today's date?"},
        {
            "role": "assistant",
            "content": """delta from today is 0 so get_date(delta=0) = "YYYY-MM-DD" so the answer is: YYYY-MM-DD""",
        },
        {"role": "user", "content": "What is yesterday's date?"},
        {
            "role": "assistant",
            "content": """delta from today is -1 so get_date(delta=-1) = "YYYY-MM-XX" so the answer is: YYYY-MM-XX""",
        },
        {"role": "user", "content": "can you reverse this string: Roe Jogan ?"},
        {
            "role": "assistant",
            "content": "reverse_string(Roe Jogan) = nagoJ eoR so the answer is: nagoJ eoR",
        },
        {"role": "user", "content": f"{question}"},
    ]

    lm = g_model

    for message in messages:
        with role(role_name=message["role"]):
            lm += message["content"]

    with assistant():
        lm = (
            lm
            + gen(
                max_tokens=50,
                stop=["."],
                tools=[reverse_string_tool, date_tool],
                temperature=0.0,
            )
            + " so the answer is: "
            + gen(
                max_tokens=50, stop=[".", "\n"], tools=[reverse_string_tool, date_tool]
            )
        )

    print(lm)

    return {"answer": lm["answer"]}


if __name__ == "__main__":
    sentence = "This trip was the best experience of my life"
    _classes = ["positive", "negative", "neutral"]

    for func in [classify_guided, classify_freeform, classify_cot]:
        start = time.time()

        result = func(classes=tuple(_classes), context=sentence)

        print(f"{func=}")
        print(f"{result}")

        print("processing time:", time.time() - start)

    _question = (
        "If you had ten apples and then you gave away half, "
        "how many would you have left? Answer with only digits"
    )

    start = time.time()

    result = question_cot(question=_question)

    print(f"{func=}")
    print(f"{result}")

    print("processing time:", time.time() - start)

    _context = " 14/08/2025 14, rue Delambre  75014 Paris "

    result = extract_date_address(context=_context)

    print(result)

    # _question = " What is the date 4512 days in the future from now? "

    _question = "Can you reverse this string: generative AI applications ?"

    result = tool_use(question=_question)

    print(result)
