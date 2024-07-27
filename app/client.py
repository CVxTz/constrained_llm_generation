import os

from celery import Celery

# Import the task from the worker

# Get Redis URL from environment variable or use default
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Configure Celery to use Redis as the broker and backend
app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)
# Configure the result expiration time to 1 hour (3600 seconds)
app.conf.update(result_expires=3600)
app.conf.update(
    task_routes={
        "tasks.guidance_cot_task": {"queue": "guidance"},
    },
)

task_name = "tasks.guidance_cot_task"
guidance_cot_task = app.signature(task_name)


def send_guidance_cot_task(classes: list, context: str):
    """
    Send an inference task to the worker and wait for the result.

    """
    task = guidance_cot_task.delay(classes, context)
    print(f"Task sent with ID: {task.id}")

    # Wait for the result
    result = task.get(timeout=120)
    return result


if __name__ == "__main__":
    sentence = "This trip was the best experience of my life"
    _classes = ["positive", "negative", "neutral"]

    _result = send_guidance_cot_task(classes=_classes, context=sentence)
    print(f"Task result: {_result}")
