from client import send_guidance_cot_task
from nicegui import run, ui


class PageData:
    def __init__(
        self,
        classes: str = None,
        context: str = None,
        answer: str = None,
        rationale: str = None,
        processing: bool = False,
    ):
        self.classes = classes if classes else ""
        self.context = context
        self.answer = answer
        self.rationale = rationale
        self.processing = processing

    def reset(self):
        self.answer = ""
        self.rationale = ""

    @property
    def valid(self):
        return (
            len(self.context) >= 10
            and len([a for a in self.classes.split(",") if a.strip()]) >= 2
        )


async def handle_submit(data: PageData):
    data.processing = True
    data.reset()

    if data.valid:
        ui.notify(
            "Inference Sent. This might take a few seconds as the LLM is running on my spare toaster.",
            type="positive",
            close_button=True,
            position="top",
        )
        classes = [a.strip() for a in data.classes.split(",")]

        result = await run.io_bound(send_guidance_cot_task, classes, data.context)

        data.answer = result["answer"]
        data.rationale = result["rationale"]

        data.processing = False

        ui.notify(
            "Inference done!",
            type="positive",
            close_button=True,
            position="top",
        )
    else:
        ui.notify(
            "Invalid data",
            type="negative",
            close_button=True,
            position="top",
        )
