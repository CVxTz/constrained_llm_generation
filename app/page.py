from nicegui import ui
from utils import PageData, handle_submit


def home():
    classes = "positive, negative, neutral"
    context = "This trip was the best experience of my life"
    page_data = PageData(classes=classes, context=context)

    # User Interface
    with ui.row().style(
        "justify-content: center; align-items: center; gap: 10em"
    ).classes("w-full"):
        with ui.column().style("align-items: center;").classes("w-full"):
            ui.label("Chain-of-Thought Zero-Shot Classification").classes("text-2xl")

            with ui.card().classes("relative w-8/12 overflow-hidden bg-slate-100"):
                ui.label("Context:").classes("text-xl")
                ui.input(
                    validation={"Too short": lambda value: len(value) >= 10}
                ).props("outlined dense maxlength=300").classes("w-full").bind_value(
                    page_data, "context"
                )
                ui.label("Comma-separated classes:").classes("text-xl")
                ui.input(
                    validation={
                        "You need at least two comma separated classes": lambda value: len(
                            [a for a in value.split(",") if a.strip()]
                        )
                        >= 2
                    }
                ).props("outlined dense").classes("w-full").bind_value(
                    page_data, "classes"
                )

                ui.button(
                    "Submit", on_click=lambda _: handle_submit(data=page_data)
                ).bind_enabled_from(page_data, "valid")

            with ui.card().classes(
                "relative w-8/12 overflow-hidden bg-slate-100"
            ).bind_visibility_from(page_data, "processing"):
                ui.skeleton().classes("w-full")

            with ui.card().classes().classes(
                "w-8/12 bg-slate-100"
            ).bind_visibility_from(page_data, "answer"):
                ui.label("Class prediction:").classes("text-lg")
                ui.label().bind_text_from(page_data, "answer").classes("font-bold")
                ui.label("Rationale:").classes("text-lg")
                ui.label().bind_text_from(page_data, "rationale").classes("font-bold")

    with ui.footer().classes("bg-slate-100 rounded-lg shadow m-4 dark:bg-gray-800"):
        with ui.row().classes(
            "w-full justify-center items-center gap-10 text-white font-bold"
        ):
            ui.link(
                "Linkedin", "https://www.linkedin.com/in/mansar/", new_tab=True
            ).classes("text-black no-underline")
            ui.link(
                "Source Code",
                "https://github.com/CVxTz/constrained_llm_generation",
                new_tab=True,
            ).classes("text-black no-underline")
