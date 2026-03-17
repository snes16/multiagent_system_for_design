#!/usr/bin/env python3
import os
import click
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()


@click.command()
@click.argument("prompt")
@click.option(
    "--format", "-f", "output_format",
    type=click.Choice(["html", "svg", "moodboard", "brandbook"]),
    default="html", show_default=True,
    help="Формат выходного файла",
)
@click.option(
    "--min-score", "-s",
    default=7.0, show_default=True,
    help="Минимальный балл качества 0-10 для принятия результата",
)
def design(prompt: str, output_format: str, min_score: float):
    """
    AI Designer — мультиагентная дизайн-система на CrewAI Flow.

    \b
    Примеры:
      python main.py "лэндинг для архитектурного бюро"
      python main.py "фирменный стиль кофейни" --format brandbook
      python main.py "сайт для фотографа" --format html --min-score 8
      python main.py "брендинг IT стартапа" --format moodboard
    """
    os.environ["MIN_QUALITY_SCORE"] = str(min_score)
    os.makedirs(os.getenv("OUTPUT_DIR", "./output"), exist_ok=True)

    from flow import DesignerFlow
    from models.state import DesignerState, OutputFormat

    state = DesignerState(
        prompt=prompt,
        output_format=OutputFormat(output_format),
    )

    try:
        flow = DesignerFlow()
        flow.kickoff(inputs={"prompt": prompt, "output_format": output_format})
    except KeyboardInterrupt:
        console.print("\n[yellow]Прервано.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Ошибка: {e}[/red]")
        raise


if __name__ == "__main__":
    design()