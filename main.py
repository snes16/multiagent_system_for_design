#!/usr/bin/env python3
import os
import click
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
from config.settings import settings
from config.logging_config import setup_logging
setup_logging()
console = Console()


@click.command()
@click.argument("prompt")
@click.option(
    "--format", "-f", "output_format",
    type=click.Choice(["html", "svg", "moodboard", "brandbook"]),
    default="html", show_default=True,
    help="Output file format",
)
@click.option(
    "--min-score", "-s",
    default=7.0, show_default=True,
    help="Minimum quality score 0-10 to accept the result",
)
def design(prompt: str, output_format: str, min_score: float):
    """
    AI Designer — multi-agent design system built on CrewAI Flow.

    \b
    Examples:
      python main.py "landing page for an architecture studio"
      python main.py "brand identity for a coffee shop" --format brandbook
      python main.py "website for a photographer" --format html --min-score 8
      python main.py "branding for an IT startup" --format moodboard
    """
    # min_score CLI override is passed via env for backward compat
    os.environ["MIN_QUALITY_SCORE"] = str(min_score)
    os.makedirs(settings.output_dir, exist_ok=True)

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
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    design()