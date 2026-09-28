"""Output helpers, mirroring the `hf` CLI conventions: data on stdout, commentary on stderr."""

import sys

import click


def result(message: str, **data: object) -> None:
    """Print a success summary on stdout: a green check plus one `key: value` line per item."""
    check = "\u2713" if _encodable() else "[OK]"
    parts = [_green(f"{check} {message}")]
    parts += [f"  {key}: {value}" for key, value in data.items() if value is not None]
    click.echo("\n".join(parts))


def log(message: str) -> None:
    """Print a progress line on stderr."""
    click.echo(_gray(message), file=sys.stderr)


def hint(message: str) -> None:
    """Print an actionable follow-up suggestion on stderr."""
    log(f"Hint: {message}")


def warn(message: str) -> None:
    """Print a non-fatal warning on stderr."""
    click.echo(_yellow(f"Warning: {message}"), file=sys.stderr)


def table(rows: list[dict[str, object]], columns: list[str]) -> None:
    """Print rows as a padded table on stdout."""
    if not rows:
        return
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row.get(column, ""))))
    click.echo(" ".join(column.ljust(widths[column]) for column in columns))
    click.echo(" ".join("-" * widths[column] for column in columns))
    for row in rows:
        click.echo(" ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))


def _encodable() -> bool:
    try:
        "✓".encode(sys.stdout.encoding or "utf-8")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _green(text: str) -> str:
    return click.style(text, fg="green")


def _yellow(text: str) -> str:
    return click.style(text, fg="yellow")


def _gray(text: str) -> str:
    return click.style(text, fg="white", dim=True)
