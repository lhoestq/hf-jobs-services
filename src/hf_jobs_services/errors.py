"""User-facing errors of `hf jobs-services`.

Click renders a `ClickException` as a single `Error: ...` line on stderr with a non-zero exit code,
which is what every failure of the extension should look like (never a traceback).
"""

import click


class ServicesError(click.ClickException):
    """A user error: wrong input, failed service, API rejection..."""
