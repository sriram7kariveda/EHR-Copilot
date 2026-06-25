"""Jinja2-based prompt template engine.

Templates are loaded from the ``src/ehr_copilot/agents/prompts/`` directory by
default.  The engine supports Jinja2 template inheritance, includes, and macro
imports out of the box.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, BaseLoader

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE_DIR = (
    Path(__file__).resolve().parent.parent / "agents" / "prompts"
)


class PromptEngine:
    """Render prompt templates using Jinja2."""

    def __init__(self, template_dir: str | Path | None = None) -> None:
        resolved_dir = Path(template_dir) if template_dir else _DEFAULT_TEMPLATE_DIR
        resolved_dir.mkdir(parents=True, exist_ok=True)

        self._env = Environment(
            loader=FileSystemLoader(str(resolved_dir)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )
        self._string_env = Environment(
            loader=BaseLoader(),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )
        logger.debug("PromptEngine initialised with template dir: %s", resolved_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, template_name: str, **kwargs: object) -> str:
        """Load a file-based template by *name* and render it with *kwargs*.

        Parameters
        ----------
        template_name:
            Filename (or relative path) inside the templates directory,
            e.g. ``"router.j2"`` or ``"reasoning/main.j2"``.
        **kwargs:
            Variables passed into the template context.

        Returns
        -------
        str
            The rendered template string.
        """
        template = self._env.get_template(template_name)
        return template.render(**kwargs)

    def render_string(self, template_str: str, **kwargs: object) -> str:
        """Render an inline Jinja2 template string.

        Useful for one-off templates that are not stored on disk.

        Parameters
        ----------
        template_str:
            Raw Jinja2 template text.
        **kwargs:
            Variables passed into the template context.

        Returns
        -------
        str
            The rendered string.
        """
        template = self._string_env.from_string(template_str)
        return template.render(**kwargs)
