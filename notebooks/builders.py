"""Reproducible notebook builders.

Notebooks are code artifacts and rot like any other code when edited by hand.
Each stage declares its cells here; the builder executes them with nbclient
so the committed .ipynb always shows real outputs synced with src/.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def build(cells: list[dict[str, str]], notebook_name: str, execute: bool = True) -> Path:
    """Render and optionally execute a notebook in notebooks/.

    cells is a list of {"type": "markdown"|"code", "source": str}. The
    executed notebook is returned so callers can inspect errors.
    """
    nb = nbformat.v4.new_notebook()
    nb["cells"] = [
        nbformat.v4.new_markdown_cell(c["source"]) if c["type"] == "markdown"
        else nbformat.v4.new_code_cell(c["source"])
        for c in cells
    ]
    nb["metadata"].update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        }
    )
    path = ROOT / "notebooks" / f"{notebook_name}.ipynb"
    if execute:
        # Run with the kernel cwd set to notebooks/ so relative paths inside
        # cells behave exactly like they do when a human opens the notebook.
        NotebookClient(nb, timeout=300, kernel_name="python3", cwd=str(ROOT / "notebooks")).execute()  # type: ignore[attr-defined]
    nbformat.write(nb, path)
    return path