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


def _make_deterministic(nb: nbformat.NotebookNode) -> None:
    """Strip nibclient/filled transient artifacts so a rebuild is byte-identical.

    nbformat assigns RANDOM hex cell ids at cell creation, and nbclient stamps
    iopub execution timestamps into each code cell's ``metadata.execution``
    (and, in some configs, per-output ``metadata``) during a run. Without
    normalizing both, every rebuild rewrites the committed .ipynb with a
    meaningless dirty diff, which breaks the project's "git status clean"
    definition of done and makes the README's rebuild promise dishonest.
    Only ``execution_count`` is kept: it is deterministic and meaningful.
    """
    for index, cell in enumerate(nb.cells):
        # Stable, human-readable ids instead of random hex: cell-01, cell-02, ...
        # Nbformat requires unique 1-64 char strings; source order guarantees it.
        cell["id"] = f"cell-{index:02d}"
        if cell.cell_type == "code":
            cell.metadata.pop("execution", None)
            for output in cell.outputs:
                # nbclient stamps per-output iopub timestamps in some configs.
                output.pop("metadata", None)


def build(cells: list[dict[str, str]], notebook_name: str, execute: bool = True) -> Path:
    """Render and optionally execute a notebook in notebooks/.

    cells is a list of {"type": "markdown"|"code", "source": str}. The
    executed notebook is returned so callers can inspect errors.

    Calling ``build`` with the same cells always writes the same file (see
    ``_make_deterministic``) — this is what keeps rebuilds diff-clean.
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
    _make_deterministic(nb)
    nbformat.write(nb, path)
    return path