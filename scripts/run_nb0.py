"""Execute NB0 on CPU and save its real printed outputs in a notebook."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

from build_colab import percent_cells

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    source = REPO / "notebooks" / "00_dpo_loss_from_scratch.py"
    cells = percent_cells(source)
    namespace = {"__name__": "__main__", "__file__": str(source)}
    execution = 0
    for i, cell in enumerate(cells):
        cell["id"] = f"nb0-{i:03d}"
        if cell["cell_type"] != "code":
            continue
        execution += 1
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile("".join(cell["source"]), str(source), "exec"), namespace)
        cell["execution_count"] = execution
        for name, stream in (("stdout", stdout), ("stderr", stderr)):
            if stream.getvalue():
                cell["outputs"].append({"output_type": "stream", "name": name, "text": stream.getvalue().splitlines(True)})
                print(stream.getvalue(), end="")
    notebook = {
        "cells": cells, "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    }
    output = source.with_suffix(".ipynb")
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Saved executed NB0: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
