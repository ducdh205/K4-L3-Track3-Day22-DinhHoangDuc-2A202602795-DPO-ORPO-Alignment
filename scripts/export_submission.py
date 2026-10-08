"""Zip lab evidence and source files, without weights or secrets."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

REPO = Path(__file__).resolve().parent.parent


def evidence_files(root: Path) -> list[Path]:
    patterns = (
        "lab22/*.py", "notebooks/*.py", "notebooks/*.ipynb", "scripts/*.py",
        "submission/*.md", "submission/screenshots/*.png",
        "data/eval/*.json", "data/eval/*.jsonl", "data/pref/*.parquet", "data/pref/stats.json",
        "adapters/*/adapter_config.json", "adapters/*/dpo_metrics.json", "adapters/*/split.json",
        "adapters/*/sft_metrics.json",
        "adapters/*/variants_summary.json", "adapters/*/grpo_metrics.json",
        "models/sft-merged/config.json",
    )
    files = {p for pattern in patterns for p in root.glob(pattern) if p.is_file()}
    for name in ("README.md", "rubric.md", "Makefile", "requirements.txt", "pyproject.toml", ".gitignore"):
        path = root / name
        if path.is_file():
            files.add(path)
    return sorted(files)


def export(root: Path, output: Path) -> Path:
    files = evidence_files(root)
    if not files:
        raise RuntimeError(f"No lab evidence or source files found in {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    print(f"Saved {len(files)} files to {output} ({output.stat().st_size / 1e6:.2f} MB)")
    print("Also download the executed Colab notebook with outputs via File → Download → .ipynb.")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "lab22-submission.zip")
    args = parser.parse_args()
    export(REPO, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
