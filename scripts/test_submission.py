"""CPU checks for the Colab submission workflow."""
from zipfile import ZipFile
import hashlib
import json

from export_submission import export
from build_colab import render, requirements


def test_export_keeps_evidence_and_omits_weights_and_keys(tmp_path):
    keep = (
        "data/eval/judge_summary.json", "data/pref/stats.json", "data/pref/eval.parquet",
        "models/sft-merged/config.json", "adapters/dpo/dpo_metrics.json",
        "submission/screenshots/03-dpo-reward-curves.png", "submission/REFLECTION.md",
        "notebooks/00_dpo_loss_from_scratch.ipynb",
    )
    omit = (".env", "adapters/dpo/adapter_model.safetensors", "models/sft-merged/model.safetensors")
    for name in (*keep, *omit):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test", encoding="utf-8")
    archive = export(tmp_path, tmp_path / "lab22-submission.zip")
    with ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == set(keep)


def test_run_all_skips_every_bonus_cell_without_gpu_imports():
    nb = render("T4")
    setup = "\n".join("".join(c["source"]) for c in nb["cells"][:4])
    assert "RUN_BONUS = False" in setup
    bonus = [c for c in nb["cells"] if "bonus" in c["metadata"].get("tags", [])]
    assert bonus
    for cell in bonus:
        exec("".join(cell["source"]), {"RUN_BONUS": False})
    headings = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"]
    assert next(i for i, h in enumerate(headings) if "⏵ `notebooks/04_" in h) < next(
        i for i, h in enumerate(headings) if "⏵ `notebooks/03b_" in h
    )
    assert not any(s.startswith(("llama-cpp-python", "lm-eval")) for s in requirements())
    assert len(requirements(bonus=True)) == 2


def test_downloaded_colab_reference_requires_matching_config_hash(tmp_path, monkeypatch):
    import verify

    monkeypatch.setattr(verify, "REPO", tmp_path)
    reference = tmp_path / "models" / "sft-merged" / "config.json"
    reference.parent.mkdir(parents=True)
    reference.write_text('{"model_type": "qwen3"}', encoding="utf-8")
    adapter = tmp_path / "adapters" / "dpo"
    adapter.mkdir(parents=True)
    recorded = "/content/lab22/models/sft-merged"
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": recorded}))
    (adapter / "dpo_metrics.json").write_text(json.dumps({
        "reference_model": recorded,
        "reference_config_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "end_reward_gap": 0.1, "eval_reward_accuracy": 0.6, "diagnosis": "INTENDED",
    }))
    problems = []
    verify.check_dpo(problems, [])
    assert not problems
    reference.write_text('{"model_type": "other"}', encoding="utf-8")
    verify.check_dpo(problems, [])
    assert any("WRONG REF" in p for p in problems)
