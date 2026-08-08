from __future__ import annotations

from pathlib import Path

from agent.__main__ import STAGES, _missing_outputs, _outputs_exist


def test_outputs_exist_requires_every_artifact(tmp_path: Path) -> None:
  work_dir = tmp_path / "work"
  work_dir.mkdir()
  (work_dir / "04b_parties.json").write_text("{}", encoding="utf-8")
  (work_dir / "04c_adjustments.json").write_text("{}", encoding="utf-8")

  s4_extract = next(spec for spec in STAGES if spec.name == "s4_extract")
  assert not _outputs_exist(work_dir, s4_extract.outputs)
  assert _missing_outputs(work_dir, s4_extract.outputs) == ["04a_covenants.json"]

  (work_dir / "04a_covenants.json").write_text("{}", encoding="utf-8")
  assert _outputs_exist(work_dir, s4_extract.outputs)


def test_s5_requires_parquet_and_json(tmp_path: Path) -> None:
  work_dir = tmp_path / "work"
  work_dir.mkdir()
  (work_dir / "05_ledger.parquet").write_text("x", encoding="utf-8")

  s5 = next(spec for spec in STAGES if spec.name == "s5_ledger")
  assert not _outputs_exist(work_dir, s5.outputs)
  assert _missing_outputs(work_dir, s5.outputs) == ["05_ledger.json"]
