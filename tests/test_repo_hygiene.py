from __future__ import annotations

from pathlib import Path


def test_no_dataset_artifacts_live_under_src_tree() -> None:
    """
    Enforce: src/ is code-only (no downloaded/generative dataset artifacts).
    Adjust allowlist if you intentionally keep small fixtures in src/.
    """
    src = Path("src")

    allowed_suffixes = {".py", ".pyi", ".md", ".txt"}
    ignored_suffixes = {".pyc"}
    ignored_dirs = {"__pycache__"}

    bad: list[str] = []
    for p in src.rglob("*"):
        if any(part in ignored_dirs for part in p.parts):
            continue
        if p.is_file():
            if p.suffix in ignored_suffixes:
                continue
            if p.suffix not in allowed_suffixes:
                bad.append(str(p))

    assert not bad, "Non-code files found under src/:\n" + "\n".join(bad)
