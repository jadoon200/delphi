"""Repository-level zero-cost and non-redistribution invariants."""

import tomllib
from pathlib import Path

from delphi.config import Settings

ROOT = Path(__file__).resolve().parent.parent
HOSTED_SDKS = {"anthropic", "cohere", "google-generativeai", "openai"}


def _dependency_names() -> set[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return {str(item).split("[")[0].split(">=")[0].lower() for item in project["dependencies"]}


def test_default_runtime_cannot_spend_money() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.llm_backend == "template"
    assert not any(name.endswith("api_key") and value for name, value in vars(settings).items())
    assert _dependency_names().isdisjoint(HOSTED_SDKS)


def test_requirements_exclude_hosted_model_sdks() -> None:
    installed = "\n".join(
        (ROOT / name).read_text().lower() for name in ("requirements.txt", "requirements-dev.txt")
    )
    for package in HOSTED_SDKS:
        assert not any(line.strip().startswith(package) for line in installed.splitlines())


def test_large_or_local_only_artifacts_are_ignored() -> None:
    ignored = set((ROOT / ".gitignore").read_text().splitlines())
    assert {".claude/", "data/", "logs/", "results/"} <= ignored
