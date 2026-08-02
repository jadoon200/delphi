from delphi.config import Settings


def test_defaults_are_explicit_and_zero_cost() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.database_url.endswith("localhost:5436/delphi")
    assert settings.snapshot_mode == "replay"
    assert settings.llm_backend == "template"


def test_environment_overrides_use_delphi_prefix(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DELPHI_SNAPSHOT_MODE", "demo")
    monkeypatch.setenv("DELPHI_RANDOM_SEED", "17")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.snapshot_mode == "demo"
    assert settings.random_seed == 17
