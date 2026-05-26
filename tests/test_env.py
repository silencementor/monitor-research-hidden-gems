import os

import pytest

pytest.importorskip("dotenv")

from research_hidden_gems.config import load_env_files


def test_load_env_files_sets_var_from_dotenv(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("RHG_TEST_ENV_VALUE=from_dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RHG_TEST_ENV_VALUE", raising=False)

    load_env_files()

    assert os.getenv("RHG_TEST_ENV_VALUE") == "from_dotenv"


def test_real_env_takes_precedence_over_dotenv(tmp_path, monkeypatch) -> None:
    (tmp_path / ".env").write_text("RHG_TEST_ENV_VALUE=from_dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RHG_TEST_ENV_VALUE", "from_shell")

    load_env_files()

    assert os.getenv("RHG_TEST_ENV_VALUE") == "from_shell"
