"""Tests for rctd CLI."""

import json

from click.testing import CliRunner

from rctd.cli import main


def test_info_human():
    """rctd info prints human-readable environment info."""
    runner = CliRunner()
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0
    assert "rctd-py" in result.output
    assert "torch" in result.output.lower() or "PyTorch" in result.output


def test_info_json():
    """rctd info --json prints valid JSON with required keys."""
    runner = CliRunner()
    result = runner.invoke(main, ["info", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "rctd_version" in data
    assert "python_version" in data
    assert "torch_version" in data
    assert "cuda_available" in data
    assert isinstance(data["cuda_available"], bool)


def test_version():
    """rctd --version prints version string."""
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output.lower()
