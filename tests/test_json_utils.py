#!/usr/bin/env python3
"""Tests for JSON utilities used in training data generation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.training.json_utils import extract_json_from_response, validate_training_sample_json


def test_extract_json_plain():
    response = '{"instruction":"do it","input":"ctx","output":"code"}'
    data = extract_json_from_response(response)
    assert data == {"instruction": "do it", "input": "ctx", "output": "code"}


def test_extract_json_code_block():
    response = """```json
{
  \"instruction\": \"do it\",
  \"input\": \"ctx\",
  \"output\": \"code\"
}
```"""
    data = extract_json_from_response(response)
    assert data == {"instruction": "do it", "input": "ctx", "output": "code"}


def test_extract_json_unescaped_newline():
    response = '{"instruction":"do it","input":"ctx","output":"line1\nline2"}'
    data = extract_json_from_response(response)
    assert data["output"] == "line1\nline2"


def test_validate_training_sample_json():
    valid = {"instruction": "do it", "input": "ctx", "output": "code"}
    assert validate_training_sample_json(valid) is True

    missing = {"instruction": "do it", "output": "code"}
    assert validate_training_sample_json(missing) is False

    wrong_type = {"instruction": "do it", "input": "ctx", "output": 42}
    assert validate_training_sample_json(wrong_type) is False
