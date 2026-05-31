"""Tests for JSON extraction utilities."""

import pytest

from mathcoach.utils.json_parser import extract_json


def test_extract_json_from_plain_object() -> None:
    payload = extract_json('{"method": "导数法", "steps": ["求导"]}')
    assert payload["method"] == "导数法"
    assert payload["steps"] == ["求导"]


def test_extract_json_from_markdown_fence() -> None:
    text = """Here is the result:
```json
{"problem_type": "函数最值问题", "goal": "求最值"}
```
"""
    payload = extract_json(text)
    assert payload["problem_type"] == "函数最值问题"
    assert payload["goal"] == "求最值"


def test_extract_json_from_embedded_prose() -> None:
    text = 'Analysis complete: {"difficulty": "中等", "goal": "求根"} Done.'
    payload = extract_json(text)
    assert payload["difficulty"] == "中等"
    assert payload["goal"] == "求根"


def test_extract_json_nested_object() -> None:
    text = '{"conditions": {"function": "x^2", "interval": "[0,1]"}, "goal": "max"}'
    payload = extract_json(text)
    assert payload["conditions"]["function"] == "x^2"
    assert payload["goal"] == "max"


def test_extract_json_empty_text_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        extract_json("   ")


def test_extract_json_unbalanced_object_raises() -> None:
    with pytest.raises(ValueError, match="Unbalanced"):
        extract_json('{"a": {"b": 1}')
