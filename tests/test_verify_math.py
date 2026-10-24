"""End-to-end and CLI tests for scripts/verify_math.py (Milestone 5).

Verifies:
- Exit codes: 0 for PASS, 1 for FAIL, 2 for UNSUPPORTED/error
- JSON output parsing
- Units parsing from key=val, JSON string, and file
- Expected unit matching and mismatch
"""

import json
from pathlib import Path
import pytest
from scripts.verify_math import main, parse_units_argument


def test_parse_units_argument(tmp_path):
    # Key-value comma-separated
    u1 = parse_units_argument("v=meter/second,d=meter,t=second")
    assert u1 == {"v": "meter/second", "d": "meter", "t": "second"}

    # Colon separator
    u2 = parse_units_argument("a:meter,b:second")
    assert u2 == {"a": "meter", "b": "second"}

    # JSON string
    u3 = parse_units_argument('{"x": "kg", "y": "meter"}')
    assert u3 == {"x": "kg", "y": "meter"}

    # File path
    file_path = tmp_path / "units.json"
    file_path.write_text(json.dumps({"m": "kilogram", "a": "meter / second**2"}), encoding="utf-8")
    u4 = parse_units_argument(str(file_path))
    assert u4 == {"m": "kilogram", "a": "meter / second**2"}

    # Empty
    assert parse_units_argument("") == {}


def test_verify_math_cli_pass(capsys):
    ret = main([
        "E = m * c ** 2",
        "-u", "E=joule,m=kilogram,c=meter/second",
        "--json",
    ])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "PASS"
    assert data["evidence"]["dimensionality_match"] is True


def test_verify_math_cli_fail(capsys):
    ret = main([
        "a + b",
        "-u", "a=meter,b=second",
        "--json",
    ])
    assert ret == 1
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "FAIL"
    assert "addition" in str(data["evidence"]["operation"])


def test_verify_math_cli_unsupported(capsys):
    ret = main([
        "sin(x)",
        "-u", "x=radian",
        "--json",
    ])
    assert ret == 2
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "UNSUPPORTED"
    assert "Call" in str(data["evidence"]["unsupported_syntax"])


def test_verify_math_cli_expected_unit(capsys):
    # Pass expected unit
    ret_pass = main([
        "d / t",
        "-u", "d=meter,t=second",
        "-x", "meter / second",
        "--json",
    ])
    assert ret_pass == 0

    # Mismatch expected unit
    ret_fail = main([
        "d / t",
        "-u", "d=meter,t=second",
        "-x", "kilogram",
        "--json",
    ])
    assert ret_fail == 1


def test_verify_math_cli_missing_expression(capsys):
    ret = main([])
    assert ret == 2
    captured = capsys.readouterr()
    assert "Missing required expression" in captured.err
