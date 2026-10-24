#!/usr/bin/env python3
"""PhysCheck CLI Verification Script (Milestone 5).

Provides a real command-line entry point to the deterministic physics expression checker.
Calls the exact checker.py implementation used across the project.
Does not duplicate checker logic or hard-code results.

Exit Codes:
  0: PASS (Expression is dimensionally consistent under supplied units)
  1: FAIL (Dimensional mismatch or physical constraint violation)
  2: UNSUPPORTED / ERROR (Unsupported syntax, unknown variables, division by zero, invalid units)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from checker import CheckResult, CheckStatus, PASS_DISCLAIMER, check_expression


def parse_units_argument(units_str: str) -> Dict[str, str]:
    """Parse a units string from JSON, a JSON file, or comma-separated pairs.

    Supported formats:
      - JSON string: '{"v": "meter / second", "d": "meter", "t": "second"}'
      - File path: 'units.json'
      - Key-value pairs: 'v=meter/second,d=meter,t=second' or 'v:meter/second,d:meter'
    """
    cleaned = units_str.strip()
    if not cleaned:
        return {}

    # Check if pointing to a readable file
    if Path(cleaned).is_file():
        with open(cleaned, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
            raise ValueError(f"File '{cleaned}' must contain a JSON object mapping variable names to units.")

    # Try JSON string
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in --units argument: {e}") from e

    # Parse key=value or key:value comma-delimited pairs
    result: Dict[str, str] = {}
    tokens = [t.strip() for t in cleaned.split(",") if t.strip()]
    for token in tokens:
        delim = "=" if "=" in token else (":" if ":" in token else None)
        if delim is None:
            raise ValueError(
                f"Invalid unit specification '{token}'. Expected 'variable=unit' or 'variable:unit'."
            )
        var, unit = token.split(delim, 1)
        var_clean = var.strip()
        unit_clean = unit.strip()
        if not var_clean or not unit_clean:
            raise ValueError(f"Malformed variable or unit in '{token}'.")
        result[var_clean] = unit_clean

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_math.py",
        description="Verify physical expressions and equations for dimensional consistency using Pint.",
    )
    parser.add_argument(
        "expression",
        nargs="?",
        default=None,
        help="Physical expression or equation string (e.g. 'E = m * c ** 2' or 'v = d / t')",
    )
    parser.add_argument(
        "-e", "--expression",
        dest="expr_flag",
        help="Alternative flag to specify expression.",
    )
    parser.add_argument(
        "-u", "--units",
        default="",
        help="Variable units as JSON string ('{\"v\": \"m/s\"}'), key=val ('v=m/s,d=m,t=s'), or JSON file.",
    )
    parser.add_argument(
        "-x", "--expected",
        default=None,
        help="Optional expected unit for single expression evaluation (e.g. 'meter / second').",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured result in JSON format.",
    )

    args = parser.parse_args(argv)

    expression = args.expr_flag or args.expression
    if not expression or not expression.strip():
        sys.stderr.write("Error: Missing required expression. Pass as positional argument or with -e.\n")
        return 2

    # Parse units
    try:
        units_dict = parse_units_argument(args.units) if args.units else {}
    except Exception as e:
        sys.stderr.write(f"Error parsing units: {e}\n")
        return 2

    # Execute deterministic check using the single source of truth in checker.py
    result: CheckResult = check_expression(
        expression=expression.strip(),
        variable_units=units_dict,
        expected_unit=args.expected,
    )

    # Format output
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        # Human-readable output
        status_symbol = {
            CheckStatus.PASS: "[PASS]",
            CheckStatus.FAIL: "[FAIL]",
            CheckStatus.UNSUPPORTED: "[UNSUPPORTED]",
        }.get(result.status, f"[{result.status}]")

        print(f"Status:          {status_symbol} {result.status.value}")
        print(f"Check Performed: {result.check_performed}")
        if result.message:
            print(f"Message:         {result.message}")
        print(f"Evidence:        {json.dumps(result.evidence)}")
        print(f"Disclaimer:      {result.assumptions.get('disclaimer', PASS_DISCLAIMER)}")

    # Return appropriate exit code
    if result.status == CheckStatus.PASS:
        return 0
    elif result.status == CheckStatus.FAIL:
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
