"""Comprehensive tests for PhysCheck deterministic expression checker (Milestone 2).

Verifies:
- Compatible and incompatible addition/subtraction
- Multiplication and division
- Restricted powers (integer, negative, fractional, dimensioned exponents, out-of-bounds)
- Unknown variables
- Invalid units
- Unsupported syntax and operators
- Division by zero
- Uploaded/pasted code is never executed
- Result structure, assumptions, and required disclaimer
"""

import os
import pytest
from checker import (
    CheckResult,
    CheckStatus,
    DeterministicChecker,
    PASS_DISCLAIMER,
    check_equation,
    check_expression,
)


# 1. Compatible Addition / Subtraction
def test_compatible_addition():
    # Same units
    res = check_expression("a + b", variable_units={"a": "meter", "b": "meter"})
    assert res.status == CheckStatus.PASS
    assert res.status == "PASS"
    assert res.is_pass
    assert "length" in str(res.evidence.get("inferred_dimensionality"))
    assert res.assumptions["disclaimer"] == PASS_DISCLAIMER

    # Convertible / compatible units (meter + kilometer)
    res_km = check_expression("a + b", variable_units={"a": "meter", "b": "kilometer"})
    assert res_km.status == CheckStatus.PASS

    # Compatible subtraction
    res_sub = check_expression("t1 - t2", variable_units={"t1": "minute", "t2": "second"})
    assert res_sub.status == CheckStatus.PASS

    # Dimensionless addition
    res_dimless = check_expression("1 + 2.5")
    assert res_dimless.status == CheckStatus.PASS


# 2. Incompatible Addition / Subtraction
def test_incompatible_addition():
    # Adding meter and second
    res = check_expression("a + b", variable_units={"a": "meter", "b": "second"})
    assert res.status == CheckStatus.FAIL
    assert res.status == "FAIL"
    assert res.is_fail
    assert "addition" in str(res.evidence.get("operation"))

    # Subtracting kilogram from meter
    res_sub = check_expression("d - m", variable_units={"d": "meter", "m": "kg"})
    assert res_sub.status == CheckStatus.FAIL
    assert res_sub.is_fail

    # Adding dimensioned quantity to dimensionless constant
    res_const = check_expression("x + 5", variable_units={"x": "meter"})
    assert res_const.status == CheckStatus.FAIL

    # Incompatible addition inside an equation
    res_eq = check_expression(
        "v = d + t",
        variable_units={"v": "meter/second", "d": "meter", "t": "second"},
    )
    assert res_eq.status == CheckStatus.FAIL


# 3. Multiplication
def test_multiplication():
    # F = m * a
    res = check_expression(
        "m * a",
        variable_units={"m": "kilogram", "a": "meter / second**2"},
    )
    assert res.status == CheckStatus.PASS
    assert "[mass]" in str(res.evidence.get("inferred_dimensionality"))
    assert "[length]" in str(res.evidence.get("inferred_dimensionality"))

    # Scalar multiplication: 2 * pi_r
    res_scalar = check_expression("2 * r", variable_units={"r": "meter"})
    assert res_scalar.status == CheckStatus.PASS
    assert "[length]" in str(res.evidence.get("inferred_dimensionality"))


# 4. Division
def test_division():
    # v = d / t
    res = check_expression("d / t", variable_units={"d": "meter", "t": "second"})
    assert res.status == CheckStatus.PASS

    # Matching expected unit
    res_exp = check_expression(
        "d / t",
        variable_units={"d": "meter", "t": "second"},
        expected_unit="meter / second",
    )
    assert res_exp.status == CheckStatus.PASS

    # Mismatch with expected unit
    res_mismatch = check_expression(
        "d / t",
        variable_units={"d": "meter", "t": "second"},
        expected_unit="second",
    )
    assert res_mismatch.status == CheckStatus.FAIL
    assert res_mismatch.evidence.get("dimensionality_match") is False


# 5. Restricted Powers
def test_powers():
    # Integer power
    res = check_expression("x ** 2", variable_units={"x": "meter"})
    assert res.status == CheckStatus.PASS
    assert "[length] ** 2" in str(res.evidence.get("inferred_dimensionality"))

    # Negative power
    res_neg = check_expression("t ** -1", variable_units={"t": "second"})
    assert res_neg.status == CheckStatus.PASS
    assert "/ [time]" in str(res_neg.evidence.get("inferred_dimensionality"))

    # Fractional power of squared unit
    res_frac = check_expression("area ** 0.5", variable_units={"area": "meter**2"})
    assert res_frac.status == CheckStatus.PASS

    # Exponent with dimensions must fail
    res_dim_exp = check_expression(
        "x ** y",
        variable_units={"x": "meter", "y": "second"},
    )
    assert res_dim_exp.status == CheckStatus.FAIL
    assert "exponent" in str(res_dim_exp.message).lower() or "power" in str(res_dim_exp.message).lower()

    # Exponent exceeding safety bound |exp| > 100 must be rejected
    res_huge = check_expression("x ** 101", variable_units={"x": "meter"})
    assert res_huge.status == CheckStatus.UNSUPPORTED


# 6. Unknown Variables
def test_unknown_variables():
    # Single unknown variable
    res = check_expression("x + y", variable_units={"x": "meter"})
    assert res.status == CheckStatus.UNSUPPORTED
    assert res.is_unsupported
    assert res.evidence.get("unknown_variable") == "y"

    # In equation target
    res_eq = check_expression("z = x", variable_units={"x": "meter"})
    assert res_eq.status == CheckStatus.UNSUPPORTED
    assert res_eq.evidence.get("unknown_variable") == "z"


# 7. Invalid Units
def test_invalid_units():
    # Nonexistent unit
    res = check_expression("x + 1", variable_units={"x": "nonexistent_unit_xyz_999"})
    assert res.status == CheckStatus.UNSUPPORTED
    assert "Invalid unit" in res.message or "nonexistent_unit" in str(res.evidence)

    # Empty unit
    res_empty = check_expression("x", variable_units={"x": ""})
    assert res_empty.status == CheckStatus.UNSUPPORTED

    # Invalid expected unit
    res_exp = check_expression("x", variable_units={"x": "meter"}, expected_unit="bogus_unit_xyz")
    assert res_exp.status == CheckStatus.UNSUPPORTED


# 8. Unsupported Syntax
def test_unsupported_syntax():
    # Function calls
    res_fn = check_expression("sin(x)", variable_units={"x": "radian"})
    assert res_fn.status == CheckStatus.UNSUPPORTED

    # Bitwise operators
    res_bit = check_expression("a & b", variable_units={"a": "meter", "b": "meter"})
    assert res_bit.status == CheckStatus.UNSUPPORTED

    # Boolean operators and literals
    res_bool = check_expression("a and b", variable_units={"a": "meter", "b": "meter"})
    assert res_bool.status == CheckStatus.UNSUPPORTED

    res_bool_lit = check_expression("True")
    assert res_bool_lit.status == CheckStatus.UNSUPPORTED

    # String literals
    res_str = check_expression("'hello' + x", variable_units={"x": "meter"})
    assert res_str.status == CheckStatus.UNSUPPORTED

    # Lists, dicts, tuples
    res_list = check_expression("[1, 2, 3]")
    assert res_list.status == CheckStatus.UNSUPPORTED

    # Attribute access
    res_attr = check_expression("x.__class__", variable_units={"x": "meter"})
    assert res_attr.status == CheckStatus.UNSUPPORTED

    # Subscripts
    res_sub = check_expression("x[0]", variable_units={"x": "meter"})
    assert res_sub.status == CheckStatus.UNSUPPORTED

    # Syntax errors
    res_syn = check_expression("a + * b", variable_units={"a": "meter", "b": "meter"})
    assert res_syn.status == CheckStatus.UNSUPPORTED


# 9. Division by Zero
def test_division_by_zero():
    # Literal division by zero
    res = check_expression("x / 0", variable_units={"x": "meter"})
    assert res.status == CheckStatus.UNSUPPORTED
    assert "zero" in res.message.lower()

    # Divisor evaluating to zero
    res_expr_zero = check_expression("x / (5 - 5)", variable_units={"x": "meter"})
    assert res_expr_zero.status == CheckStatus.UNSUPPORTED

    # Zero base with negative exponent
    res_pow_zero = check_expression("0 ** -2")
    assert res_pow_zero.status == CheckStatus.UNSUPPORTED


# 10. Test that uploaded / pasted code is NEVER executed
def test_uploaded_code_never_executed(tmp_path):
    sentinel = {"executed": False}

    # Helper function that sets sentinel
    def malicious_probe():
        sentinel["executed"] = True
        return 1

    # Code that would call probe if evaluated
    test_code_1 = "malicious_probe()"
    res_1 = check_expression(test_code_1, variable_units={})
    assert res_1.status == CheckStatus.UNSUPPORTED
    assert sentinel["executed"] is False

    # Code attempting exec/eval
    test_code_2 = "exec('sentinel[\"executed\"] = True')"
    res_2 = check_expression(test_code_2, variable_units={})
    assert res_2.status == CheckStatus.UNSUPPORTED
    assert sentinel["executed"] is False

    # Code attempting file system side effects
    target_file = tmp_path / "pwned.txt"
    test_code_3 = f"open(r'{target_file}', 'w').write('compromised')"
    res_3 = check_expression(test_code_3, variable_units={})
    assert res_3.status == CheckStatus.UNSUPPORTED
    assert not target_file.exists()

    # Code attempting imports
    test_code_4 = "__import__('os').system('echo probe')"
    res_4 = check_expression(test_code_4, variable_units={})
    assert res_4.status == CheckStatus.UNSUPPORTED

    # Multiline script
    multiline_script = """
import os
def attack():
    pass
attack()
"""
    res_5 = check_expression(multiline_script, variable_units={})
    assert res_5.status == CheckStatus.UNSUPPORTED


# 11. Equations and Class Interface
def test_equations_and_checker_class():
    checker = DeterministicChecker()

    # Valid equation: E = m * c ** 2
    res_energy = checker.check(
        "E = m * c ** 2",
        variable_units={
            "E": "joule",
            "m": "kilogram",
            "c": "meter / second",
        },
    )
    assert res_energy.status == CheckStatus.PASS

    # Mismatched equation: E = m * c
    res_mismatch = checker.check(
        "E = m * c",
        variable_units={
            "E": "joule",
            "m": "kilogram",
            "c": "meter / second",
        },
    )
    assert res_mismatch.status == CheckStatus.FAIL

    # check_equation helper function
    res_eq = check_equation(
        "v",
        "d / t",
        variable_units={"v": "meter / second", "d": "meter", "t": "second"},
    )
    assert res_eq.status == CheckStatus.PASS


# 12. Result Structure and Required Disclaimer
def test_result_structure_and_disclaimer():
    res = check_expression(
        "a + b",
        variable_units={"a": "meter", "b": "meter"},
    )
    # Structured fields
    assert hasattr(res, "status")
    assert hasattr(res, "check_performed")
    assert hasattr(res, "evidence")
    assert hasattr(res, "assumptions")
    assert hasattr(res, "message")

    # Disclaimer check
    assert "disclaimer" in res.assumptions
    assert res.assumptions["disclaimer"] == PASS_DISCLAIMER
    assert (
        "PASS means only that the supported check passed under the supplied units; "
        "it does not mean the whole scientific program is correct."
    ) in res.assumptions["disclaimer"]

    # Dict-like and to_dict access
    assert res["status"] == "PASS"
    as_dict = res.to_dict()
    assert isinstance(as_dict, dict)
    assert as_dict["status"] == "PASS"
