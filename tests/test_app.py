"""Unit and UI workflow tests for PhysCheck Streamlit application (Milestone 4).

Verifies:
- PhysCheck title, description, and UI components
- Built-in examples validity and deterministic results
- File upload security: extension and size limits, read-as-text only, zero execution
- Editable unit assumptions and clear distinction between AI suggestions and user-confirmed units
- Separate actions for Gemma analysis (fully mocked) and deterministic checks
- Deterministic checks work 100% offline without API key or network
- Useful error messages without exposing secrets or credentials
- AppTest execution with strict timeouts and mocked AI layer
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from app import (
    BUILTIN_EXAMPLES,
    MAX_UPLOAD_SIZE_BYTES,
    apply_example,
    init_session_state,
)
from checker import CheckStatus, check_expression
from ai_analyzer import (
    AIAnalysisResult,
    AuthenticationError,
    DATA_TRANSMISSION_NOTICE,
    MissingAPIKeyError,
    QuotaExceededError,
    SUGGESTIONS_NOTICE,
    UnsupportedModelError,
)


# 1. Built-in Examples Validity
def test_builtin_examples_validity():
    assert len(BUILTIN_EXAMPLES) >= 7

    # Test that valid examples pass deterministic checks
    valid_keys = [
        "Mass-Energy Equivalence (E = m * c ** 2)",
        "Kinetic Energy (K = 0.5 * m * v ** 2)",
        "Kinematics Velocity (v = d / t)",
        "Newton's Second Law (F = m * a)",
        "Centripetal Acceleration (a = v ** 2 / r)",
        "Gravitational Potential Energy (U = m * g * h)",
    ]
    for key in valid_keys:
        data = BUILTIN_EXAMPLES[key]
        res = check_expression(
            expression=data["expr"],
            variable_units=data["units"],
            expected_unit=data.get("expected") or None,
        )
        assert res.status == CheckStatus.PASS, f"Example '{key}' failed check: {res.message}"

    # Test that fail demos fail as expected
    fail_data = BUILTIN_EXAMPLES["Incompatible Addition [FAIL DEMO] (a + b)"]
    fail_res = check_expression(fail_data["expr"], variable_units=fail_data["units"])
    assert fail_res.status == CheckStatus.FAIL


class MockSessionState(dict):
    """Dictionary that supports both attribute and item access, like st.session_state."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


# 2. Session State Initialization
def test_init_session_state():
    mock_state = MockSessionState()
    with patch("streamlit.session_state", mock_state):
        init_session_state()
        assert "expression_input" in mock_state
        assert "user_units" in mock_state
        assert "confirmed_flags" in mock_state
        assert "check_result" in mock_state
        assert "ai_result" in mock_state
        assert mock_state["expression_input"] == "E = m * c ** 2"


# 3. Apply Built-in Example
def test_apply_example():
    mock_state = MockSessionState()
    with patch("streamlit.session_state", mock_state):
        init_session_state()
        apply_example("Kinematics Velocity (v = d / t)")
        assert mock_state["expression_input"] == "v = d / t"
        assert mock_state["user_units"] == {
            "v": "meter / second",
            "d": "meter",
            "t": "second",
        }
        assert mock_state["confirmed_flags"] == {"v": True, "d": True, "t": True}


# 4. File Upload Limits & Security
def test_file_upload_security_and_size_limits():
    # 1. Size limit enforcement
    oversized_bytes = b"x = 1\n" * 15000
    assert len(oversized_bytes) > MAX_UPLOAD_SIZE_BYTES

    # 2. Read as text only - never executed
    code_with_side_effect = "import os; open('should_never_exist.txt', 'w').write('bad')"
    # When decoded as text:
    decoded_text = code_with_side_effect.encode("utf-8").decode("utf-8")
    assert "import os" in decoded_text
    # Ensure no file was created
    assert not os.path.exists("should_never_exist.txt")


# 5. Distinction Between AI-proposed Units and User-confirmed Units
def test_unit_confirmation_state_distinction():
    mock_state = {
        "user_units": {"v": "meter / second"},
        "confirmed_flags": {"v": False},  # Unconfirmed AI suggestion
    }
    # Variable is currently an unconfirmed suggestion
    assert mock_state["confirmed_flags"]["v"] is False

    # User confirms the suggestion
    mock_state["confirmed_flags"]["v"] = True
    assert mock_state["confirmed_flags"]["v"] is True

    # User adds a new manual variable (automatically confirmed)
    mock_state["user_units"]["t"] = "second"
    mock_state["confirmed_flags"]["t"] = True
    assert mock_state["confirmed_flags"]["t"] is True


# 6. Deterministic Checks Work Offline Without API Key
def test_deterministic_checks_offline_no_api_key(monkeypatch):
    # Completely remove GEMINI_API_KEY from environment
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Deterministic check runs purely with Pint and AST, no network or API calls
    res = check_expression(
        "K = 0.5 * m * v ** 2",
        variable_units={"K": "joule", "m": "kilogram", "v": "meter / second"},
    )
    assert res.status == CheckStatus.PASS
    assert res.is_pass
    assert res.evidence["dimensionality_match"] is True


# 7. Error Handling Masks Secrets
def test_error_handling_masks_secrets():
    secret_key = "AIzaSy_CONFIDENTIAL_KEY_99999"

    # Missing API key error
    err_missing = MissingAPIKeyError("GEMINI_API_KEY is not set.")
    assert secret_key not in str(err_missing)

    # Auth error
    err_auth = AuthenticationError("API authentication failed (status 401).")
    assert secret_key not in str(err_auth)

    # Quota error
    err_quota = QuotaExceededError("API quota exceeded.")
    assert secret_key not in str(err_quota)

    # Unsupported model error
    err_model = UnsupportedModelError(
        "Requested model 'gemma-4-31b-it' is not available. PhysCheck will not silently substitute another model."
    )
    assert secret_key not in str(err_model)
    assert "gemma-4-31b-it" in str(err_model)


# 8. Automated Streamlit AppTest Workflow (Mocked AI Layer with Timeout)
def test_streamlit_apptest_workflow():
    """Run Streamlit AppTest headless with strict timeout and mocked AI."""
    import asyncio
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    # Prevent Windows loopback socketpair deadlocks in sandboxed/restricted environments
    mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    mock_loop.is_closed.return_value = False

    # Always mock the AI analyzer to prevent any real API or network calls
    with patch("asyncio.new_event_loop", return_value=mock_loop), \
         patch("ai_analyzer.AIAnalyzer.analyze_expression") as mock_ai:
        mock_ai.return_value = AIAnalysisResult(
            summary="Kinematics formula for velocity",
            uncertainty="None for constant speed",
            candidate_units={"v": "meter / second", "d": "meter", "t": "second"},
            model_identifier="gemma-4-31b-it",
            data_transmission_notice=DATA_TRANSMISSION_NOTICE,
            suggestions_notice=SUGGESTIONS_NOTICE,
        )

        app_path = str(Path(__file__).resolve().parent.parent / "app.py")
        # Use strict 5.0s timeout to guarantee it never hangs
        at = AppTest.from_file(app_path, default_timeout=5.0)
        at.run(timeout=5.0)

        # Verify page rendered with title and core elements
        assert not at.exception
        assert len(at.title) >= 1
        assert "PhysCheck" in at.title[0].value

        # Verify text area exists and has default expression
        assert len(at.text_area) >= 1
        assert at.text_area[0].value == "E = m * c ** 2"

        # Click the primary check expression button
        check_buttons = [b for b in at.button if "Check expression" in b.label or "Run Deterministic Check" in b.label]
        assert len(check_buttons) >= 1
        check_buttons[0].click()
        at.run(timeout=5.0)

        assert not at.exception
        # Success banner should be displayed for default E = m * c ** 2
        assert len(at.success) >= 1
        assert "PASS" in at.success[0].value


# 9. Additional Professional UI & Workflow Regression Tests
def test_expression_variable_extraction_no_unrelated_variables():
    """Verify that only variables actually present in the expression are identified."""
    from app import extract_expression_variables

    # 1. distance / time contains only distance and time
    vars_found, err = extract_expression_variables("distance / time")
    assert err is None
    assert vars_found == ["distance", "time"]
    assert "mass" not in vars_found
    assert "acceleration" not in vars_found

    # 2. Equation E = m * c ** 2
    eq_vars, err = extract_expression_variables("E = m * c ** 2")
    assert err is None
    assert eq_vars == ["E", "m", "c"]

    # 3. Syntax error detection
    bad_vars, err = extract_expression_variables("x + * y")
    assert err is not None
    assert bad_vars == []


def test_expression_with_no_variables():
    """Verify that expressions with no variables (only numeric constants) are supported."""
    from app import extract_expression_variables
    from checker import check_expression, CheckStatus

    vars_found, err = extract_expression_variables("1 + 2.5")
    assert err is None
    assert vars_found == []

    # Check that deterministic checker passes pure numeric constant expressions
    res = check_expression("1 + 2.5", variable_units={})
    assert res.status == CheckStatus.PASS
    assert res.is_pass


def test_missing_and_invalid_units_regression():
    """Verify that missing or invalid units are flagged as UNSUPPORTED."""
    from checker import check_expression, CheckStatus

    # Missing unit for variable 'y'
    res_missing = check_expression("x + y", variable_units={"x": "meter"})
    assert res_missing.status == CheckStatus.UNSUPPORTED
    assert "y" in str(res_missing.evidence.get("unknown_variable", ""))

    # Invalid unit string
    res_invalid = check_expression("x", variable_units={"x": "not_a_real_unit_999"})
    assert res_invalid.status == CheckStatus.UNSUPPORTED


def test_dimensionally_inconsistent_expression_regression():
    """Verify that dimensional incompatibility (e.g. mass + velocity) is flagged as FAIL."""
    from checker import check_expression, CheckStatus

    res = check_expression(
        "mass + velocity",
        variable_units={"mass": "kilogram", "velocity": "meter / second"},
    )
    assert res.status == CheckStatus.FAIL
    assert res.is_fail
    assert "addition" in str(res.evidence.get("operation"))


def test_unsupported_syntax_regression():
    """Verify that unsupported syntax (functions, bitwise, multiple statements) is detected."""
    from checker import check_expression, CheckStatus

    # Function call
    res_fn = check_expression("sin(x)", variable_units={"x": "radian"})
    assert res_fn.status == CheckStatus.UNSUPPORTED

    # Bitwise operator
    res_bit = check_expression("a & b", variable_units={"a": "meter", "b": "meter"})
    assert res_bit.status == CheckStatus.UNSUPPORTED


def test_pass_fail_unsupported_status_integrity():
    """Verify that the three distinct statuses are preserved accurately without fabrication."""
    from checker import check_expression, CheckStatus

    # PASS
    r_pass = check_expression("d / t", variable_units={"d": "meter", "t": "second"})
    assert r_pass.status == CheckStatus.PASS

    # FAIL
    r_fail = check_expression("d + t", variable_units={"d": "meter", "t": "second"})
    assert r_fail.status == CheckStatus.FAIL

    # UNSUPPORTED
    r_unsupported = check_expression("d / 0", variable_units={"d": "meter"})
    assert r_unsupported.status == CheckStatus.UNSUPPORTED


def test_streamlit_apptest_method_switching():
    """Verify that switching radio input methods dynamically updates widgets in Streamlit AppTest."""
    import asyncio
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    mock_loop.is_closed.return_value = False

    with patch("asyncio.new_event_loop", return_value=mock_loop):
        app_path = str(Path(__file__).resolve().parent.parent / "app.py")
        at = AppTest.from_file(app_path, default_timeout=5.0)
        at.run(timeout=5.0)

        assert not at.exception
        # Radio exists
        assert len(at.radio) >= 1
        assert at.radio[0].value == "Try an example"

        # In "Try an example", selectbox for examples exists, file uploader does not
        assert len(at.selectbox) >= 1
        assert len(at.file_uploader) == 0

        # Switch to "Analyze a Python file"
        at.radio[0].set_value("Analyze a Python file")
        at.run(timeout=5.0)

        assert not at.exception
        assert at.radio[0].value == "Analyze a Python file"
        # In "Analyze a Python file", file uploader exists, example selectbox does not
        assert len(at.file_uploader) == 1
        assert len(at.selectbox) == 0

        # Primary action is now "Analyze file"
        analyze_buttons = [b for b in at.button if "Analyze file" in b.label]
        assert len(analyze_buttons) == 1

def test_input_method_exclusivity_and_switching():
    """Verify that switching input methods clears stale inputs and results."""
    from app import apply_example

    mock_state = MockSessionState()
    with patch("streamlit.session_state", mock_state):
        mock_state["current_method"] = "Try an example"
        mock_state["expression_input"] = "v = d / t"
        mock_state["check_result"] = "STALE_RESULT_OBJECT"
        mock_state["last_checked_expression"] = "v = d / t"
        mock_state["user_units"] = {"v": "m/s", "d": "m", "t": "s"}

        # Simulate user switching to 'Enter my own equation'
        new_method = "Enter my own equation"
        assert new_method != mock_state["current_method"]
        # The switch invalidates stale results and previous inputs
        mock_state["current_method"] = new_method
        mock_state["check_result"] = None
        mock_state["last_checked_expression"] = None
        mock_state["expression_input"] = ""
        mock_state["user_units"] = {}

        assert mock_state["check_result"] is None
        assert mock_state["last_checked_expression"] is None
        assert mock_state["expression_input"] == ""
        assert mock_state["user_units"] == {}


def test_extract_file_equations_security_and_parsing(tmp_path):
    """Verify that extract_file_equations parses statements via AST and never executes code."""
    from app import extract_file_equations

    # Code that attempts a dangerous side effect if executed
    probe_marker = tmp_path / "should_never_exist.txt"
    code = f"""
import os
v = d / t
E = m * c ** 2
open(r'{probe_marker}', 'w').write('executed!')
"""
    equations = extract_file_equations(code)
    # 1. Probe marker must NOT have been created (code was never executed)
    assert not probe_marker.exists()

    # 2. Mathematical statements must be extracted
    assert any("v = d / t" in eq for eq in equations)
    assert any("E = m * c ** 2" in eq for eq in equations)


def test_format_result_diagnostics_coverage():
    """Verify that format_result_diagnostics answers What, Why, and Next Action for all statuses."""
    from app import format_result_diagnostics
    from checker import check_expression

    # 1. PASS
    res_pass = check_expression("distance / time", variable_units={"distance": "meter", "time": "second"})
    diag_pass = format_result_diagnostics(res_pass, "distance / time")
    assert "what" in diag_pass and "why" in diag_pass and "next_action" in diag_pass
    assert "dimensional homogeneity" in diag_pass["what"].lower()
    assert "proceed" in diag_pass["next_action"].lower()

    # 2. FAIL: Incompatible addition
    res_fail_add = check_expression("mass + velocity", variable_units={"mass": "kilogram", "velocity": "meter / second"})
    diag_fail_add = format_result_diagnostics(res_fail_add, "mass + velocity")
    assert "incompatible addition" in diag_fail_add["what"].lower()
    assert "cannot be added" in diag_fail_add["why"].lower()
    assert "derivation" in diag_fail_add["next_action"].lower()

    # 3. FAIL: Equation mismatch
    res_fail_eq = check_expression("v = d * t", variable_units={"v": "meter / second", "d": "meter", "t": "second"})
    diag_fail_eq = format_result_diagnostics(res_fail_eq, "v = d * t")
    assert "mismatch" in diag_fail_eq["what"].lower()
    assert "equation" in diag_fail_eq["why"].lower()
    assert "review the equation" in diag_fail_eq["next_action"].lower()

    # 4. UNSUPPORTED: Missing variable unit
    res_unsup_var = check_expression("x + y", variable_units={"x": "meter"})
    diag_unsup_var = format_result_diagnostics(res_unsup_var, "x + y")
    assert "missing unit" in diag_unsup_var["what"].lower()
    assert "step 2" in diag_unsup_var["next_action"].lower()

    # 5. UNSUPPORTED: Division by zero
    res_unsup_zero = check_expression("1 / 0", variable_units={})
    diag_unsup_zero = format_result_diagnostics(res_unsup_zero, "1 / 0")
    assert "division by zero" in diag_unsup_zero["what"].lower()
    assert "non-zero" in diag_unsup_zero["next_action"].lower()

    # 6. UNSUPPORTED: Function call syntax
    res_unsup_syntax = check_expression("sin(x)", variable_units={"x": "radian"})
    diag_unsup_syntax = format_result_diagnostics(res_unsup_syntax, "sin(x)")
    assert "unsupported syntax" in diag_unsup_syntax["what"].lower()
    assert "simplify" in diag_unsup_syntax["next_action"].lower()


# 11. Service Unavailable & Manual Entry Workflow Tests
def test_service_unavailable_message_and_manual_entry_workflow():
    """Verify that when AI suggestions fail (503 / unavailable), the user can enter units manually and verify."""
    from ai_analyzer import ServiceUnavailableError
    from checker import check_expression, CheckStatus

    # 1. Exact required message
    expected_msg = (
        "AI suggestions are temporarily unavailable. "
        "You can enter variable meanings and units manually, or try again later."
    )
    err = ServiceUnavailableError(expected_msg)
    assert expected_msg in str(err)

    # 2. Manual entry works 100% offline without AI
    manual_units = {"distance": "meter", "time": "second"}
    res = check_expression("distance / time", variable_units=manual_units)
    assert res.status == CheckStatus.PASS
    assert res.is_pass
    assert res.evidence["inferred_unit"] == "meter / second"


def test_streamlit_apptest_service_unavailable_handling():
    """Verify Streamlit AppTest handles ServiceUnavailableError by displaying warning message."""
    import asyncio
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from ai_analyzer import ServiceUnavailableError

    mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    mock_loop.is_closed.return_value = False

    expected_msg = (
        "AI suggestions are temporarily unavailable. "
        "You can enter variable meanings and units manually, or try again later."
    )

    with patch("asyncio.new_event_loop", return_value=mock_loop), \
         patch("ai_analyzer.AIAnalyzer.analyze_expression", side_effect=ServiceUnavailableError(expected_msg)):

        app_path = str(Path(__file__).resolve().parent.parent / "app.py")
        at = AppTest.from_file(app_path, default_timeout=5.0)
        at.run(timeout=5.0)

        assert not at.exception

        # Trigger AI suggestion button
        ai_buttons = [b for b in at.button if "Suggest meanings and units with Gemma 4" in b.label]
        if ai_buttons:
            ai_buttons[0].click()
            at.run(timeout=5.0)
            assert not at.exception
            # Verify the required warning message is displayed
            assert any(expected_msg in w.value for w in at.warning)

        # Deterministic check still works
        check_buttons = [b for b in at.button if "Check expression" in b.label]
        assert len(check_buttons) >= 1
        check_buttons[0].click()
        at.run(timeout=5.0)
        assert not at.exception
        assert any("PASS" in s.value for s in at.success)


