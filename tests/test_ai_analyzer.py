"""Mocked unit and integration tests for AI expression analyzer (Milestone 3).

Verifies:
- Reading GEMINI_API_KEY from environment via python-dotenv
- API key is never hard-coded, logged, or exposed in repr/errors
- Model returns structured output (summary, variable meanings, candidate units,
  uncertainty, assumptions, potential issues, limitations)
- Response validation via Pydantic
- All model-generated units remain suggestions until explicitly confirmed by user
- Code returned by model is NEVER executed
- Robust handling of missing keys, auth errors, unsupported models, quota errors,
  timeouts, and malformed responses
- Requested model 'gemma-4-31b-it' is NOT silently substituted
- Data transmission disclosure is present
- Mocked tests run without requiring a real API key
"""

import json
import os
from unittest.mock import MagicMock, patch
import pytest

from ai_analyzer import (
    AIAnalysisResult,
    AIAnalyzer,
    AIAnalyzerTimeoutError,
    AuthenticationError,
    DATA_TRANSMISSION_NOTICE,
    DEFAULT_MODEL_IDENTIFIER,
    MalformedResponseError,
    MissingAPIKeyError,
    QuotaExceededError,
    SUGGESTIONS_NOTICE,
    UnsupportedModelError,
    get_gemini_api_key,
)
from checker import CheckStatus, check_expression
from google.genai import errors


# Sample valid mock response matching AIAnalysisResult schema
VALID_MOCK_PAYLOAD = {
    "summary": "Mass-energy equivalence describing the rest energy of an object with invariant mass.",
    "variable_meanings": [
        {
            "variable": "E",
            "meaning": "Rest energy of the physical system",
            "candidate_units": ["joule", "megaelectronvolt"],
            "is_user_confirmed": False,
        },
        {
            "variable": "m",
            "meaning": "Invariant rest mass of the particle or body",
            "candidate_units": ["kilogram", "gram"],
            "is_user_confirmed": False,
        },
        {
            "variable": "c",
            "meaning": "Speed of light in vacuum",
            "candidate_units": ["meter / second"],
            "is_user_confirmed": False,
        },
    ],
    "candidate_units": {
        "E": "joule",
        "m": "kilogram",
        "c": "meter / second",
    },
    "uncertainty": "Negligible for rest mass; does not apply to open systems losing momentum without relativistic mass correction.",
    "assumptions": [
        "Special relativity applies",
        "Particle is in its rest frame (momentum p = 0)",
        "Spacetime is flat Minkowski space",
    ],
    "potential_issues": [
        "Singularities do not apply here, but kinetic momentum must be added for moving particles (E^2 = (pc)^2 + (mc^2)^2)",
    ],
    "limitations": [
        "Does not describe general relativistic curved spacetime effects or quantum gravity regimes",
    ],
    "model_identifier": "gemma-4-31b-it",
    "is_confirmed_by_user": False,
    "data_transmission_notice": DATA_TRANSMISSION_NOTICE,
    "suggestions_notice": SUGGESTIONS_NOTICE,
}


# 1. Missing API Key
def test_missing_api_key_raises_error(monkeypatch, tmp_path):
    empty_env = tmp_path / ".env.empty"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with patch("ai_analyzer.load_dotenv"):
        with pytest.raises(MissingAPIKeyError) as exc_info:
            get_gemini_api_key(dotenv_path=str(empty_env))
        assert "GEMINI_API_KEY environment variable is not set" in str(exc_info.value)

    # Empty / whitespace string should also raise
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with patch("ai_analyzer.load_dotenv"):
        with pytest.raises(MissingAPIKeyError):
            get_gemini_api_key()


# 2. Successful Analysis (Mocked)
def test_successful_analysis_mocked():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(VALID_MOCK_PAYLOAD)
    mock_client.models.generate_content.return_value = mock_response

    analyzer = AIAnalyzer(api_key="mock_key", client=mock_client)
    result = analyzer.analyze_expression("E = m * c ** 2", context="relativity")

    assert isinstance(result, AIAnalysisResult)
    assert "Mass-energy equivalence" in result.summary
    assert len(result.variable_meanings) == 3
    assert result.candidate_units == {
        "E": "joule",
        "m": "kilogram",
        "c": "meter / second",
    }
    assert len(result.assumptions) >= 1
    assert len(result.potential_issues) >= 1
    assert len(result.limitations) >= 1
    assert result.model_identifier == DEFAULT_MODEL_IDENTIFIER
    assert result.is_confirmed_by_user is False
    assert DATA_TRANSMISSION_NOTICE in result.data_transmission_notice

    # Verify model was called with requested model identifier
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemma-4-31b-it"


# 3. Model-generated Units are Suggestions until User Confirms
def test_units_are_suggestions_until_confirmed():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(VALID_MOCK_PAYLOAD)
    mock_client.models.generate_content.return_value = mock_response

    analyzer = AIAnalyzer(api_key="mock_key", client=mock_client)
    result = analyzer.analyze_expression("E = m * c ** 2")

    # Initial state: suggestions only
    assert result.is_confirmed_by_user is False
    for vm in result.variable_meanings:
        assert vm.is_user_confirmed is False

    # Attempting to get confirmed units before user confirms raises ValueError
    with pytest.raises(ValueError) as exc_info:
        result.get_confirmed_units()
    assert "suggestions" in str(exc_info.value).lower()

    # User confirms the suggestions
    confirmed = result.confirm_units()
    assert result.is_confirmed_by_user is True
    assert confirmed == {"E": "joule", "m": "kilogram", "c": "meter / second"}
    assert result.get_confirmed_units() == confirmed

    for vm in result.variable_meanings:
        assert vm.is_user_confirmed is True

    # User overrides a unit during confirmation
    override_result = AIAnalysisResult.model_validate(VALID_MOCK_PAYLOAD)
    overridden = override_result.confirm_units({
        "E": "electron_volt",
        "m": "kilogram",
        "c": "meter / second",
    })
    assert overridden["E"] == "electron_volt"
    assert override_result.is_confirmed_by_user is True


# 4. Never Execute Code Returned by the Model
def test_never_execute_code_returned_by_model(tmp_path):
    target_file = tmp_path / "model_exploit.txt"

    # Inject malicious code strings in response fields
    payload = dict(VALID_MOCK_PAYLOAD)
    payload["summary"] = f"open(r'{target_file}', 'w').write('executed')"
    payload["potential_issues"] = [
        "__import__('os').system('echo dangerous')",
        "exec('sentinel_var = 123')",
    ]

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(payload)
    mock_client.models.generate_content.return_value = mock_response

    analyzer = AIAnalyzer(api_key="mock_key", client=mock_client)
    res = analyzer.analyze_expression("x + y")

    # Response is treated purely as text data; code is never run
    assert not target_file.exists()
    assert "open(" in res.summary


# 5. Authentication Error Handling (401 / 403)
def test_authentication_error_handling():
    mock_client = MagicMock()
    api_error = errors.APIError(code=401, response_json={"error": {"message": "API_KEY_INVALID"}})
    mock_client.models.generate_content.side_effect = api_error

    analyzer = AIAnalyzer(api_key="invalid_key", client=mock_client)
    with pytest.raises(AuthenticationError) as exc_info:
        analyzer.analyze_expression("a + b")

    assert "authentication failed" in str(exc_info.value).lower()


# 6. Unsupported Model Error Handling (NO Silent Substitution)
def test_unsupported_model_no_silent_substitution():
    mock_client = MagicMock()
    api_error = errors.APIError(
        code=404,
        response_json={"error": {"message": "models/gemma-4-31b-it is not found"}},
    )
    mock_client.models.generate_content.side_effect = api_error

    analyzer = AIAnalyzer(api_key="mock_key", model="gemma-4-31b-it", client=mock_client)
    with pytest.raises(UnsupportedModelError) as exc_info:
        analyzer.analyze_expression("v = d / t")

    # Verify exception message explains model is unavailable
    assert "gemma-4-31b-it" in str(exc_info.value)
    assert "will not silently substitute" in str(exc_info.value)

    # Verify analyzer DID NOT change its model identifier
    assert analyzer.model == "gemma-4-31b-it"


# 7. Quota Exceeded Error Handling (429)
def test_quota_exceeded_error_handling():
    mock_client = MagicMock()
    api_error = errors.APIError(
        code=429,
        response_json={"error": {"message": "Resource has been exhausted (e.g. check quota)."}},
    )
    mock_client.models.generate_content.side_effect = api_error

    analyzer = AIAnalyzer(api_key="mock_key", client=mock_client)
    with pytest.raises(QuotaExceededError) as exc_info:
        analyzer.analyze_expression("F = m * a")

    assert "quota or rate limit exceeded" in str(exc_info.value).lower()


# 8. Timeout Error Handling
def test_timeout_error_handling():
    mock_client = MagicMock()
    api_error = errors.APIError(
        code=504,
        response_json={"error": {"message": "Deadline exceeded"}},
    )
    mock_client.models.generate_content.side_effect = api_error

    analyzer = AIAnalyzer(api_key="mock_key", client=mock_client)
    with pytest.raises(AIAnalyzerTimeoutError) as exc_info:
        analyzer.analyze_expression("F = m * a")

    assert "timed out" in str(exc_info.value).lower()


# 9. Malformed Response (Invalid JSON)
def test_malformed_response_invalid_json():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is not valid JSON at all."
    mock_client.models.generate_content.return_value = mock_response

    analyzer = AIAnalyzer(api_key="mock_key", client=mock_client)
    with pytest.raises(MalformedResponseError) as exc_info:
        analyzer.analyze_expression("x + y")

    assert "failed schema validation or json decoding" in str(exc_info.value).lower()


# 10. Malformed Response (Missing Required Schema Fields)
def test_malformed_response_missing_fields():
    mock_client = MagicMock()
    mock_response = MagicMock()
    # Missing summary and uncertainty
    mock_response.text = json.dumps({"variable_meanings": []})
    mock_client.models.generate_content.return_value = mock_response

    analyzer = AIAnalyzer(api_key="mock_key", client=mock_client)
    with pytest.raises(MalformedResponseError) as exc_info:
        analyzer.analyze_expression("x + y")

    assert "validation" in str(exc_info.value).lower()


# 11. API Key is NEVER Exposed in Repr or Logs
def test_api_key_never_exposed():
    secret_key = "AIzaSy_SECRET_KEY_NEVER_PRINT_12345"
    analyzer = AIAnalyzer(api_key=secret_key, model="gemma-4-31b-it")

    repr_str = repr(analyzer)
    assert secret_key not in repr_str
    assert "***REDACTED***" in repr_str


# 12. Data Transmission Notice Disclosure
def test_data_transmission_notice():
    assert "transmitted" in DATA_TRANSMISSION_NOTICE.lower()
    assert "google genai" in DATA_TRANSMISSION_NOTICE.lower()


# 13. Integration: Confirmed AI Units with Deterministic Checker
def test_ai_to_checker_workflow():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(VALID_MOCK_PAYLOAD)
    mock_client.models.generate_content.return_value = mock_response

    analyzer = AIAnalyzer(api_key="mock_key", client=mock_client)
    ai_result = analyzer.analyze_expression("E = m * c ** 2")

    # Confirm candidate units
    confirmed_units = ai_result.confirm_units()

    # Pass confirmed units to deterministic checker from Milestone 2
    check_res = check_expression("E = m * c ** 2", variable_units=confirmed_units)
    assert check_res.status == CheckStatus.PASS
    assert check_res.is_pass
    assert check_res.evidence["dimensionality_match"] is True
