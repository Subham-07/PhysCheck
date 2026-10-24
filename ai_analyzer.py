"""AI Expression Analyzer for PhysCheck (Milestone 3).

DATA TRANSMISSION NOTICE:
-------------------------
Code, mathematical expressions, variable names, and context submitted for AI
analysis are transmitted over the network to the configured API provider (Google GenAI).
Do NOT submit confidential code, proprietary secrets, or private personal data
if your organization or compliance policies disallow external API transmission.

KEY SAFETY RULES:
-----------------
1. Reads GEMINI_API_KEY securely from environment or .env via python-dotenv.
2. Never hard-codes, logs, prints, or exposes the API key.
3. Requests structured output with strict schema: summary, variable meanings,
   candidate units, uncertainty, assumptions, potential issues, and limitations.
4. Validates the response via Pydantic before returning.
5. All model-generated units are PRELIMINARY SUGGESTIONS ONLY until explicitly
   confirmed by the user.
6. NEVER executes code returned by the model.
7. Uses the requested model identifier 'gemma-4-31b-it'.
8. NEVER silently substitutes a different model if the requested model is unavailable.
9. Handles missing keys, authentication errors, unsupported model errors,
   quota errors, timeouts, and malformed responses.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, model_validator

from google import genai
from google.genai import types, errors

# Mandatory notices and disclaimers
DATA_TRANSMISSION_NOTICE = (
    "NOTICE: Code, expressions, and context submitted for AI analysis are "
    "transmitted over the network to the configured API provider (Google GenAI)."
)

SUGGESTIONS_NOTICE = (
    "All model-generated units, variable interpretations, and physical assumptions "
    "are preliminary suggestions only and MUST be explicitly confirmed by the user "
    "before use in deterministic verification or scientific calculations."
)

DEFAULT_MODEL_IDENTIFIER = "gemma-4-31b-it"


# ============================================================================
# Exception Hierarchy
# ============================================================================

class AIAnalysisError(Exception):
    """Base exception for all AI analysis errors."""


class MissingAPIKeyError(AIAnalysisError):
    """Raised when GEMINI_API_KEY is not set or empty."""


class AuthenticationError(AIAnalysisError):
    """Raised when the API key is invalid or unauthorized (401/403)."""


class UnsupportedModelError(AIAnalysisError):
    """Raised when the requested model is unavailable or unsupported by the API provider."""


class QuotaExceededError(AIAnalysisError):
    """Raised when API quota or rate limits are exhausted (429)."""


class AIAnalyzerTimeoutError(AIAnalysisError):
    """Raised when an API request times out."""


class ServiceUnavailableError(AIAnalysisError):
    """Raised when the AI service is temporarily unavailable (HTTP 503 or 5xx) after retries."""


class RetriesExhaustedError(AIAnalysisError):
    """Raised when transient retries for 503, 429, or 5xx are exhausted."""


class MalformedResponseError(AIAnalysisError):
    """Raised when model output fails JSON parsing or schema validation."""


# Retry configuration defaults
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0  # seconds
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_MAX_DELAY = 8.0
DEFAULT_OVERALL_TIMEOUT = 30.0  # seconds

# Transient retryable HTTP status codes
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def sanitize_error_message(msg: str, secret: Optional[str] = None) -> str:
    """Mask API keys or credentials from error messages to protect secrets."""
    if not msg:
        return ""
    import re
    cleaned = str(msg)
    if secret and secret.strip():
        cleaned = cleaned.replace(secret.strip(), "***REDACTED***")
    # Redact standard Gemini API key format (AIzaSy...)
    cleaned = re.sub(r"AIzaSy[A-Za-z0-9_-]{33}", "***REDACTED***", cleaned)
    # Redact any generic key= or api_key= pattern
    cleaned = re.sub(r"(api[_-]?key\s*[:=]\s*)[^\s,;&'\"]+", r"\1***REDACTED***", cleaned, flags=re.IGNORECASE)
    return cleaned


def is_retryable_error(exc: Exception) -> bool:
    """Determine whether an error is transient and retryable (503, 429, 5xx, timeout).

    Non-retryable errors (e.g. 401, 403, 404, bad requests, invalid API key) return False.
    """
    # 1. Official Google GenAI API errors
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        # Explicit non-retryable status codes
        if code in (400, 401, 403, 404, 422):
            return False

        # Retryable status codes: 503, 429, 408, and all 5xx
        if code in RETRYABLE_STATUS_CODES or (code is not None and 500 <= code < 600):
            return True

        status = str(getattr(exc, "status", "") or "").upper()
        if any(term in status for term in ("UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED")):
            return True

        msg = str(getattr(exc, "message", "") or str(exc)).lower()
        if "503" in msg or "service unavailable" in msg or "temporarily unavailable" in msg:
            return True

        return False

    # 2. Specific PhysCheck transient exceptions
    if isinstance(exc, (ServiceUnavailableError, QuotaExceededError, AIAnalyzerTimeoutError)):
        return True

    # 3. Non-retryable PhysCheck exceptions
    if isinstance(exc, (MissingAPIKeyError, AuthenticationError, UnsupportedModelError, MalformedResponseError, ValueError)):
        return False

    # 4. Check error string for transient network/server keywords while excluding client/auth errors
    err_str = str(exc).lower()
    if any(auth_term in err_str for term in ("unauthenticated", "401", "403", "forbidden", "api_key_invalid", "404", "not found")):
        return False

    if any(transient_term in err_str for transient_term in ("503", "service unavailable", "timed out", "timeout", "connection reset", "connection refused", "502 bad gateway")):
        return True

    return False


# ============================================================================
# Structured Output Schema (Pydantic Models)
# ============================================================================

class CandidateUnitItem(BaseModel):
    """A suggested candidate physical unit for a variable."""

    variable: str = Field(description="Variable identifier or symbol name")
    unit: str = Field(description="Candidate physical unit string compatible with Pint (e.g. 'meter / second')")


class VariableMeaning(BaseModel):
    """Physical meaning and suggested units for an extracted variable."""

    variable: str = Field(description="Variable identifier or symbol name")
    meaning: str = Field(description="Physical meaning, definition, or context of the variable")
    candidate_units: List[str] = Field(
        default_factory=list,
        description="Candidate physical unit strings compatible with Pint (e.g., 'meter / second')",
    )
    is_user_confirmed: bool = Field(
        default=False,
        description="Whether this unit suggestion has been confirmed by user. Always False initially.",
    )


class AIAnalysisResult(BaseModel):
    """Structured response produced by the AI physical expression analyzer."""

    summary: str = Field(description="High-level physical description of the expression or formula")
    variable_meanings: List[VariableMeaning] = Field(
        default_factory=list,
        description="Physical interpretations and candidate units for variables",
    )
    suggested_units: List[CandidateUnitItem] = Field(
        default_factory=list,
        description="List of suggested units for variables (suggestions only)",
    )
    uncertainty: str = Field(
        description="Ambiguities, uncertainties, or alternative interpretations in the formulation",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Physical assumptions underlying this formula (e.g., non-relativistic, ideal fluid)",
    )
    potential_issues: List[str] = Field(
        default_factory=list,
        description="Potential physical law violations, singularities, domain bounds, or dimensional risks",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Limitations of the model analysis and formulation scope",
    )
    model_identifier: str = Field(
        default=DEFAULT_MODEL_IDENTIFIER,
        description="The exact model identifier used for analysis",
    )
    is_confirmed_by_user: bool = Field(
        default=False,
        description="Flag indicating whether suggestions have been confirmed by user. Always False on creation.",
    )
    data_transmission_notice: str = Field(
        default=DATA_TRANSMISSION_NOTICE,
        description="Transmission disclosure notice",
    )
    suggestions_notice: str = Field(
        default=SUGGESTIONS_NOTICE,
        description="Disclaimer stating model units are unconfirmed suggestions",
    )

    @model_validator(mode="before")
    @classmethod
    def _handle_candidate_units_dict(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "candidate_units" in data and "suggested_units" not in data:
                cu = data["candidate_units"]
                if isinstance(cu, dict):
                    data["suggested_units"] = [
                        {"variable": k, "unit": v} for k, v in cu.items()
                    ]
        return data

    @property
    def candidate_units(self) -> Dict[str, str]:
        """Dictionary mapping variable names to their primary candidate unit strings."""
        units = {item.variable: item.unit for item in self.suggested_units}
        for vm in self.variable_meanings:
            if vm.variable not in units and vm.candidate_units:
                units[vm.variable] = vm.candidate_units[0]
        return units

    def confirm_units(self, confirmed_mapping: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Confirm candidate units for use in deterministic checker.

        Args:
            confirmed_mapping: Optional override mapping from user. If None, confirms current candidates.

        Returns:
            Dictionary of confirmed variable units.
        """
        if confirmed_mapping is not None:
            self.suggested_units = [
                CandidateUnitItem(variable=k, unit=v) for k, v in confirmed_mapping.items()
            ]

        self.is_confirmed_by_user = True
        for vm in self.variable_meanings:
            if vm.variable in self.candidate_units:
                vm.is_user_confirmed = True
        return dict(self.candidate_units)

    def get_confirmed_units(self) -> Dict[str, str]:
        """Return variable units only if explicitly confirmed by the user.

        Raises:
            ValueError: If units have not yet been confirmed by user.
        """
        if not self.is_confirmed_by_user:
            raise ValueError(
                "Candidate units have not been confirmed by the user. "
                "Treat all AI-generated units as suggestions until explicitly confirmed."
            )
        return dict(self.candidate_units)


# ============================================================================
# API Key Retrieval
# ============================================================================

def get_gemini_api_key(dotenv_path: Optional[str] = None) -> str:
    """Retrieve GEMINI_API_KEY from environment or .env using python-dotenv.

    Security Note: Never hard-codes, prints, logs, or exposes the key.
    """
    load_dotenv(dotenv_path=dotenv_path, override=False)
    key = os.getenv("GEMINI_API_KEY")
    if not key or not key.strip():
        raise MissingAPIKeyError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please configure GEMINI_API_KEY in your environment or .env file."
        )
    return key.strip()


# ============================================================================
# AI Analyzer Implementation
# ============================================================================

class AIAnalyzer:
    """Physical expression analyzer powered by Google GenAI.

    DISCLOSURE:
    Expressions and context passed to this analyzer are transmitted over the
    network to the configured Google GenAI API endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL_IDENTIFIER,
        client: Optional[genai.Client] = None,
        timeout: Optional[float] = 30.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        max_delay: float = DEFAULT_MAX_DELAY,
        overall_timeout: Optional[float] = DEFAULT_OVERALL_TIMEOUT,
    ):
        """Initialize AIAnalyzer.

        Args:
            api_key: Optional API key. If omitted, loaded securely from environment.
            model: Model identifier. Defaults strictly to 'gemma-4-31b-it'.
            client: Optional pre-configured genai.Client (useful for testing/mocking).
            timeout: API timeout in seconds.
            max_retries: Maximum number of retries for transient errors (default 3).
            initial_delay: Initial retry backoff delay in seconds (default 1.0).
            backoff_factor: Exponential backoff multiplier (default 2.0).
            max_delay: Maximum delay cap between retries in seconds (default 8.0).
            overall_timeout: Maximum overall timeout across retries in seconds (default 30.0).
        """
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.overall_timeout = overall_timeout
        self._raw_api_key = api_key
        self._client = client

    def __repr__(self) -> str:
        # Prevent API key leakage in logs or representations
        return (
            f"AIAnalyzer(model={self.model!r}, timeout={self.timeout!r}, "
            f"max_retries={self.max_retries}, api_key='***REDACTED***')"
        )

    def _get_client(self) -> genai.Client:
        """Initialize or return the genai.Client without exposing credentials."""
        if self._client is not None:
            return self._client

        api_key = self._raw_api_key or get_gemini_api_key()
        # Explicitly configure attempts=1 in SDK client options so the SDK does not perform
        # duplicate internal retries; PhysCheck manages retries with backoff and UI feedback.
        http_options = types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
            timeout=int(self.timeout * 1000) if self.timeout else None,
        )
        self._client = genai.Client(api_key=api_key, http_options=http_options)
        return self._client

    def analyze_expression(
        self,
        expression: str,
        context: Optional[str] = None,
        on_retry: Optional[Callable[[int, int, float, Exception], None]] = None,
    ) -> AIAnalysisResult:
        """Analyze a physical formula or expression and produce structured physical reasoning.

        DATA TRANSMISSION NOTICE:
        The expression and context provided will be transmitted to the Google GenAI API.

        Args:
            expression: Mathematical expression or formula string (e.g. 'E = m * c ** 2').
            context: Optional domain context (e.g. 'special relativity, rest mass energy').
            on_retry: Optional callback invoked before each retry with (attempt, max_retries, delay, exception).

        Returns:
            AIAnalysisResult with candidate units (suggestions only until confirmed),
            variable meanings, summary, assumptions, uncertainty, and limitations.
        """
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("Expression must be a non-empty string.")

        client = self._get_client()

        prompt = (
            f"You are a physics expert. Analyze the following physical expression or formula:\n\n"
            f"Expression: {expression.strip()}\n"
        )
        if context:
            prompt += f"Context: {context.strip()}\n\n"

        prompt += (
            "\nAnalyze this formula and return a strictly structured JSON object adhering to the schema:\n"
            "- summary: Brief explanation of the physical meaning and application\n"
            "- variable_meanings: List of variables, their physical meaning, and candidate units compatible with Pint\n"
            "- candidate_units: Dictionary mapping each variable name to its most likely Pint unit string (e.g. 'meter / second')\n"
            "- uncertainty: Any ambiguity, competing definitions, or uncertainties in this formula\n"
            "- assumptions: Fundamental physical assumptions required for this equation to hold\n"
            "- potential_issues: Boundary violations, divide-by-zero risks, relativistic limits, or unit mismatches\n"
            "- limitations: Regimes where this formula is invalid or breaks down\n"
        )

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AIAnalysisResult,
        )

        start_time = time.time()
        attempt = 0
        response = None

        while attempt <= self.max_retries:
            # Check overall timeout before making API request
            if self.overall_timeout is not None and (time.time() - start_time) > self.overall_timeout:
                raise AIAnalyzerTimeoutError(
                    f"Overall timeout of {self.overall_timeout}s exceeded while analyzing expression."
                )

            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                break
            except Exception as e:
                # 1. Non-retryable errors fail immediately without retry
                if not is_retryable_error(e):
                    if isinstance(e, errors.APIError):
                        self._handle_api_error(e)
                    self._handle_general_exception(e)

                # 2. Transient error: check if retries exhausted
                if attempt >= self.max_retries:
                    self._handle_retries_exhausted(e, attempt)

                # 3. Compute exponential backoff delay
                delay = min(self.initial_delay * (self.backoff_factor ** attempt), self.max_delay)

                # Check overall timeout before sleeping
                if self.overall_timeout is not None and (time.time() - start_time + delay) > self.overall_timeout:
                    raise AIAnalyzerTimeoutError(
                        f"Overall timeout of {self.overall_timeout}s exceeded while waiting to retry."
                    ) from e

                # 4. Notify caller (e.g. UI) of retry
                if on_retry is not None:
                    try:
                        on_retry(attempt + 1, self.max_retries, delay, e)
                    except Exception:
                        pass

                time.sleep(delay)
                attempt += 1

        # Validate response content
        if not response or not getattr(response, "text", None):
            raise MalformedResponseError("Received empty or null response from model.")

        raw_text = response.text.strip()
        try:
            # Validate JSON against Pydantic model
            parsed_data = AIAnalysisResult.model_validate_json(raw_text)
            # Enforce that the model identifier matches and user confirmation is False
            parsed_data.model_identifier = self.model
            parsed_data.is_confirmed_by_user = False
            for vm in parsed_data.variable_meanings:
                vm.is_user_confirmed = False
            return parsed_data
        except (ValidationError, json.JSONDecodeError) as e:
            raise MalformedResponseError(
                f"Model response failed schema validation or JSON decoding: {e}"
            ) from e

    def _handle_retries_exhausted(self, e: Exception, attempts_made: int) -> None:
        """Handle exhausted retries for transient errors."""
        code = getattr(e, "code", None)
        clean_msg = sanitize_error_message(str(e))
        clean_lower = clean_msg.lower()

        if code == 429 or "quota" in clean_lower or "resource_exhausted" in clean_lower:
            raise QuotaExceededError(
                f"API quota or rate limit exceeded after {attempts_made} retries: {clean_msg}"
            ) from e

        if code in (408, 504) or "timeout" in clean_lower or "deadline_exceeded" in clean_lower:
            raise AIAnalyzerTimeoutError(
                f"API request timed out after {attempts_made} retries: {clean_msg}"
            ) from e

        # HTTP 503 Service Unavailable or general 5xx Server Error
        raise ServiceUnavailableError(
            "AI suggestions are temporarily unavailable. You can enter variable meanings and units manually, or try again later."
        ) from e

    def _handle_api_error(self, e: errors.APIError) -> None:
        """Map official google.genai errors.APIError to specific PhysCheck exceptions.

        STRICT RULE: Never silently substitute a different model if the requested model is unavailable.
        Never expose API key in error messages.
        """
        code = getattr(e, "code", None)
        msg = getattr(e, "message", str(e))
        clean_msg = sanitize_error_message(str(msg))
        msg_lower = clean_msg.lower()

        # 401 / 403 Authentication & Permission errors
        if code in (401, 403) or "unauthenticated" in msg_lower or "api_key_invalid" in msg_lower:
            raise AuthenticationError(
                f"API authentication failed (status {code}). "
                f"Please verify that your GEMINI_API_KEY is valid."
            ) from e

        # 404 / Model Not Found / Unsupported Model
        # STRICT REQUIREMENT: Do NOT silently substitute a different model!
        if (
            code == 404
            or "not_found" in msg_lower
            or "model not found" in msg_lower
            or "not supported" in msg_lower
            or "unsupported" in msg_lower
        ):
            raise UnsupportedModelError(
                f"Requested model '{self.model}' is not available or unsupported by the API provider (status {code}): {clean_msg}. "
                f"PhysCheck will not silently substitute a different model."
            ) from e

        # 429 Quota / Rate limit
        if code == 429 or "resource_exhausted" in msg_lower or "quota" in msg_lower:
            raise QuotaExceededError(
                f"API quota or rate limit exceeded (status {code}): {clean_msg}"
            ) from e

        # Timeout indications
        if code in (408, 504) or "timeout" in msg_lower or "deadline_exceeded" in msg_lower:
            raise AIAnalyzerTimeoutError(
                f"API request timed out (status {code}): {clean_msg}"
            ) from e

        # 503 Service Unavailable / 5xx Server Error
        if code == 503 or (code is not None and 500 <= code < 600) or "unavailable" in msg_lower:
            raise ServiceUnavailableError(
                "AI suggestions are temporarily unavailable. You can enter variable meanings and units manually, or try again later."
            ) from e

        # General API error (e.g. 400 Bad Request)
        raise AIAnalysisError(f"API error occurred (status {code}): {clean_msg}") from e

    def _handle_general_exception(self, e: Exception) -> None:
        """Handle transport or generic non-API exceptions securely."""
        clean_str = sanitize_error_message(str(e))
        error_str = clean_str.lower()
        if "timeout" in error_str or "timed out" in error_str:
            raise AIAnalyzerTimeoutError(f"Request timed out while analyzing expression: {clean_str}") from e
        if "quota" in error_str or "resource_exhausted" in error_str:
            raise QuotaExceededError(f"API quota exceeded: {clean_str}") from e
        if "unauthenticated" in error_str or "api_key_invalid" in error_str:
            raise AuthenticationError(f"API authentication failed: {clean_str}") from e
        if "not found" in error_str and self.model in clean_str:
            raise UnsupportedModelError(
                f"Requested model '{self.model}' is not available or unsupported by API provider. "
                f"PhysCheck will not silently substitute another model."
            ) from e
        if "503" in error_str or "service unavailable" in error_str:
            raise ServiceUnavailableError(
                "AI suggestions are temporarily unavailable. You can enter variable meanings and units manually, or try again later."
            ) from e
        raise AIAnalysisError(f"API request failed: {clean_str}") from e


def analyze_expression_with_ai(
    expression: str,
    context: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL_IDENTIFIER,
    on_retry: Optional[Callable[[int, int, float, Exception], None]] = None,
) -> AIAnalysisResult:
    """Convenience helper to analyze an expression using AIAnalyzer.

    DATA TRANSMISSION NOTICE:
    Transmits the expression and context over the network to the configured API provider.
    """
    analyzer = AIAnalyzer(api_key=api_key, model=model)
    return analyzer.analyze_expression(expression=expression, context=context, on_retry=on_retry)
