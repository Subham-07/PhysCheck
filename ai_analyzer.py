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
from typing import Any, Dict, List, Optional
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


class MalformedResponseError(AIAnalysisError):
    """Raised when model output fails JSON parsing or schema validation."""


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
    ):
        """Initialize AIAnalyzer.

        Args:
            api_key: Optional API key. If omitted, loaded securely from environment.
            model: Model identifier. Defaults strictly to 'gemma-4-31b-it'.
            client: Optional pre-configured genai.Client (useful for testing/mocking).
            timeout: API timeout in seconds.
        """
        self.model = model
        self.timeout = timeout
        self._raw_api_key = api_key
        self._client = client

    def __repr__(self) -> str:
        # Prevent API key leakage in logs or representations
        return f"AIAnalyzer(model={self.model!r}, timeout={self.timeout!r}, api_key='***REDACTED***')"

    def _get_client(self) -> genai.Client:
        """Initialize or return the genai.Client without exposing credentials."""
        if self._client is not None:
            return self._client

        api_key = self._raw_api_key or get_gemini_api_key()
        self._client = genai.Client(api_key=api_key)
        return self._client

    def analyze_expression(
        self,
        expression: str,
        context: Optional[str] = None,
    ) -> AIAnalysisResult:
        """Analyze a physical formula or expression and produce structured physical reasoning.

        DATA TRANSMISSION NOTICE:
        The expression and context provided will be transmitted to the Google GenAI API.

        Args:
            expression: Mathematical expression or formula string (e.g. 'E = m * c ** 2').
            context: Optional domain context (e.g. 'special relativity, rest mass energy').

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

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except errors.APIError as e:
            self._handle_api_error(e)
        except Exception as e:
            # Check for network timeout or other wrapped transport errors
            error_str = str(e).lower()
            if "timeout" in error_str or "timed out" in error_str:
                raise AIAnalyzerTimeoutError(f"Request timed out while analyzing expression: {e}") from e
            if "quota" in error_str or "resource_exhausted" in error_str:
                raise QuotaExceededError(f"API quota exceeded: {e}") from e
            if "unauthenticated" in error_str or "api_key_invalid" in error_str:
                raise AuthenticationError(f"API authentication failed: {e}") from e
            if "not found" in error_str and self.model in str(e):
                raise UnsupportedModelError(
                    f"Requested model '{self.model}' is not available or unsupported by API provider. "
                    f"PhysCheck will not silently substitute another model."
                ) from e
            raise AIAnalysisError(f"API request failed: {e}") from e

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

    def _handle_api_error(self, e: errors.APIError) -> None:
        """Map official google.genai errors.APIError to specific PhysCheck exceptions.

        STRICT RULE: Never silently substitute a different model if the requested model is unavailable.
        """
        code = getattr(e, "code", None)
        msg = getattr(e, "message", str(e))
        msg_lower = msg.lower()

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
                f"Requested model '{self.model}' is not available or unsupported by the API provider (status {code}): {msg}. "
                f"PhysCheck will not silently substitute a different model."
            ) from e

        # 429 Quota / Rate limit
        if code == 429 or "resource_exhausted" in msg_lower or "quota" in msg_lower:
            raise QuotaExceededError(
                f"API quota or rate limit exceeded (status {code}): {msg}"
            ) from e

        # Timeout indications
        if code in (408, 504) or "timeout" in msg_lower or "deadline_exceeded" in msg_lower:
            raise AIAnalyzerTimeoutError(
                f"API request timed out (status {code}): {msg}"
            ) from e

        # General API error
        raise AIAnalysisError(f"API error occurred (status {code}): {msg}") from e


def analyze_expression_with_ai(
    expression: str,
    context: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL_IDENTIFIER,
) -> AIAnalysisResult:
    """Convenience helper to analyze an expression using AIAnalyzer.

    DATA TRANSMISSION NOTICE:
    Transmits the expression and context over the network to the configured API provider.
    """
    analyzer = AIAnalyzer(api_key=api_key, model=model)
    return analyzer.analyze_expression(expression=expression, context=context)
