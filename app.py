"""PhysCheck - Scientific Physics Law & Dimensional Consistency Auditor.

Refactored professional scientific-software interface.
Integrates checker.py and ai_analyzer.py without duplicating logic.

Key Principles:
- Clean, restrained, professional scientific UI without decorative emojis or jargon.
- Clear visual hierarchy organized into Sections A through F.
- Automatically identifies real variables actually used in the expression via AST.
- Clearly distinguishes AI-proposed unit suggestions from user-confirmed units.
- Independent offline deterministic verification (operates without API dependencies).
- Clear, actionable result states (PASS, FAIL, UNSUPPORTED) with evidence and practical next steps.
- Security: uploads read as plain text only, zero code execution (no eval / exec).
"""

from __future__ import annotations

import ast
import os
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

# Core imports - never duplicate checker or ai_analyzer logic
from checker import (
    CheckResult,
    CheckStatus,
    PASS_DISCLAIMER,
    check_equation,
    check_expression,
    get_unit_registry,
)
from ai_analyzer import (
    AIAnalysisError,
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
)

# File upload security constraint
MAX_UPLOAD_SIZE_BYTES = 50 * 1024  # 50 KB size limit

# Standard physical symbol defaults for readable explanatory context
COMMON_VARIABLE_METADATA = {
    "distance": {"meaning": "Distance or displacement", "example": "meter"},
    "d": {"meaning": "Distance or displacement", "example": "meter"},
    "x": {"meaning": "Position or displacement", "example": "meter"},
    "time": {"meaning": "Elapsed time", "example": "second"},
    "t": {"meaning": "Elapsed time", "example": "second"},
    "velocity": {"meaning": "Velocity or speed", "example": "meter / second"},
    "v": {"meaning": "Velocity or speed", "example": "meter / second"},
    "mass": {"meaning": "Invariant mass", "example": "kilogram"},
    "m": {"meaning": "Invariant mass", "example": "kilogram"},
    "acceleration": {"meaning": "Acceleration", "example": "meter / second ** 2"},
    "a": {"meaning": "Acceleration", "example": "meter / second ** 2"},
    "force": {"meaning": "Force", "example": "newton"},
    "F": {"meaning": "Force", "example": "newton"},
    "energy": {"meaning": "Energy or work", "example": "joule"},
    "E": {"meaning": "Total or rest energy", "example": "joule"},
    "K": {"meaning": "Kinetic energy", "example": "joule"},
    "U": {"meaning": "Potential energy", "example": "joule"},
    "c": {"meaning": "Speed of light in vacuum", "example": "meter / second"},
    "g": {"meaning": "Gravitational acceleration", "example": "meter / second ** 2"},
    "h": {"meaning": "Height or altitude", "example": "meter"},
    "r": {"meaning": "Radius or radial distance", "example": "meter"},
    "p": {"meaning": "Linear momentum", "example": "kilogram * meter / second"},
}

# Verified built-in examples matching existing test suite and user requirements
BUILTIN_EXAMPLES = {
    "Select an example...": {
        "expr": "",
        "units": {},
        "meanings": {},
        "expected": "",
    },
    "Velocity (distance / time)": {
        "expr": "distance / time",
        "units": {"distance": "meter", "time": "second"},
        "meanings": {"distance": "Distance or displacement", "time": "Elapsed time"},
        "expected": "meter / second",
    },
    "Kinetic Energy (0.5 * mass * velocity**2)": {
        "expr": "0.5 * mass * velocity**2",
        "units": {"mass": "kilogram", "velocity": "meter / second"},
        "meanings": {"mass": "Mass of moving body", "velocity": "Velocity of moving body"},
        "expected": "joule",
    },
    "Incompatible Addition [FAIL DEMO] (mass + velocity)": {
        "expr": "mass + velocity",
        "units": {"mass": "kilogram", "velocity": "meter / second"},
        "meanings": {"mass": "Mass", "velocity": "Velocity"},
        "expected": "",
    },
    "Mass-Energy Equivalence (E = m * c ** 2)": {
        "expr": "E = m * c ** 2",
        "units": {"E": "joule", "m": "kilogram", "c": "meter / second"},
        "meanings": {"E": "Rest energy", "m": "Rest mass", "c": "Speed of light"},
        "expected": "joule",
    },
    "Kinetic Energy (K = 0.5 * m * v ** 2)": {
        "expr": "K = 0.5 * m * v ** 2",
        "units": {"K": "joule", "m": "kilogram", "v": "meter / second"},
        "meanings": {"K": "Kinetic energy", "m": "Mass", "v": "Velocity"},
        "expected": "joule",
    },
    "Kinematics Velocity (v = d / t)": {
        "expr": "v = d / t",
        "units": {"v": "meter / second", "d": "meter", "t": "second"},
        "meanings": {"v": "Velocity", "d": "Distance", "t": "Time"},
        "expected": "meter / second",
    },
    "Newton's Second Law (F = m * a)": {
        "expr": "F = m * a",
        "units": {"F": "newton", "m": "kilogram", "a": "meter / second ** 2"},
        "meanings": {"F": "Net force", "m": "Mass", "a": "Acceleration"},
        "expected": "newton",
    },
    "Centripetal Acceleration (a = v ** 2 / r)": {
        "expr": "a = v ** 2 / r",
        "units": {"a": "meter / second ** 2", "v": "meter / second", "r": "meter"},
        "meanings": {"a": "Centripetal acceleration", "v": "Tangential speed", "r": "Orbit radius"},
        "expected": "meter / second ** 2",
    },
    "Gravitational Potential Energy (U = m * g * h)": {
        "expr": "U = m * g * h",
        "units": {"U": "joule", "m": "kilogram", "g": "meter / second ** 2", "h": "meter"},
        "meanings": {"U": "Potential energy", "m": "Mass", "g": "Gravitational acceleration", "h": "Height"},
        "expected": "joule",
    },
    "Incompatible Addition [FAIL DEMO] (a + b)": {
        "expr": "a + b",
        "units": {"a": "meter", "b": "second"},
        "meanings": {"a": "Displacement", "b": "Duration"},
        "expected": "",
    },
    "Mismatched Equation [FAIL DEMO] (v = d * t)": {
        "expr": "v = d * t",
        "units": {"v": "meter / second", "d": "meter", "t": "second"},
        "meanings": {"v": "Velocity", "d": "Distance", "t": "Time"},
        "expected": "meter / second",
    },
}


def extract_expression_variables(expression_str: str) -> Tuple[List[str], Optional[str]]:
    """Safely extract variable identifiers present in an expression using restricted AST parsing.

    Returns:
        (list_of_variable_identifiers, error_message_if_any)
    """
    cleaned = expression_str.strip()
    if not cleaned:
        return [], None
    try:
        tree = ast.parse(cleaned, mode="exec")
    except SyntaxError as e:
        return [], f"Syntax error: {e.msg}"

    if len(tree.body) != 1:
        return [], "Expression must contain a single statement or equation."

    stmt = tree.body[0]
    if not isinstance(stmt, (ast.Expr, ast.Assign)):
        return [], f"Unsupported statement type '{type(stmt).__name__}'. Only expressions or equations are supported."

    # Preserve appearance order while deduplicating
    vars_found: List[str] = []
    for node in ast.walk(stmt):
        if isinstance(node, ast.Name):
            if node.id not in vars_found:
                vars_found.append(node.id)
    return vars_found, None


def init_session_state() -> None:
    """Initialize session state variables with clean defaults."""
    if "expression_input" not in st.session_state:
        st.session_state["expression_input"] = "E = m * c ** 2"
    if "expected_unit_input" not in st.session_state:
        st.session_state["expected_unit_input"] = ""
    if "user_units" not in st.session_state:
        st.session_state["user_units"] = {"E": "joule", "m": "kilogram", "c": "meter / second"}
    if "variable_meanings" not in st.session_state:
        st.session_state["variable_meanings"] = {
            "E": "Rest energy",
            "m": "Rest mass",
            "c": "Speed of light in vacuum",
        }
    if "confirmed_flags" not in st.session_state:
        st.session_state["confirmed_flags"] = {"E": True, "m": True, "c": True}
    if "unit_sources" not in st.session_state:
        st.session_state["unit_sources"] = {"E": "Confirmed assumption", "m": "Confirmed assumption", "c": "Confirmed assumption"}
    if "ai_result" not in st.session_state:
        st.session_state["ai_result"] = None
    if "check_result" not in st.session_state:
        st.session_state["check_result"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None


def apply_example(example_key: str) -> None:
    """Load a built-in physics example into session state."""
    if example_key in BUILTIN_EXAMPLES and BUILTIN_EXAMPLES[example_key]["expr"]:
        data = BUILTIN_EXAMPLES[example_key]
        st.session_state["expression_input"] = data["expr"]
        st.session_state["expected_unit_input"] = data.get("expected", "")
        st.session_state["user_units"] = dict(data["units"])
        st.session_state["confirmed_flags"] = {k: True for k in data["units"]}
        st.session_state["unit_sources"] = {k: "Built-in example" for k in data["units"]}

        meanings = data.get("meanings", {})
        st.session_state["variable_meanings"] = {
            k: meanings.get(k, COMMON_VARIABLE_METADATA.get(k, {}).get("meaning", "Meaning not provided"))
            for k in data["units"]
        }
        st.session_state["check_result"] = None
        st.session_state["ai_result"] = None


def main() -> None:
    st.set_page_config(
        page_title="PhysCheck - Physics and Dimensional-Consistency Checker",
        layout="wide",
    )
    init_session_state()

    # =========================================================================
    # Header: Clean, minimal scientific aesthetic (no decorative emojis/essays)
    # =========================================================================
    st.title("PhysCheck")
    st.caption("Physics and dimensional-consistency checker for scientific expressions.")
    st.write(
        "Enter a mathematical expression, define the variables and their units, "
        "and check whether the expression is dimensionally consistent."
    )
    st.caption(
        "Scope: checks supported mathematical expressions. It does not prove that a complete physical model is correct."
    )
    st.divider()

    # =========================================================================
    # Section A — Choose an example or enter an expression
    # =========================================================================
    st.subheader("Section A — Choose an example or enter an expression")

    col_example, col_load = st.columns([4, 1])
    with col_example:
        selected_example = st.selectbox(
            "Load a built-in example",
            options=list(BUILTIN_EXAMPLES.keys()),
            index=0,
            key="example_selector",
            help="Select a supported physical formula or diagnostic failure example.",
        )
    with col_load:
        st.write("")  # Spacing alignment
        st.write("")
        if st.button("Load example", use_container_width=True):
            if selected_example != "Select an example...":
                apply_example(selected_example)
                st.rerun()

    expr_value = st.text_area(
        "Mathematical expression or equation",
        value=st.session_state["expression_input"],
        height=85,
        help="Enter an expression or equation. Examples: 'distance / time', '0.5 * mass * velocity**2', 'E = m * c ** 2'.",
        key="expr_input_area",
    )
    st.session_state["expression_input"] = expr_value
    st.caption(
        "Supported syntax: numeric constants, named variables, addition (+), subtraction (-), "
        "multiplication (*), division (/), parentheses, and dimensionless powers (**)."
    )

    expected_u_val = st.text_input(
        "Optional expected unit for single expression (leave blank if checking an equation)",
        value=st.session_state["expected_unit_input"],
        placeholder="e.g. meter / second, joule, newton",
        help="If provided, validates that the evaluated expression matches this target physical unit.",
        key="expected_unit_text_input",
    )
    st.session_state["expected_unit_input"] = expected_u_val

    # Upload source file (.py) with clear scope and size restrictions
    with st.expander("Upload source file (.py)", expanded=False):
        st.caption(
            f"Accepted format: `.py` files only up to {MAX_UPLOAD_SIZE_BYTES // 1024} KB. "
            "Uploaded code is inspected as plain text only within the supported expression scope. "
            "Uploaded code is never executed."
        )
        uploaded_file = st.file_uploader(
            "Select Python file",
            type=["py"],
            accept_multiple_files=False,
            key="py_file_input",
        )
        if uploaded_file is not None:
            if uploaded_file.size > MAX_UPLOAD_SIZE_BYTES:
                st.error(
                    f"File size ({uploaded_file.size / 1024:.1f} KB) exceeds the limit of "
                    f"{MAX_UPLOAD_SIZE_BYTES // 1024} KB. Upload rejected."
                )
            else:
                raw_bytes = uploaded_file.getvalue()
                try:
                    code_text = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    code_text = raw_bytes.decode("latin-1", errors="replace")

                st.info(f"Loaded '{uploaded_file.name}' ({len(code_text)} characters) safely as static text.")
                code_lines = [
                    line.strip()
                    for line in code_text.splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
                if code_lines:
                    first_line = code_lines[0]
                    if st.button(f"Use first statement line: '{first_line[:40]}...'"):
                        st.session_state["expression_input"] = first_line
                        st.rerun()

    st.divider()

    # =========================================================================
    # Section B — Define the variables
    # =========================================================================
    st.subheader("Section B — Define the variables")
    st.write(
        "Variables present in the current expression are identified automatically. "
        "Define the physical meaning and unit for each variable."
    )

    # Automatically extract real variables from current expression via AST
    active_vars, parse_error = extract_expression_variables(st.session_state["expression_input"])

    if parse_error:
        st.error(
            f"Invalid expression syntax: {parse_error}. "
            "Please check operators, parentheses, or variable identifiers."
        )
    elif not active_vars:
        if st.session_state["expression_input"].strip():
            st.info("This expression contains only numeric constants. No variables are required.")
        else:
            st.info("Enter an expression above to define variable assumptions.")
    else:
        # Synchronize active variables into session state without deleting confirmed data
        for v in active_vars:
            if v not in st.session_state["user_units"]:
                meta = COMMON_VARIABLE_METADATA.get(v, {})
                st.session_state["user_units"][v] = meta.get("example", "")
                st.session_state["variable_meanings"][v] = meta.get("meaning", "Meaning not provided")
                # Defaults start unconfirmed unless user confirms them
                st.session_state["confirmed_flags"][v] = bool(meta.get("example", ""))
                st.session_state["unit_sources"][v] = "Default physical symbol" if meta.get("example") else "Manual input"

        # Table header
        header_col1, header_col2, header_col3, header_col4, header_col5 = st.columns([1.5, 3, 2.5, 2, 1])
        header_col1.markdown("**Variable**")
        header_col2.markdown("**Meaning**")
        header_col3.markdown("**Unit**")
        header_col4.markdown("**Status**")
        header_col5.markdown("**Action**")

        unconfirmed_vars = []
        missing_unit_vars = []

        for var_name in active_vars:
            current_unit = st.session_state["user_units"].get(var_name, "")
            current_meaning = st.session_state["variable_meanings"].get(var_name, "Meaning not provided")
            is_confirmed = st.session_state["confirmed_flags"].get(var_name, False)
            source_info = st.session_state["unit_sources"].get(var_name, "Manual input")

            if not current_unit.strip():
                missing_unit_vars.append(var_name)
            elif not is_confirmed:
                unconfirmed_vars.append(var_name)

            col1, col2, col3, col4, col5 = st.columns([1.5, 3, 2.5, 2, 1])

            # Column 1: Variable identifier
            col1.code(var_name, language=None)

            # Column 2: Meaning (editable)
            edited_meaning = col2.text_input(
                f"Meaning of {var_name}",
                value=current_meaning,
                key=f"meaning_input_{var_name}",
                label_visibility="collapsed",
            )
            if edited_meaning != current_meaning:
                st.session_state["variable_meanings"][var_name] = edited_meaning

            # Column 3: Unit (editable)
            meta = COMMON_VARIABLE_METADATA.get(var_name, {})
            placeholder_text = f"e.g. {meta['example']}" if meta.get("example") else "Unit required (e.g. meter)"
            edited_unit = col3.text_input(
                f"Unit for {var_name}",
                value=current_unit,
                placeholder=placeholder_text,
                key=f"unit_input_{var_name}",
                label_visibility="collapsed",
            )
            if edited_unit != current_unit:
                st.session_state["user_units"][var_name] = edited_unit.strip()
                st.session_state["confirmed_flags"][var_name] = True
                st.session_state["unit_sources"][var_name] = "User specified"

            # Column 4: Status
            if not current_unit.strip():
                col4.caption("Unit required")
            elif is_confirmed:
                col4.caption(f"Confirmed ({source_info})")
            else:
                col4.caption(f"Suggestion: {source_info}")

            # Column 5: Action (Confirm button)
            if current_unit.strip() and not is_confirmed:
                if col5.button("Confirm", key=f"confirm_btn_{var_name}", use_container_width=True):
                    st.session_state["confirmed_flags"][var_name] = True
                    st.session_state["unit_sources"][var_name] = "User confirmed"
                    st.rerun()
            else:
                col5.write("")

        # Bulk confirmation action
        if unconfirmed_vars and not missing_unit_vars:
            if st.button("Confirm all suggested units", use_container_width=False):
                for v in unconfirmed_vars:
                    st.session_state["confirmed_flags"][v] = True
                    st.session_state["unit_sources"][v] = "User confirmed"
                st.rerun()

        # Optional AI Assistant integration (Gemma 4)
        st.write("")
        ai_col1, ai_col2 = st.columns([2, 3])
        with ai_col1:
            analyze_ai = st.button(
                "Propose meanings and units with Gemma 4",
                help="Queries Gemma 4 to propose variable interpretations and candidate units. All proposals remain unconfirmed suggestions until you confirm them.",
                use_container_width=True,
            )
        with ai_col2:
            st.caption(
                "Optional AI assistance. Proposals are marked as unconfirmed suggestions and do not replace confirmed units."
            )

        if analyze_ai:
            if not st.session_state["expression_input"].strip():
                st.warning("Please enter an expression before requesting AI analysis.")
            else:
                with st.spinner(f"Requesting physical reasoning from {DEFAULT_MODEL_IDENTIFIER}..."):
                    try:
                        analyzer = AIAnalyzer()
                        ai_result = analyzer.analyze_expression(st.session_state["expression_input"])
                        st.session_state["ai_result"] = ai_result

                        # Merge candidate units as UNCONFIRMED suggestions
                        for v, u in ai_result.candidate_units.items():
                            if v in active_vars:
                                if not st.session_state["confirmed_flags"].get(v, False) or not st.session_state["user_units"].get(v):
                                    st.session_state["user_units"][v] = u
                                    st.session_state["confirmed_flags"][v] = False
                                    st.session_state["unit_sources"][v] = f"Gemma 4 ({DEFAULT_MODEL_IDENTIFIER})"

                        # Merge meanings if available
                        for vm in ai_result.variable_meanings:
                            if vm.variable in active_vars and vm.meaning:
                                if st.session_state["variable_meanings"].get(vm.variable) in ("", "Meaning not provided"):
                                    st.session_state["variable_meanings"][vm.variable] = vm.meaning

                        st.success("Gemma 4 analysis completed. Suggested candidate units and meanings are displayed above for confirmation.")
                        st.rerun()
                    except MissingAPIKeyError:
                        st.warning(
                            "What was detected: GEMINI_API_KEY environment variable is not set.\n\n"
                            "Why it matters: AI suggestions from Gemma 4 require a configured Google GenAI API key.\n\n"
                            "What to do next: Define GEMINI_API_KEY in your .env file. Deterministic checks remain fully available offline."
                        )
                    except AuthenticationError as e:
                        st.error(f"Authentication failed: {e}. Please check that your GEMINI_API_KEY is valid.")
                    except UnsupportedModelError as e:
                        st.error(f"Model unavailable: {e}. PhysCheck does not substitute alternative models.")
                    except QuotaExceededError as e:
                        st.error(f"API quota exceeded: {e}.")
                    except AIAnalyzerTimeoutError as e:
                        st.error(f"Request timed out: {e}.")
                    except MalformedResponseError as e:
                        st.error(f"Model response was malformed: {e}.")
                    except AIAnalysisError as e:
                        st.error(f"AI analysis failed: {e}.")
                    except Exception as e:
                        st.error(f"Unexpected error: {type(e).__name__} - {e}")

    st.divider()

    # =========================================================================
    # Section C — Run the check
    # =========================================================================
    st.subheader("Section C — Run the check")

    # Determine readiness and diagnostics
    can_check = True
    check_blockers: List[str] = []

    if not st.session_state["expression_input"].strip():
        can_check = False
        check_blockers.append("Enter an expression in Section A.")
    elif parse_error:
        can_check = False
        check_blockers.append(f"Fix expression syntax: {parse_error}")
    elif active_vars:
        for v in active_vars:
            unit = st.session_state["user_units"].get(v, "").strip()
            if not unit:
                can_check = False
                check_blockers.append(f"Variable '{v}' requires a physical unit.")
            elif not st.session_state["confirmed_flags"].get(v, False):
                can_check = False
                check_blockers.append(f"Unit for variable '{v}' ('{unit}') is an unconfirmed suggestion. Please click Confirm.")

    check_button_clicked = st.button(
        "Check expression",
        type="primary",
        disabled=not can_check,
        use_container_width=True,
        help="Evaluates the expression for dimensional consistency using Pint and restricted AST parsing.",
    )
    st.caption("PhysCheck evaluates the expression using the variables and units shown above.")

    if not can_check and st.session_state["expression_input"].strip():
        st.warning(
            "**What was detected**: One or more required variable assumptions remain incomplete.\n\n"
            "**Why it matters**: PhysCheck requires confirmed physical units for every variable to verify dimensional consistency.\n\n"
            f"**What to do next**: {check_blockers[0]}"
        )

    # Perform deterministic check when triggered
    if check_button_clicked and can_check:
        exp_u = st.session_state["expected_unit_input"].strip() or None
        # Only pass active variables to checker
        active_units = {v: st.session_state["user_units"][v] for v in active_vars}
        check_result = check_expression(
            expression=st.session_state["expression_input"].strip(),
            variable_units=active_units,
            expected_unit=exp_u,
        )
        st.session_state["check_result"] = check_result

    st.divider()

    # =========================================================================
    # Section D — Show the result
    # =========================================================================
    st.subheader("Section D — Result")

    res: Optional[CheckResult] = st.session_state["check_result"]

    if res is None:
        st.info("Click **Check expression** above to inspect dimensional consistency.")
    else:
        # Status Card
        if res.status == CheckStatus.PASS:
            st.success("### PASS: Dimensional consistency verified")
            st.write(
                "The expression passed the supported check under the displayed variable-unit assumptions."
            )
        elif res.status == CheckStatus.FAIL:
            st.error("### FAIL: Dimensional incompatibility detected")
            st.write(
                "The checker found a physical dimension mismatch in the evaluated expression."
            )
        else:
            st.warning("### UNSUPPORTED: Check outside supported scope")
            st.write(
                "The expression or requested operation cannot be evaluated within the current checker scope."
            )

        # 1. Summary of inputs checked
        st.markdown(f"**Expression checked**: `{st.session_state['expression_input']}`")

        # Table of confirmed variable units used
        if active_vars:
            summary_data = []
            for v in active_vars:
                summary_data.append({
                    "Variable": v,
                    "Meaning": st.session_state["variable_meanings"].get(v, "Meaning not provided"),
                    "Confirmed Unit": st.session_state["user_units"].get(v, ""),
                })
            st.table(summary_data)

        # 2. Mathematical & Dimensional Evidence
        st.markdown("**Evidence**")
        ev = res.evidence
        if res.status == CheckStatus.PASS:
            if ev.get("dimensionality_match") is True and "lhs_dimensionality" in ev:
                st.write(f"- Equation LHS dimension: `{ev.get('lhs_dimensionality')}` ({ev.get('lhs_unit')})")
                st.write(f"- Equation RHS dimension: `{ev.get('rhs_dimensionality')}` ({ev.get('rhs_inferred_unit')})")
                st.write("- Dimensional equality match: **Consistent**")
            elif "expected_dimensionality" in ev:
                st.write(f"- Inferred dimension: `{ev.get('inferred_dimensionality')}` ({ev.get('inferred_unit')})")
                st.write(f"- Expected target dimension: `{ev.get('expected_dimensionality')}` ({ev.get('expected_unit')})")
                st.write("- Target dimension match: **Consistent**")
            else:
                st.write(f"- Evaluated dimension: `{ev.get('inferred_dimensionality')}`")
                st.write(f"- Evaluated unit: `{ev.get('inferred_unit')}`")

            st.markdown(
                "**Plain-language interpretation**: All terms and operations within this expression satisfy "
                "dimensional homogeneity. Addition, subtraction, and equations equate compatible physical dimensions."
            )
            st.info(
                "Notice: Dimensional consistency indicates that algebraic operations between physical dimensions are valid. "
                "It does not prove that numerical coefficients or the underlying physical model are correct."
            )

        elif res.status == CheckStatus.FAIL:
            if "operation" in ev and ev["operation"] in ("addition", "subtraction"):
                st.write(f"- Operation: `{ev.get('operation')}`")
                st.write(f"- Left operand: `{ev.get('left_units')}` (dimension: `{ev.get('left_dimensionality')}`)")
                st.write(f"- Right operand: `{ev.get('right_units')}` (dimension: `{ev.get('right_dimensionality')}`)")
                st.write(f"- Diagnostic: Cannot {ev.get('operation')} incompatible dimensions `{ev.get('left_dimensionality')}` and `{ev.get('right_dimensionality')}`.")
            elif ev.get("dimensionality_match") is False:
                st.write(f"- Left side dimension: `{ev.get('lhs_dimensionality')}` ({ev.get('lhs_unit')})")
                st.write(f"- Right side dimension: `{ev.get('rhs_dimensionality')}` ({ev.get('rhs_inferred_unit')})")
                st.write("- Diagnostic: Equation sides have conflicting physical dimensions.")
            else:
                st.write(f"- Diagnostic details: {res.message}")

            st.markdown(
                "**Plain-language interpretation**: This formula violates the law of dimensional homogeneity. "
                "In physics, quantities with differing dimensions cannot be added, subtracted, or equated."
            )
            st.markdown(
                "**Practical next steps**:\n"
                "1. Verify your derivation to check whether a factor, velocity, or time variable was omitted.\n"
                "2. Check that units assigned in Section B accurately represent each symbol.\n"
                "3. If exponents are used, verify that each exponent is dimensionless."
            )

        else:  # UNSUPPORTED
            st.write(f"- Diagnostic details: {res.message}")
            st.markdown(
                "**Plain-language interpretation**: The checker encountered an operation outside its supported grammar "
                "(such as an unknown function call, division by zero, bitwise operator, or an exponent magnitude exceeding 100)."
            )
            st.markdown(
                "**Practical next steps**:\n"
                "1. Simplify the expression to supported operators (+, -, *, /, **, ()).\n"
                "2. Ensure all variable names are valid identifiers with confirmed units.\n"
                "3. Ensure divisors evaluate to non-zero values."
            )

    st.divider()

    # =========================================================================
    # Section F — Advanced information & Technical details
    # =========================================================================
    with st.expander("About this check (Technical details)", expanded=False):
        st.markdown("#### Dimensional Analysis Methodology")
        st.write(
            "PhysCheck uses Python's standard `ast` module to parse expressions into an abstract syntax tree "
            "under a strictly restricted visitor pattern. Code is never passed to `eval` or `exec`. "
            "Quantities and physical dimensions are verified using **Pint**, which tracks SI base units "
            "(`[length]`, `[mass]`, `[time]`, `[current]`, `[temperature]`, etc.)."
        )

        st.markdown("#### Supported Mathematical Grammar")
        st.markdown(
            "- **Constants**: Real finite numeric integers and floats (e.g. `2`, `0.5`, `1e-3`).\n"
            "- **Variables**: Explicitly supplied identifiers (e.g. `m`, `v`, `t`, `c`).\n"
            "- **Operators**: `+`, `-`, `*`, `/`, unary `+` and `-`, grouping `()`, and powers `**`.\n"
            "- **Powers constraint**: Exponents must be strictly dimensionless finite real numbers ($|exp| \\le 100$)."
        )

        st.markdown("#### Disclaimer")
        st.write(PASS_DISCLAIMER)

        if st.session_state["ai_result"] is not None:
            st.markdown("#### Gemma 4 Reasoning Context")
            air: AIAnalysisResult = st.session_state["ai_result"]
            st.write(f"**Model**: `{air.model_identifier}`")
            st.write(f"**Summary**: {air.summary}")
            st.write(f"**Uncertainty / Ambiguity**: {air.uncertainty}")
            st.write(f"**Physical Assumptions**: {', '.join(air.assumptions) if air.assumptions else 'None stated'}")
            st.write(f"**Potential Issues**: {', '.join(air.potential_issues) if air.potential_issues else 'None detected'}")
            st.write(f"**Formulation Limitations**: {', '.join(air.limitations) if air.limitations else 'None stated'}")

        if res is not None:
            st.markdown("#### Raw Checker Evidence (JSON)")
            st.json(res.to_dict())


if __name__ == "__main__":
    main()
