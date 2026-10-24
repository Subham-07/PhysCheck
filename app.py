"""PhysCheck - Scientific Physics Law & Dimensional Consistency Auditor.

Guided scientific user interface.
Workflow: Choose input method -> Enter or upload input -> Confirm variables and units -> Check -> Results & Next Actions.
Integrates checker.py and ai_analyzer.py without duplicating backend logic.
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

# File upload security constraint: .py only, maximum 50 KB
MAX_UPLOAD_SIZE_BYTES = 50 * 1024

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

# Verified built-in examples matching test suite and user requirements
BUILTIN_EXAMPLES = {
    "Select an example...": {
        "expr": "",
        "units": {},
        "meanings": {},
        "expected": "",
    },
    "Mass-Energy Equivalence (E = m * c ** 2)": {
        "expr": "E = m * c ** 2",
        "units": {"E": "joule", "m": "kilogram", "c": "meter / second"},
        "meanings": {"E": "Rest energy", "m": "Rest mass", "c": "Speed of light"},
        "expected": "joule",
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
    "Incompatible Addition [FAIL DEMO] (mass + velocity)": {
        "expr": "mass + velocity",
        "units": {"mass": "kilogram", "velocity": "meter / second"},
        "meanings": {"mass": "Mass", "velocity": "Velocity"},
        "expected": "",
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

# Clear one-sentence explanations for what each example demonstrates
EXAMPLE_EXPLANATIONS = {
    "Mass-Energy Equivalence (E = m * c ** 2)": (
        "Demonstrates dimensional equality between rest energy (joules) and mass multiplied by the square of the speed of light."
    ),
    "Velocity (distance / time)": (
        "Demonstrates deriving velocity dimensions ([length] / [time]) by dividing distance in meters by duration in seconds."
    ),
    "Kinetic Energy (0.5 * mass * velocity**2)": (
        "Demonstrates verifying that mass times velocity squared produces energy units, ignoring the dimensionless prefactor 0.5."
    ),
    "Kinetic Energy (K = 0.5 * m * v ** 2)": (
        "Demonstrates checking an equation where kinetic energy equates to half mass times velocity squared."
    ),
    "Kinematics Velocity (v = d / t)": (
        "Demonstrates verifying that velocity in meters per second equals distance in meters divided by time in seconds."
    ),
    "Newton's Second Law (F = m * a)": (
        "Demonstrates verifying that force in newtons equals mass in kilograms multiplied by acceleration in meters per second squared."
    ),
    "Centripetal Acceleration (a = v ** 2 / r)": (
        "Demonstrates verifying that tangential velocity squared divided by orbital radius produces acceleration dimensions."
    ),
    "Gravitational Potential Energy (U = m * g * h)": (
        "Demonstrates verifying that mass times gravitational acceleration times height equates to potential energy in joules."
    ),
    "Incompatible Addition [FAIL DEMO] (mass + velocity)": (
        "Demonstrates detecting a physical error when attempting to add mass ([mass]) to velocity ([length] / [time])."
    ),
    "Incompatible Addition [FAIL DEMO] (a + b)": (
        "Demonstrates flagging an incompatible addition when attempting to add displacement in meters to duration in seconds."
    ),
    "Mismatched Equation [FAIL DEMO] (v = d * t)": (
        "Demonstrates detecting an invalid physics equation where velocity is erroneously equated to distance multiplied by time."
    ),
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


def extract_file_equations(code_text: str) -> List[str]:
    """Safely extract mathematical equations/assignments from Python source text without executing code."""
    if not code_text.strip():
        return []
    try:
        tree = ast.parse(code_text, mode="exec")
    except SyntaxError:
        # Fall back to non-empty non-comment lines
        return [
            line.strip()
            for line in code_text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    equations: List[str] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            if targets:
                try:
                    val_str = ast.unparse(stmt.value)
                    eq_str = f"{targets[0]} = {val_str}"
                    if eq_str not in equations:
                        equations.append(eq_str)
                except Exception:
                    pass
        elif isinstance(stmt, ast.Expr):
            try:
                expr_str = ast.unparse(stmt.value)
                if expr_str not in equations:
                    equations.append(expr_str)
            except Exception:
                pass

    if not equations:
        equations = [
            line.strip()
            for line in code_text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return equations


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
        st.session_state["unit_sources"] = {
            "E": "Example assumption",
            "m": "Example assumption",
            "c": "Example assumption",
        }
    if "ai_result" not in st.session_state:
        st.session_state["ai_result"] = None
    if "check_result" not in st.session_state:
        st.session_state["check_result"] = None
    if "last_checked_expression" not in st.session_state:
        st.session_state["last_checked_expression"] = None
    if "current_method" not in st.session_state:
        st.session_state["current_method"] = "Try an example"
    if "selected_example_name" not in st.session_state:
        st.session_state["selected_example_name"] = "Mass-Energy Equivalence (E = m * c ** 2)"


def apply_example(example_key: str) -> None:
    """Load a built-in physics example into session state."""
    if example_key in BUILTIN_EXAMPLES and BUILTIN_EXAMPLES[example_key]["expr"]:
        data = BUILTIN_EXAMPLES[example_key]
        st.session_state["expression_input"] = data["expr"]
        st.session_state["expected_unit_input"] = data.get("expected", "")
        st.session_state["user_units"] = dict(data["units"])
        st.session_state["confirmed_flags"] = {k: True for k in data["units"]}
        st.session_state["unit_sources"] = {k: "Example assumption" for k in data["units"]}

        meanings = data.get("meanings", {})
        st.session_state["variable_meanings"] = {
            k: meanings.get(k, COMMON_VARIABLE_METADATA.get(k, {}).get("meaning", "Meaning not provided"))
            for k in data["units"]
        }
        st.session_state["check_result"] = None
        st.session_state["ai_result"] = None
        st.session_state["last_checked_expression"] = None


def format_result_diagnostics(res: CheckResult, expression_str: str) -> Dict[str, str]:
    """Generate plain-English diagnostic explanation, reason, and next action based strictly on checker evidence."""
    ev = res.evidence

    if res.status == CheckStatus.PASS:
        what = (
            "All operations within this expression satisfy dimensional homogeneity. "
            "Terms being added, subtracted, or equated share identical physical dimensions."
        )
        why = (
            "In physics, equations must be dimensionally homogeneous. "
            "Quantities can only be equated or added if they share the same physical dimensions."
        )
        next_action = (
            "You can proceed with your calculation under these confirmed assumptions. "
            "Remember that dimensional consistency does not prove that numerical parameters "
            "or the underlying scientific model are correct."
        )

    elif res.status == CheckStatus.FAIL:
        op = ev.get("operation")
        if op in ("addition", "subtraction"):
            l_u = ev.get("left_units", "left term")
            r_u = ev.get("right_units", "right term")
            l_d = ev.get("left_dimensionality", "unknown dimension")
            r_d = ev.get("right_dimensionality", "unknown dimension")
            what = f"Incompatible {op}: attempted to {op} '{l_u}' (dimension {l_d}) and '{r_u}' (dimension {r_d})."
            why = f"In physics, quantities with different physical dimensions ({l_d} and {r_d}) cannot be added or subtracted."
            next_action = "Check your derivation. Ensure both terms share the same physical dimensions before adding or subtracting them."
        elif ev.get("dimensionality_match") is False:
            lhs_d = ev.get("lhs_dimensionality", "unknown")
            rhs_d = ev.get("rhs_dimensionality", "unknown")
            what = f"Dimensional mismatch between equation sides: left side has dimension {lhs_d}, but right side has dimension {rhs_d}."
            why = "An equation states that the left side equals the right side; both sides must therefore represent the exact same physical quantity."
            next_action = "Review the equation definition. Check whether a factor, variable, or power was omitted on either side."
        elif "exponent" in str(ev).lower() or "power" in str(ev).lower():
            what = f"Dimensioned exponent: {res.message}"
            why = "In physics, exponents must be dimensionless pure numbers."
            next_action = "Review the exponent in your formula and ensure all variable units in the exponent cancel out to a dimensionless value."
        else:
            what = f"Dimensional incompatibility: {res.message}"
            why = "The expression violates physical dimensional consistency constraints."
            next_action = "Review the expression algebra and verify the units assigned to each variable."

    else:  # UNSUPPORTED
        if "unknown_variable" in ev or "Unknown variable" in res.message:
            var = ev.get("unknown_variable", "unspecified")
            what = f"Variable missing unit: '{var}' does not have a defined physical unit."
            why = "PhysCheck requires an explicit physical unit for every variable to verify dimensional consistency."
            next_action = "Provide and confirm a valid physical unit for all variables in Step 2."
        elif "division by zero" in res.message.lower() or ev.get("error_type") == "ZeroDivisionError":
            what = "Division by zero detected in expression evaluation."
            why = "Division by zero is mathematically undefined."
            next_action = "Ensure all divisors and denominators evaluate to non-zero values."
        elif "node_type" in ev or "Unsupported syntax" in res.message:
            node = ev.get("node_type", "unsupported syntax")
            what = f"Unsupported syntax ({node}): PhysCheck encountered syntax outside its supported rules."
            why = "PhysCheck restricts syntax to standard arithmetic (+, -, *, /, **, ()) to guarantee safety and avoid code execution."
            next_action = "Simplify the expression to use basic arithmetic operators. Functions like sin(), cos(), or sqrt() are not currently supported."
        else:
            what = f"Unsupported expression: {res.message}"
            why = "The input cannot be evaluated with PhysCheck's currently supported rules."
            next_action = "Simplify the expression or ensure all variable units use standard Pint unit names (e.g. meter, second, kilogram)."

    return {"what": what, "why": why, "next_action": next_action}


def main() -> None:
    st.set_page_config(
        page_title="PhysCheck - Physics and Dimensional-Consistency Checker",
        layout="wide",
    )
    init_session_state()

    # =========================================================================
    # Header: Clean, minimal scientific aesthetic
    # =========================================================================
    st.title("PhysCheck")
    st.caption("Check the dimensional consistency of scientific equations.")
    st.write("Choose an example, check your own equation, or analyze a Python source file.")
    st.divider()

    # =========================================================================
    # Single-choice Selector: Exactly one input method active at a time
    # =========================================================================
    st.markdown("### What would you like to check?")

    method_options = [
        "Try an example",
        "Enter my own equation",
        "Analyze a Python file",
    ]

    selected_method = st.radio(
        "What would you like to check?",
        options=method_options,
        index=method_options.index(st.session_state.get("current_method", "Try an example")),
        key="input_method_selector",
        horizontal=True,
        label_visibility="collapsed",
        help="Select exactly one workflow. Switching methods clears stale results.",
    )

    # Detect method switch to invalidate stale inputs and results
    prev_method = st.session_state.get("current_method", "Try an example")
    if selected_method != prev_method:
        st.session_state["current_method"] = selected_method
        st.session_state["check_result"] = None
        st.session_state["ai_result"] = None
        st.session_state["last_checked_expression"] = None

        if selected_method == "Try an example":
            apply_example("Mass-Energy Equivalence (E = m * c ** 2)")
            st.session_state["selected_example_name"] = "Mass-Energy Equivalence (E = m * c ** 2)"
        elif selected_method == "Enter my own equation":
            st.session_state["expression_input"] = ""
            st.session_state["expected_unit_input"] = ""
            st.session_state["user_units"] = {}
            st.session_state["variable_meanings"] = {}
            st.session_state["confirmed_flags"] = {}
            st.session_state["unit_sources"] = {}
        elif selected_method == "Analyze a Python file":
            st.session_state["expression_input"] = ""
            st.session_state["expected_unit_input"] = ""
            st.session_state["user_units"] = {}
            st.session_state["variable_meanings"] = {}
            st.session_state["confirmed_flags"] = {}
            st.session_state["unit_sources"] = {}
        st.rerun()

    # =========================================================================
    # Step 1 — Input (Strictly Mutually Exclusive)
    # =========================================================================
    if selected_method == "Try an example":
        st.markdown("### Step 1 — Input: Select an example")

        example_keys = [k for k in BUILTIN_EXAMPLES.keys() if k != "Select an example..."]
        current_ex_name = st.session_state.get("selected_example_name", "Mass-Energy Equivalence (E = m * c ** 2)")
        if current_ex_name not in example_keys:
            current_ex_name = example_keys[0]

        chosen_example = st.selectbox(
            "Choose a verified physics example formulation",
            options=example_keys,
            index=example_keys.index(current_ex_name),
            key="example_selector_dropdown",
            help="Select a supported physical formula or failure demonstration.",
        )

        if chosen_example != st.session_state.get("selected_example_name"):
            st.session_state["selected_example_name"] = chosen_example
            apply_example(chosen_example)
            st.rerun()

        # One-sentence explanation of what the example demonstrates
        st.caption(EXAMPLE_EXPLANATIONS.get(chosen_example, "Preconfigured scientific equation example."))

        # Display the loaded expression
        st.text_area(
            "Mathematical expression for this example",
            value=st.session_state["expression_input"],
            height=70,
            disabled=True,
            help="Formula loaded from the selected example.",
            key="example_expr_display_area",
        )

    elif selected_method == "Enter my own equation":
        st.markdown("### Step 1 — Input: Enter your equation")

        manual_expr = st.text_area(
            "Enter your mathematical expression",
            value=st.session_state["expression_input"],
            placeholder="distance / time",
            height=70,
            help="Enter a mathematical expression or equation, e.g. distance / time or E = m * c ** 2.",
            key="manual_expr_input_area",
        )

        # Clear result if expression was changed by user
        if manual_expr != st.session_state["expression_input"]:
            st.session_state["expression_input"] = manual_expr
            st.session_state["check_result"] = None
            st.session_state["last_checked_expression"] = None

        st.caption(
            "Supported syntax: numeric constants, named variables, addition (+), subtraction (-), "
            "multiplication (*), division (/), parentheses, and dimensionless powers (**)."
        )

        expected_u = st.text_input(
            "Optional target unit (leave blank if checking an equation)",
            value=st.session_state.get("expected_unit_input", ""),
            placeholder="e.g. meter / second, joule, newton",
            help="If provided, validates that a single expression matches this target physical unit.",
            key="manual_expected_unit_input",
        )
        if expected_u != st.session_state.get("expected_unit_input", ""):
            st.session_state["expected_unit_input"] = expected_u
            st.session_state["check_result"] = None

    elif selected_method == "Analyze a Python file":
        st.markdown("### Step 1 — Input: Upload a Python file")
        st.caption(
            f"Supported file format: `.py` files only (maximum {MAX_UPLOAD_SIZE_BYTES // 1024} KB). "
            "PhysCheck inspects source code as plain text using restricted AST parsing to identify "
            "mathematical assignment statements and equations. Uploaded code is strictly inspected and never executed."
        )

        uploaded_file = st.file_uploader(
            "Select Python source file (.py)",
            type=["py"],
            accept_multiple_files=False,
            key="py_file_upload_control",
            help=f"Select a Python script up to {MAX_UPLOAD_SIZE_BYTES // 1024} KB.",
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

                eqs = extract_file_equations(code_text)
                if not eqs:
                    st.warning(
                        "No supported mathematical assignments or equations were found in this file. "
                        "Ensure your file contains statements such as 'v = d / t' or 'E = m * c ** 2'."
                    )
                else:
                    st.info(f"Loaded '{uploaded_file.name}' safely as static text ({len(eqs)} statement(s) found).")
                    chosen_file_eq = st.selectbox(
                        "Select statement from file to check",
                        options=eqs,
                        key="file_eq_choice_dropdown",
                    )
                    if chosen_file_eq != st.session_state.get("expression_input"):
                        st.session_state["expression_input"] = chosen_file_eq
                        st.session_state["check_result"] = None
                        st.session_state["last_checked_expression"] = None

                    st.text_area(
                        "Statement to check",
                        value=st.session_state["expression_input"],
                        height=70,
                        disabled=True,
                        key="file_expr_display_area",
                    )
        else:
            st.info("Upload a Python (.py) file above to inspect its equations.")

    # =========================================================================
    # Step 2 — Confirm Variables and Units
    # (Displayed only when the input is valid enough to inspect)
    # =========================================================================
    current_expr = st.session_state.get("expression_input", "").strip()
    active_vars: List[str] = []
    parse_error: Optional[str] = None

    if current_expr:
        active_vars, parse_error = extract_expression_variables(current_expr)

        if parse_error:
            st.error(
                f"Cannot inspect variables: {parse_error}. "
                "Please correct the expression syntax in Step 1."
            )
        else:
            st.divider()
            st.markdown("### Step 2 — Confirm variables and units")
            st.write("PhysCheck needs to know what each variable represents and which physical unit it uses.")

            if not active_vars:
                st.info("This expression contains only numeric constants. No variable units are required.")
            else:
                # Ensure active variables exist in session state
                for v in active_vars:
                    if v not in st.session_state["user_units"]:
                        meta = COMMON_VARIABLE_METADATA.get(v, {})
                        st.session_state["user_units"][v] = meta.get("example", "")
                        st.session_state["variable_meanings"][v] = meta.get("meaning", "Meaning not provided")
                        st.session_state["confirmed_flags"][v] = False
                        st.session_state["unit_sources"][v] = "Common symbol suggestion" if meta.get("example") else "Manual entry"

                # Table of variables
                header_col1, header_col2, header_col3, header_col4, header_col5 = st.columns([1.5, 3, 2.5, 2.5, 1.2])
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
                    source_info = st.session_state["unit_sources"].get(var_name, "Manual entry")

                    if not current_unit.strip():
                        missing_unit_vars.append(var_name)
                    elif not is_confirmed:
                        unconfirmed_vars.append(var_name)

                    col1, col2, col3, col4, col5 = st.columns([1.5, 3, 2.5, 2.5, 1.2])

                    # Variable Name
                    col1.code(var_name, language=None)

                    # Meaning
                    edited_meaning = col2.text_input(
                        f"Meaning of {var_name}",
                        value=current_meaning,
                        key=f"meaning_input_{var_name}",
                        label_visibility="collapsed",
                    )
                    if edited_meaning != current_meaning:
                        st.session_state["variable_meanings"][var_name] = edited_meaning

                    # Unit
                    meta = COMMON_VARIABLE_METADATA.get(var_name, {})
                    placeholder_text = f"e.g. {meta['example']}" if meta.get("example") else "e.g. meter"
                    edited_unit = col3.text_input(
                        f"Unit for {var_name}",
                        value=current_unit,
                        placeholder=placeholder_text,
                        key=f"unit_input_{var_name}",
                        label_visibility="collapsed",
                    )
                    if edited_unit != current_unit:
                        st.session_state["user_units"][var_name] = edited_unit.strip()
                        st.session_state["confirmed_flags"][var_name] = bool(edited_unit.strip())
                        st.session_state["unit_sources"][var_name] = "User defined"
                        st.session_state["check_result"] = None

                    # Confirmation Status
                    if not current_unit.strip():
                        col4.caption("Unit required")
                    elif is_confirmed:
                        col4.caption(f"Confirmed ({source_info})")
                    else:
                        col4.caption(f"Suggestion: {source_info} (unconfirmed)")

                    # Action Button
                    if current_unit.strip() and not is_confirmed:
                        if col5.button("Confirm", key=f"confirm_var_{var_name}", use_container_width=True):
                            st.session_state["confirmed_flags"][var_name] = True
                            st.session_state["unit_sources"][var_name] = "User confirmed"
                            st.session_state["check_result"] = None
                            st.rerun()
                    else:
                        col5.write("")

                # Bulk confirmation
                if unconfirmed_vars and not missing_unit_vars:
                    if st.button("Confirm all suggested units", key="bulk_confirm_btn"):
                        for v in unconfirmed_vars:
                            st.session_state["confirmed_flags"][v] = True
                            st.session_state["unit_sources"][v] = "User confirmed"
                        st.session_state["check_result"] = None
                        st.rerun()

                # Optional AI Assistance with Gemma 4
                st.write("")
                ai_col1, ai_col2 = st.columns([2.5, 3.5])
                with ai_col1:
                    analyze_ai = st.button(
                        "Suggest meanings and units with Gemma 4",
                        help="Queries Gemma 4 to suggest variable interpretations and candidate units. All proposals remain unconfirmed suggestions until you confirm them.",
                        use_container_width=True,
                        key="gemma_suggest_btn",
                    )
                with ai_col2:
                    st.caption(
                        "Optional AI assistance. Candidate units are marked as unconfirmed suggestions and do not replace confirmed units."
                    )

                if analyze_ai:
                    with st.spinner(f"Requesting physical reasoning from {DEFAULT_MODEL_IDENTIFIER}..."):
                        try:
                            analyzer = AIAnalyzer()
                            ai_res = analyzer.analyze_expression(current_expr)
                            st.session_state["ai_result"] = ai_res

                            # Merge candidate units as UNCONFIRMED suggestions
                            for v, u in ai_res.candidate_units.items():
                                if v in active_vars:
                                    if not st.session_state["confirmed_flags"].get(v, False) or not st.session_state["user_units"].get(v):
                                        st.session_state["user_units"][v] = u
                                        st.session_state["confirmed_flags"][v] = False
                                        st.session_state["unit_sources"][v] = f"Gemma 4 ({DEFAULT_MODEL_IDENTIFIER})"

                            # Merge meanings if available
                            for vm in ai_res.variable_meanings:
                                if vm.variable in active_vars and vm.meaning:
                                    if st.session_state["variable_meanings"].get(vm.variable) in ("", "Meaning not provided"):
                                        st.session_state["variable_meanings"][vm.variable] = vm.meaning

                            st.success("Gemma 4 suggestions received. Review the candidate units above and click Confirm.")
                            st.rerun()
                        except MissingAPIKeyError:
                            st.warning(
                                "What was detected: GEMINI_API_KEY environment variable is not set.\n\n"
                                "Why it matters: AI suggestions require a Google GenAI API key in .env.\n\n"
                                "What to do next: Define GEMINI_API_KEY in your .env file, or type units manually. Deterministic checks work 100% offline."
                            )
                        except AuthenticationError as e:
                            st.error(f"Authentication failed: {e}. Check that your GEMINI_API_KEY is valid.")
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

    # =========================================================================
    # Step 3 — Check Action
    # =========================================================================
    st.divider()
    st.markdown("### Step 3 — Check")

    can_check = True
    check_blockers: List[str] = []

    if not current_expr:
        can_check = False
        check_blockers.append("Please provide an expression or upload a file in Step 1.")
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

    primary_action_label = "Analyze file" if selected_method == "Analyze a Python file" else "Check expression"

    check_button_clicked = st.button(
        primary_action_label,
        type="primary",
        disabled=not can_check,
        use_container_width=True,
        key="primary_check_button",
        help="Evaluates the expression for dimensional consistency using Pint and restricted AST parsing.",
    )

    if not can_check and current_expr:
        st.info(f"Cannot run check yet: {check_blockers[0]}")

    if check_button_clicked and can_check:
        active_units = {v: st.session_state["user_units"][v] for v in active_vars}
        exp_u = st.session_state.get("expected_unit_input", "").strip() or None
        check_result = check_expression(
            expression=current_expr,
            variable_units=active_units,
            expected_unit=exp_u,
        )
        st.session_state["check_result"] = check_result
        st.session_state["last_checked_expression"] = current_expr
        st.rerun()

    # =========================================================================
    # Step 4 — Results & Suggested Next Actions
    # =========================================================================
    st.divider()
    st.markdown("### Your results")

    res: Optional[CheckResult] = st.session_state.get("check_result")

    # Invalidate stale result if expression changed since last check
    if res is not None and st.session_state.get("last_checked_expression") != current_expr:
        res = None
        st.session_state["check_result"] = None

    if res is None:
        st.info("Your results will appear here after you complete the inputs and run the check.")
    else:
        diag = format_result_diagnostics(res, current_expr)

        if res.status == CheckStatus.PASS:
            st.success("### PASS: Dimensional consistency verified")
            st.write("The expression passed the dimensional-consistency check under the assumptions shown.")

            st.markdown(f"**Expression checked**: `{current_expr}`")

            # Dimensional evidence
            ev = res.evidence
            st.markdown("**Dimensional Evidence**")
            if ev.get("dimensionality_match") is True and "lhs_dimensionality" in ev:
                st.write(f"- Left side dimension: `{ev.get('lhs_dimensionality')}` ({ev.get('lhs_unit')})")
                st.write(f"- Right side dimension: `{ev.get('rhs_dimensionality')}` ({ev.get('rhs_inferred_unit')})")
                st.write("- Dimensional equality match: **Consistent**")
            elif "expected_dimensionality" in ev:
                st.write(f"- Inferred dimension: `{ev.get('inferred_dimensionality')}` ({ev.get('inferred_unit')})")
                st.write(f"- Target dimension: `{ev.get('expected_dimensionality')}` ({ev.get('expected_unit')})")
                st.write("- Target dimension match: **Consistent**")
            else:
                st.write(f"- Evaluated physical dimension: `{ev.get('inferred_dimensionality')}`")
                st.write(f"- Inferred unit: `{ev.get('inferred_unit')}`")

            # 3 guided questions
            st.markdown(f"**1. What did PhysCheck find?**\n{diag['what']}")
            st.markdown(f"**2. Why does it matter?**\n{diag['why']}")
            st.markdown(f"**3. Suggested next action**\n{diag['next_action']}")

        elif res.status == CheckStatus.FAIL:
            st.error("### FAIL: Dimensional problem detected")
            st.write("The checker found a problem with this expression.")

            st.markdown(f"**Expression checked**: `{current_expr}`")

            ev = res.evidence
            st.markdown("**Diagnostic Evidence**")
            if "operation" in ev and ev["operation"] in ("addition", "subtraction"):
                st.write(f"- Operation: `{ev.get('operation')}`")
                st.write(f"- Left term: `{ev.get('left_units')}` (dimension: `{ev.get('left_dimensionality')}`)")
                st.write(f"- Right term: `{ev.get('right_units')}` (dimension: `{ev.get('right_dimensionality')}`)")
            elif ev.get("dimensionality_match") is False:
                st.write(f"- Left side dimension: `{ev.get('lhs_dimensionality')}` ({ev.get('lhs_unit')})")
                st.write(f"- Right side dimension: `{ev.get('rhs_dimensionality')}` ({ev.get('rhs_inferred_unit')})")
            else:
                st.write(f"- Diagnostic detail: {res.message}")

            st.markdown(f"**1. What did PhysCheck find?**\n{diag['what']}")
            st.markdown(f"**2. Why does it matter?**\n{diag['why']}")
            st.markdown(f"**3. Suggested next action**\n{diag['next_action']}")

        else:  # UNSUPPORTED
            st.warning("### UNSUPPORTED: Outside supported rules")
            st.write("PhysCheck cannot check this expression with its currently supported rules.")

            st.markdown(f"**Expression checked**: `{current_expr}`")
            st.write(f"- Limitation detail: {res.message}")

            st.markdown(f"**1. What did PhysCheck find?**\n{diag['what']}")
            st.markdown(f"**2. Why does it matter?**\n{diag['why']}")
            st.markdown(f"**3. Suggested next action**\n{diag['next_action']}")

        # Collapsible technical details
        with st.expander("Technical details", expanded=False):
            st.markdown("#### Methodology")
            st.write(
                "PhysCheck uses Python's standard `ast` module to inspect expressions under a strictly "
                "restricted visitor pattern. Uploaded and entered expressions are never executed (`eval` and `exec` are prohibited). "
                "Units and physical dimensions are tracked deterministically using Pint."
            )
            st.info(PASS_DISCLAIMER)
            st.markdown("#### Raw Result Data")
            st.json(res.to_dict())


if __name__ == "__main__":
    main()
