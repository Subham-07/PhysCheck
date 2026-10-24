---
name: physics-law-auditor
description: Audits physical laws, equations, conservation constraints, and dimensional consistency using Pint and restricted AST evaluation.
---

# Physics Law Auditor

The **Physics Law Auditor** skill verifies the mathematical and physical validity of scientific expressions, formulas, and equations. It combines deterministic unit and dimensional analysis (using [Pint](https://pint.readthedocs.io/)) with restricted AST parsing to guarantee that physical equations satisfy dimensional homogeneity, conservation constraints, and basic sanity bounds.

---

## 1. When to Use This Skill

Activate or invoke this skill whenever an agent, workflow, or user needs to:
- Verify that a physical formula or equation is dimensionally consistent before using it in calculations or simulations.
- Check that terms in additions or subtractions share identical physical dimensions (e.g., verifying that lengths are not added to durations).
- Validate that powers and exponents in physical relationships are dimensionless and within safe operational bounds.
- Verify dimensional compatibility between the Left-Hand Side (LHS) and Right-Hand Side (RHS) of a physical law (e.g., $E = mc^2$, $F = ma$, $v = d/t$).
- Check whether an expression's output matches an expected target unit.
- Guard against unsafe code execution when evaluating untrusted user-submitted or LLM-generated equations.

---

## 2. When NOT to Use This Skill

Do **not** use this skill for:
- Full differential equation solving or multi-variable numerical simulations (use dedicated numerical solvers like SciPy or SUNDIALS).
- Evaluating arbitrary Python code or scripts (this skill strictly prohibits code execution).
- Transcendental functions with dimensional arguments (e.g., $\sin(x)$ requires dimensionless angle $x$, but general function calls are outside the current restricted grammar).
- Statistical regression or machine learning model training.

---

## 3. Supported Expressions & Input Assumptions

The skill accepts expressions adhering to a strict, documented mathematical subset:

### Supported Syntax
- **Numeric constants**: Real finite integers and floating-point literals (e.g., `2`, `0.5`, `1e-3`, `3.14159`).
- **Named variables**: Identifiers explicitly accompanied by physical units (e.g., `v`, `m`, `c`, `t`, `rho`).
- **Basic arithmetic operators**:
  - Addition (`+`) and subtraction (`-`)
  - Multiplication (`*`) and division (`/`)
  - Unary positive (`+x`) and negation (`-x`)
- **Grouping**: Parentheses (`(` and `)`) to control precedence.
- **Equations**:
  - Assignment equations: `v = d / t`
  - Equality equations: `v == d / t`
- **Carefully restricted powers (`**`)**:
  - Base can have physical dimensions or be dimensionless.
  - **Exponent MUST be strictly dimensionless**.
  - Exponent magnitude must be finite with $|exp| \le 100$ to prevent denial-of-service / overflow.
  - Division by zero in powers (e.g., $0^{-n}$) is strictly disallowed.

### Input Assumptions
- All variables used in the expression must be explicitly declared in the supplied unit mapping.
- All unit names must be valid Pint unit definitions (e.g., `meter`, `second`, `kilogram`, `joule`, `newton`, `watt`).

---

## 4. Unit Confirmation Requirements

PhysCheck strictly differentiates between **AI-proposed candidate units** and **user-confirmed units**:

1. **AI Proposals are Preliminary Suggestions Only**:
   - Model-generated units from Gemma 4 (`gemma-4-31b-it`) or any LLM must be treated as untrusted suggestions.
   - Initial state flag: `is_confirmed_by_user = False`.
2. **Explicit User Confirmation**:
   - Automated workflows and users must review and confirm candidate units before passing them to the deterministic checker as authoritative.
   - If an automated pipeline invokes verification, unit assumptions must be explicitly recorded in the verification report.
3. **No Unconfirmed Unit Propagation**:
   - Calling `get_confirmed_units()` on unconfirmed results will raise an error, preventing accidental reliance on hallucinations.

---

## 5. Deterministic Checker Architecture & Mechanics

The deterministic checker ([checker.py](file:///D:/PhysCheck/checker.py)) operates on the following core principles:

1. **Restricted AST Parsing**:
   - Uses Python's built-in `ast.parse` in a restricted visitor mode.
   - **Zero use of `eval` or `exec`**.
   - Input strings are parsed into syntax trees; non-whitelisted nodes (function calls, imports, comprehensions, attributes, subscripts, bitwise operators) are immediately rejected.
2. **Never Executes Source Code**:
   - Uploaded or pasted code is read solely as static text.
3. **Pint Dimensional Registry**:
   - Constructs `pint.Quantity` objects for each variable and evaluates the expression tree.
   - Enforces dimensional homogeneity on addition ($[A] == [B]$) and subtraction.
   - Multiplies and divides dimensional units algebraically ($[A] \cdot [B]$, $[A] / [B]$).
4. **Structured Result**:
   - Produces a [`CheckResult`](file:///D:/PhysCheck/checker.py#L52-L73) containing `status`, `check_performed`, `evidence`, `assumptions`, and `message`.

---

## 6. Interpreting Results (PASS, FAIL, UNSUPPORTED)

| Status | Exact Meaning | Action Required |
| :--- | :--- | :--- |
| **`PASS`** | The supported check **passed under the supplied units**. All additions/subtractions are dimensionally compatible, powers are dimensionless, and equation sides match in dimension. | Proceed with the calculation. **Notice**: PASS does *not* mean the overall scientific theory or derivation is correct—only that it is dimensionally consistent. |
| **`FAIL`** | A **dimensional incompatibility or constraint violation** occurred (e.g., adding meters to seconds, dimensioned exponent, or LHS dimension $\neq$ RHS dimension). | The formula contains a dimensional error. Review the derivation, exponent units, or variable definitions. |
| **`UNSUPPORTED`** | The input contains syntax or operations outside the allowed subset (e.g., function calls `sin()`, bitwise operators, unknown variables, invalid units, division by zero, or exponents $> 100$). | Simplify the expression, ensure all variables have defined units, or verify syntax. |

### Mandatory Disclaimer
Every `PASS` result carries the strict assumption notice:
> *"PASS means only that the supported check passed under the supplied units; it does not mean the whole scientific program is correct."*

---

## 7. CLI Verification Script (`scripts/verify_math.py`)

The skill provides an automated CLI script to audit equations without launching the web UI.

### Location
[scripts/verify_math.py](file:///D:/PhysCheck/scripts/verify_math.py)

### Invocation Syntax
```powershell
.\.venv\Scripts\python.exe scripts/verify_math.py "<expression>" -u "<units>" [-x "<expected_unit>"] [--json]
```

### Exit Codes
- `0`: Check `PASS`
- `1`: Check `FAIL` (dimensional violation)
- `2`: `UNSUPPORTED` or invalid input / CLI arguments

---

## 8. Documented Usage Examples

### Example 1: Verifying Mass-Energy Equivalence ($E = mc^2$)
```powershell
.\.venv\Scripts\python.exe scripts/verify_math.py "E = m * c ** 2" -u "E=joule,m=kilogram,c=meter/second" --json
```
**Output**:
```json
{
  "status": "PASS",
  "check_performed": "Dimensional equality check: 'E' ([mass] * [length] ** 2 / [time] ** 2) == RHS ([mass] * [length] ** 2 / [time] ** 2)",
  "evidence": {
    "lhs_variable": "E",
    "lhs_unit": "joule",
    "lhs_dimensionality": "[mass] * [length] ** 2 / [time] ** 2",
    "rhs_inferred_unit": "kilogram * meter ** 2 / second ** 2",
    "rhs_dimensionality": "[mass] * [length] ** 2 / [time] ** 2",
    "dimensionality_match": true
  },
  "assumptions": {
    "disclaimer": "PASS means only that the supported check passed under the supplied units; it does not mean the whole scientific program is correct.",
    "variable_units": {
      "E": "joule",
      "m": "kilogram",
      "c": "meter/second"
    }
  },
  "message": "Dimensional check passed: 'E' matches RHS dimensionality"
}
```

### Example 2: Detecting Incompatible Addition (Adding Length to Time)
```powershell
.\.venv\Scripts\python.exe scripts/verify_math.py "d + t" -u "d=meter,t=second" --json
```
**Output**:
```json
{
  "status": "FAIL",
  "check_performed": "Expression internal dimensional compatibility",
  "evidence": {
    "operation": "addition",
    "left_units": "meter",
    "right_units": "second",
    "left_dimensionality": "[length]",
    "right_dimensionality": "[time]"
  },
  "message": "Incompatible addition: cannot add meter (dim: [length]) and second (dim: [time])"
}
```

### Example 3: Rejecting Unsafe Code or Function Calls
```powershell
.\.venv\Scripts\python.exe scripts/verify_math.py "__import__('os').system('echo probe')"
```
**Output**:
```
Status:          [UNSUPPORTED] UNSUPPORTED
Check Performed: Expression AST evaluation
Message:         Unsupported syntax or AST node: Call
Evidence:        {"unsupported_syntax": "Unsupported syntax or AST node: Call", "node_type": "Call"}
```

---

## 9. Limitations & Safety Restrictions

1. **No Code Execution**: Neither `app.py`, `scripts/verify_math.py`, nor `checker.py` will ever run `exec()`, `eval()`, or execute shell code found in inputs.
2. **Dimensionless Powers Only**: Fractional or integer powers are supported only if the exponent evaluates to a dimensionless real number.
3. **No Transcendental Function Evaluation**: Trigonometric, logarithmic, and exponential functions currently trigger `UNSUPPORTED`.
4. **Data Transmission Notice**: When using the optional Gemma 4 integration (`ai_analyzer.py`), expression strings are transmitted to the external Google GenAI endpoint. Deterministic checks via `verify_math.py` and `checker.py` run 100% locally and offline without data transmission.
