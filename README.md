# PhysCheck

**PhysCheck** is a scientific software tool and agent skill framework for auditing physical laws, equations, conservation constraints, and dimensional consistency. It combines deterministic dimensional analysis powered by [Pint](https://pint.readthedocs.io/) and restricted Python Abstract Syntax Tree (AST) parsing with optional AI-assisted parameter analysis powered by Google's **Gemma 4** (`gemma-4-31b-it`).

---

## Table of Contents

- [Problem Statement & Overview](#problem-statement--overview)
- [Architecture & Key Features](#architecture--key-features)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Virtual Environment Setup (Windows)](#virtual-environment-setup-windows)
  - [Dependencies](#dependencies)
- [Local API Key Configuration](#local-api-key-configuration)
- [Quick Start](#quick-start)
  - [Web Application (Streamlit)](#web-application-streamlit)
  - [Command-Line Verification Script](#command-line-verification-script)
- [Running Automated Tests](#running-automated-tests)
- [Supported Expression Syntax](#supported-expression-syntax)
- [Verification Results & Interpretation](#verification-results--interpretation)
- [Verified Examples & Expected Results](#verified-examples--expected-results)
- [AI Integration Details (Gemma 4)](#ai-integration-details-gemma-4)
- [Privacy & Data Transmission Notice](#privacy--data-transmission-notice)
- [Limitations & Known Constraints](#limitations--known-constraints)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Problem Statement & Overview

In scientific computing, computational physics, and engineering simulations, equations and mathematical formulations often suffer from subtle dimensional errors—such as inadvertently adding terms of different physical dimensions (e.g., adding a length to a duration), using dimensional exponents, or equating incompatible quantities. Furthermore, when scientists or AI code assistants generate mathematical formulations, the resulting expressions may appear plausible while violating fundamental dimensional homogeneity.

**PhysCheck solves this problem by:**
1. **Deterministically parsing and validating mathematical expressions** against rigorous dimensional and unit constraints using Pint and Python's AST parser.
2. **Strictly forbidding code execution** (`eval` / `exec`) to ensure complete security when inspecting untrusted, user-submitted, or LLM-generated code.
3. **Offering optional AI parameter extraction** via the `gemma-4-31b-it` model to suggest variable meanings and physical units, while enforcing that AI suggestions remain unconfirmed until explicitly reviewed and confirmed by a human user or scientific workflow.

> [!IMPORTANT]
> **Mandatory Scientific Disclaimer**:
> **`PASS` means only that the supported check passed under the supplied units; it does not mean the whole scientific program is correct.** Dimensional consistency is a necessary condition for physical validity, but not a sufficient proof of scientific theory correctness.

---

## Architecture & Key Features

```
PhysCheck/
├── checker.py                 # Deterministic AST and Pint dimensional analysis engine
├── ai_analyzer.py             # Gemma 4 (gemma-4-31b-it) structured parameter analyzer
├── app.py                     # Professional scientific Streamlit web interface
├── requirements.txt           # Verified project dependencies
├── .env.example               # Template for optional API credentials
├── .gitignore                 # Excludes .venv, .env, and test caches
├── LICENSE                    # Standard MIT License
├── README.md                  # Comprehensive documentation and guides
├── scripts/
│   ├── README.md              # Documentation for automation scripts
│   └── verify_math.py         # Real CLI entry point for deterministic checks
├── skills/
│   └── physics-law-auditor/
│       └── SKILL.md           # Agent skill specification for AI agent runtimes
└── tests/
    ├── test_sanity.py         # Environment and dependency sanity tests
    ├── test_checker.py        # 13 tests for AST rules, Pint units, and dimensional logic
    ├── test_ai_analyzer.py    # 13 mocked tests for Gemma 4 integration and error states
    ├── test_app.py            # 14 tests for Streamlit UI workflow and security limits
    └── test_verify_math.py    # 6 end-to-end CLI integration tests
```

### Core Features
- **100% Offline Deterministic Verification**: The core checker (`checker.py`) and CLI script (`scripts/verify_math.py`) require zero internet connection and zero API credentials.
- **Restricted AST Grammar**: AST parser whitelist prevents code injection, arbitrary function execution, imports, and system calls.
- **Pint Unit Integration**: Leverages Pint's dimensional analysis system to handle SI, non-SI, and compound physical dimensions (`[mass]`, `[length]`, `[time]`, etc.).
- **Equation & Expression Support**: Verifies single expressions (e.g., `0.5 * mass * velocity**2`) or equations (e.g., `E = m * c ** 2`, `v == d / t`).
- **Gemma 4 Structured Analysis**: Integration with `gemma-4-31b-it` using Google GenAI SDK (`google-genai`) to infer physical meanings, candidate units, assumptions, and potential issues.
- **Strict Human-in-the-Loop Confirmation**: Units proposed by AI are tagged as tentative suggestions and are never treated as authoritative until confirmed.
- **Clean Scientific UI**: Streamlit interface structured into clear operational sections (A through F), avoiding decorative emojis and visual clutter.

---

## Installation & Setup

### Prerequisites
- **Python**: Version 3.13 or newer (verified on Python 3.13.11).
- **Operating System**: Windows (tested), Linux, or macOS.
- **Git**: For version control.

### Virtual Environment Setup (Windows)

1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd D:\PhysCheck
   ```

2. Create a virtual environment (if not already created):
   ```powershell
   python -m venv .venv
   ```

3. **Execution Policy Notice for Windows**:
   PowerShell script execution policies frequently block execution of `.\.venv\Scripts\Activate.ps1`. You do **not** need to enable script execution if you invoke the Python binary directly:
   ```powershell
   # Use the virtual environment binary directly for all commands:
   .\.venv\Scripts\python.exe <command>
   ```
   *(Optional)* If you prefer activating the environment in PowerShell:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   .\.venv\Scripts\Activate.ps1
   ```

### Dependencies

Install the verified project dependencies from `requirements.txt`:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Current dependencies in `requirements.txt`:
- `python-dotenv>=1.0.1`: Environment variable management.
- `pytest>=8.3.0`: Test runner and assertions.
- `pint>=0.24.0`: Physical quantities, units, and dimensional analysis.
- `google-genai>=0.1.0`: Official Google GenAI SDK for Gemma 4.
- `pydantic>=2.0.0`: Schema definition and validation.
- `streamlit>=1.35.0`: Interactive scientific web dashboard.

---

## Local API Key Configuration

The core dimensional checker is completely functional **without an API key**. An API key is only required if you choose to invoke the optional AI analyzer powered by Gemma 4.

1. Copy the example environment file:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Open `.env` in a text editor and provide your Google Gemini / GenAI API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```

> [!WARNING]
> - Never commit your `.env` file or expose your API key in source code, logs, or bug reports. `.env` is excluded by `.gitignore`.
> - If `GEMINI_API_KEY` is absent or invalid, the Streamlit UI and CLI remain fully usable for offline deterministic checking.

---

## Quick Start

### Web Application (Streamlit)

Launch the interactive scientific interface:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The application will start locally (typically at `http://localhost:8501`).

#### Web Application Workflow:
1. **Section A (Input Formulation)**: Choose a built-in example from the dropdown, enter an expression manually, or upload a `.py` file (plain-text inspection, $\le 50\text{ KB}$).
2. **Section B (Variable Extraction & Unit Assumptions)**: Review the real variables detected in your expression. Enter Pint-compatible units (e.g., `meter`, `second`, `kilogram`).
3. **Section C (Optional AI Analysis)**: Click **Analyze Formulation with Gemma 4** to fetch physical interpretations and proposed units from the `gemma-4-31b-it` model.
4. **Section D (Unit Confirmation)**: Inspect candidate units and click **Confirm Units for Verification** to authorize them for calculation.
5. **Section E (Deterministic Verification)**: Click **Run Deterministic Physics Check** to execute the AST dimensional auditor.
6. **Section F (Results & Diagnostics)**: Inspect the structured result status (`PASS`, `FAIL`, or `UNSUPPORTED`), exact check performed, dimensional evidence, and next steps.

---

### Command-Line Verification Script

The CLI entry point `scripts/verify_math.py` executes the exact same deterministic checker used by the application, suitable for CI/CD pipelines and agent workflows.

#### Syntax:
```powershell
.\.venv\Scripts\python.exe scripts/verify_math.py "<expression>" -u "<units>" [-x "<expected_unit>"] [--json]
```

#### CLI Options:
- `expression`: Positional expression or equation (e.g. `"distance / time"` or `"E = m * c ** 2"`).
- `-u, --units`: Variable units supplied as key-value pairs (`"d=meter,t=second"`) or a JSON string (`'{"d": "meter", "t": "second"}'`).
- `-x, --expected`: *(Optional)* Target unit that a single expression must match.
- `--json`: Format output as machine-readable JSON.

#### CLI Exit Codes:
- `0`: Verification **`PASS`**.
- `1`: Verification **`FAIL`** (dimensional inconsistency detected).
- `2`: Verification **`UNSUPPORTED`** or invalid syntax/arguments.

#### CLI Usage Examples:

**1. Check a single physical expression:**
```powershell
.\.venv\Scripts\python.exe scripts/verify_math.py "distance / time" -u "distance=meter,time=second"
```
*Output:*
```text
Status:          [PASS] PASS
Check Performed: Dimensional consistency check for expression: 'distance / time'
Message:         All operations within the expression are dimensionally consistent
Evidence:        {"inferred_unit": "meter / second", "inferred_dimensionality": "[length] / [time]", "inferred_quantity": "1.0 meter / second"}
Disclaimer:      PASS means only that the supported check passed under the supplied units; it does not mean the whole scientific program is correct.
```

**2. Check mass-energy equivalence with JSON output:**
```powershell
.\.venv\Scripts\python.exe scripts/verify_math.py "E = m * c ** 2" -u "E=joule,m=kilogram,c=meter/second" --json
```
*Output:*
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

**3. Detecting an incompatible addition (Exit Code 1):**
```powershell
.\.venv\Scripts\python.exe scripts/verify_math.py "mass + velocity" -u "mass=kilogram,velocity=meter/second" --json
```
*Output:*
```json
{
  "status": "FAIL",
  "check_performed": "Expression internal dimensional compatibility",
  "evidence": {
    "operation": "addition",
    "left_units": "kilogram",
    "right_units": "meter / second",
    "left_dimensionality": "[mass]",
    "right_dimensionality": "[length] / [time]",
    "left_quantity": "1.0 kilogram",
    "right_quantity": "1.0 meter / second"
  },
  "assumptions": {
    "disclaimer": "PASS means only that the supported check passed under the supplied units; it does not mean the whole scientific program is correct.",
    "variable_units": {
      "mass": "kilogram",
      "velocity": "meter/second"
    }
  },
  "message": "Incompatible addition: cannot add kilogram (dim: [mass]) and meter / second (dim: [length] / [time])"
}
```

---

## Running Automated Tests

PhysCheck includes an automated test suite with 46 tests covering sanity checks, deterministic AST evaluation, Pint dimensional compatibility, mocked Gemma 4 API integrations, CLI behavior, and Streamlit workflows.

Execute the test suite using pytest:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

To run a specific test module:
```powershell
# Run deterministic checker tests:
.\.venv\Scripts\python.exe -m pytest tests/test_checker.py -v

# Run AI analyzer tests (all mocked, no live API key needed):
.\.venv\Scripts\python.exe -m pytest tests/test_ai_analyzer.py -v

# Run CLI integration tests:
.\.venv\Scripts\python.exe -m pytest tests/test_verify_math.py -v

# Run Streamlit interface tests:
.\.venv\Scripts\python.exe -m pytest tests/test_app.py -v
```

---

## Supported Expression Syntax

PhysCheck parses formulas using Python's `ast` module against a strict whitelist of safe mathematical nodes:

### Supported Elements:
- **Numeric constants**: Real integer and floating-point literals (e.g., `0.5`, `2`, `1e-3`, `3.14159`).
- **Named variables**: Identifiers explicitly associated with physical units (e.g., `mass`, `velocity`, `E`, `c`, `t`, `rho`).
- **Basic arithmetic operators**:
  - Addition (`+`) and subtraction (`-`)
  - Multiplication (`*`) and division (`/`)
  - Unary positive (`+x`) and negation (`-x`)
- **Parentheses**: For grouping and precedence control (`(a + b) * c`).
- **Equations**:
  - Assignment equations: `v = d / t`
  - Equality comparisons: `v == d / t`
  - Verifies that LHS dimensionality matches RHS dimensionality.
- **Restricted powers (`**`)**:
  - Base may be dimensioned or dimensionless.
  - **Exponent MUST be strictly dimensionless** (e.g., `x ** 2` is valid; `x ** mass` triggers a `FAIL`).
  - Exponent magnitude must be finite with $|exp| \le 100$ to prevent computational overflow / DoS.
  - Exponents causing division by zero (e.g., $0^{-2}$) are rejected.

### Unsupported Syntax (triggers `UNSUPPORTED`):
- Function calls (`sin(x)`, `sqrt(x)`, `exp(x)`).
- Code execution constructs (`import`, `eval`, `exec`, `open`, `__import__`).
- Object-oriented syntax (attributes `obj.attr`, method calls `x.method()`).
- Data structures and subscripts (lists `[1, 2]`, dictionaries, tuples, indexing `a[0]`).
- Bitwise operators (`&`, `|`, `^`, `<<`, `>>`).
- Control flow (`if`, `for`, `while`, `def`, `class`).

---

## Verification Results & Interpretation

The checker returns one of three structured statuses:

| Status | Meaning | Practical Next Steps |
| :--- | :--- | :--- |
| **`PASS`** | The expression or equation satisfies all dimensional homogeneity rules under the supplied units. All terms in additions/subtractions match dimensions, powers are dimensionless, and equation sides have equivalent dimensionality. | Proceed with calculations. Keep in mind the mandatory disclaimer: dimensional consistency does not prove physical theory correctness. |
| **`FAIL`** | A physical or dimensional inconsistency was detected (e.g., adding mass to velocity, dimensioned exponent, or LHS dimension $\neq$ RHS dimension). | The formula contains a dimensional flaw. Check variable definitions, algebraic derivation, or power units. |
| **`UNSUPPORTED`** | The input contains syntax outside the whitelisted subset, unknown variables, invalid Pint unit names, or division by zero. | Check for missing variable units, typos in unit names, or remove unsupported operators/function calls. |

---

## Verified Examples & Expected Results

The following table summarizes verified physical formulations, unit configurations, and their deterministic audit outcomes:

| Expression | Confirmed Units | Status | Result / Inferred Dimensions |
| :--- | :--- | :--- | :--- |
| `distance / time` | `distance: meter`, `time: second` | **PASS** | Unit: `meter / second`<br>Dim: `[length] / [time]` |
| `0.5 * mass * velocity**2` | `mass: kilogram`, `velocity: meter / second` | **PASS** | Unit: `kilogram * meter ** 2 / second ** 2`<br>Dim: `[mass] * [length] ** 2 / [time] ** 2` (energy) |
| `E = m * c ** 2` | `E: joule`, `m: kilogram`, `c: meter / second` | **PASS** | LHS: `[mass] * [length] ** 2 / [time] ** 2`<br>RHS: `[mass] * [length] ** 2 / [time] ** 2` (match) |
| `K = 0.5 * m * v ** 2` | `K: joule`, `m: kilogram`, `v: meter / second` | **PASS** | Both sides match energy dimensions |
| `v = d / t` | `v: meter / second`, `d: meter`, `t: second` | **PASS** | Both sides match velocity dimensions |
| `F = m * a` | `F: newton`, `m: kilogram`, `a: meter / second ** 2` | **PASS** | Both sides match force dimensions |
| `a = v ** 2 / r` | `a: meter / second ** 2`, `v: meter / second`, `r: meter` | **PASS** | Both sides match acceleration dimensions |
| `U = m * g * h` | `U: joule`, `m: kilogram`, `g: meter / second ** 2`, `h: meter` | **PASS** | Both sides match energy dimensions |
| `mass + velocity` | `mass: kilogram`, `velocity: meter / second` | **FAIL** | Incompatible addition: cannot add `[mass]` and `[length] / [time]` |
| `v = d / t` | `v: second`, `d: meter`, `t: second` | **FAIL** | Incompatible equation: LHS (`[time]`) $\neq$ RHS (`[length] / [time]`) |
| `mass ** length` | `mass: kilogram`, `length: meter` | **FAIL** | Exponent must be dimensionless, got `[length]` |
| `distance / 0` | `distance: meter` | **UNSUPPORTED** | Division by zero in AST constant evaluation |
| `sin(theta)` | `theta: radian` | **UNSUPPORTED** | Function calls (`Call`) outside restricted grammar |
| `__import__('os').system('dir')`| *(any)* | **UNSUPPORTED** | Code execution prohibited; rejected by AST validator |

---

## AI Integration Details (Gemma 4)

PhysCheck provides optional AI-assisted analysis to help users understand physical formulas and suggest unit candidates:

- **Official Model Identifier**: `gemma-4-31b-it` (defined as `DEFAULT_MODEL_IDENTIFIER` in `ai_analyzer.py`).
- **SDK**: Official Google GenAI SDK (`google-genai`).
- **Structured Schema**: The AI analyzer requests strict JSON responses validated by Pydantic models:
  - `summary`: Physical overview of the formulation.
  - `variable_meanings`: Physical meaning and common units for each variable.
  - `suggested_units`: List of Pint-compatible candidate unit strings.
  - `uncertainty`: AI's self-assessed confidence level and ambiguities.
  - `assumptions`: Assumed physical context (e.g., non-relativistic mechanics, ideal gas).
  - `potential_issues`: Potential dimensional traps or edge conditions.
  - `limitations`: Contextual boundaries where the formula ceases to apply.
- **No Silent Substitution**: If `gemma-4-31b-it` is unavailable or returns an error, the system raises an explicit `UnsupportedModelError` rather than silently switching to an unapproved model.
- **Zero Execution of AI Output**: Code or syntax returned by the model is never passed to an execution interpreter.

---

## Privacy & Data Transmission Notice

> [!IMPORTANT]
> **Data Transmission Disclosure**:
> - **Deterministic Checks (Offline)**: All operations performed by `checker.py`, `scripts/verify_math.py`, and the **Run Deterministic Physics Check** button in `app.py` execute **100% locally on your machine**. No mathematical expressions, variable names, or file contents are transmitted over any network.
> - **AI Analysis (Cloud API)**: When you click **Analyze Formulation with Gemma 4** in the web interface or call `ai_analyzer.py`, the submitted expression string, variable names, and surrounding code context are transmitted over HTTPS to Google's GenAI API servers for processing.
> - **Confidentiality Advice**: Do **not** submit proprietary formulas, unpublished research data, or confidential source code to the AI analyzer if your organization prohibits transmission to external cloud services.

---

## Limitations & Known Constraints

1. **Dimensional Consistency $\neq$ Physical Correctness**:
   A dimensional pass guarantees dimensional homogeneity, but cannot verify whether constant dimensionless prefactors (e.g., $1/2$ in $K = \frac{1}{2}mv^2$ versus $K = mv^2$) or the underlying physical law correctly models reality.
2. **No Transcendental Function Support**:
   Trigonometric (`sin`, `cos`, `tan`), exponential (`exp`), and logarithmic (`log`, `ln`) functions are currently rejected as `UNSUPPORTED`. While their arguments must be dimensionless in physics, arbitrary function evaluation is excluded to maintain strict parser guarantees.
3. **Dimensionless Exponents Only**:
   Exponents must evaluate to dimensionless real numbers. Physical expressions with dimensioned powers are mathematically and physically invalid and will trigger a `FAIL`.
4. **No Code Execution**:
   PhysCheck inspects uploaded `.py` files strictly as static text strings to extract equations. It does not execute scripts, resolve imports, or trace runtime state.
5. **No ODE / PDE Solving**:
   PhysCheck is a dimensional auditor, not a numerical simulation framework or differential equation solver.

---

## Troubleshooting

### 1. PowerShell Script Execution Policy Error on Windows
- **Symptom**: `File ... Activate.ps1 cannot be loaded because running scripts is disabled on this system.`
- **Remedy**: Run commands directly using the virtual environment Python executable:
  ```powershell
  .\.venv\Scripts\python.exe -m streamlit run app.py
  ```
  Or change the execution policy for your current user:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### 2. Missing or Invalid API Key
- **Symptom**: `MissingAPIKeyError: GEMINI_API_KEY environment variable is not set or empty.`
- **Remedy**:
  - Deterministic checks do not need a key; you can ignore this error if you only need the checker.
  - To use Gemma 4, create a `.env` file containing `GEMINI_API_KEY=your_key` in the project root directory.

### 3. Unit Not Recognized by Pint
- **Symptom**: `UndefinedUnitError: '...' is not defined in the unit registry.`
- **Remedy**:
  - Check unit spelling against Pint's standard names (e.g., use `meter` instead of `metres` or `m`, `second` instead of `sec`, `kilogram` instead of `kg`).
  - Compound units can be separated by spaces or asterisks: `meter / second ** 2`, `kilogram * meter / second`.

### 4. Expression Returns `UNSUPPORTED`
- **Symptom**: Result status is `UNSUPPORTED`.
- **Remedy**:
  - Verify that all variables in the expression have defined units.
  - Ensure the expression does not use unsupported operators (e.g., `^`, `&`, function calls like `sqrt()`).
  - Use `**` instead of `^` for exponentiation.

---

## Contributing

Contributions to PhysCheck are welcome! Please observe the following principles:

1. **Preserve Safety Guarantees**: Never introduce `eval()`, `exec()`, or dynamic code execution into equation evaluation.
2. **Preserve Disclaimer Integrity**: Any new reporting or output mode must preserve the mandatory disclaimer that `PASS` applies strictly to supported checks under confirmed assumptions.
3. **Strict Human Confirmation**: AI-generated units must always remain preliminary suggestions until confirmed by a user.
4. **Comprehensive Testing**: All new features or syntax support must include corresponding tests in `tests/`. Ensure all 46 existing tests pass:
   ```powershell
   .\.venv\Scripts\python.exe -m pytest -v
   ```

---

## License

This project is licensed under the terms of the **MIT License**. See the [LICENSE](LICENSE) file for complete details.
