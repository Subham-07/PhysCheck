"""Deterministic Expression and Dimensionality Checker for PhysCheck.

Milestone 2:
- Uses Pint for unit handling and dimensional compatibility.
- Uses Python AST in a restricted parser mode (never eval or exec).
- Never executes uploaded or pasted code.
- Supports documented expression subset: numeric constants, named variables with
  explicitly supplied units, addition, subtraction, multiplication, division,
  parentheses, and restricted powers.
- Rejects unsupported syntax, unknown variables, invalid units, unsafe operations,
  and division by zero as UNSUPPORTED.
- Reports PASS, FAIL, or UNSUPPORTED with evidence, assumptions, and exact check performed.
- Includes disclaimer: PASS means only that the supported check passed under the
  supplied units; it does not mean the whole scientific program is correct.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, asdict
from enum import Enum
import math
from typing import Any, Dict, Optional, Union

import pint

# Global shared UnitRegistry for performance
_DEFAULT_UREG: Optional[pint.UnitRegistry] = None


def get_unit_registry() -> pint.UnitRegistry:
    """Return or initialize the singleton UnitRegistry."""
    global _DEFAULT_UREG
    if _DEFAULT_UREG is None:
        _DEFAULT_UREG = pint.UnitRegistry()
    return _DEFAULT_UREG


PASS_DISCLAIMER = (
    "PASS means only that the supported check passed under the supplied units; "
    "it does not mean the whole scientific program is correct."
)


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNSUPPORTED = "UNSUPPORTED"

    def __str__(self) -> str:
        return self.value


@dataclass
class CheckResult:
    """Structured result of a PhysCheck verification."""

    status: CheckStatus
    check_performed: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    assumptions: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

    @property
    def is_pass(self) -> bool:
        return self.status == CheckStatus.PASS

    @property
    def is_fail(self) -> bool:
        return self.status == CheckStatus.FAIL

    @property
    def is_unsupported(self) -> bool:
        return self.status == CheckStatus.UNSUPPORTED

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


class CheckerInternalError(Exception):
    """Base internal exception for AST checker flow control."""


class UnsupportedSyntaxError(CheckerInternalError):
    def __init__(self, message: str, node_type: Optional[str] = None):
        super().__init__(message)
        self.node_type = node_type


class UnknownVariableError(CheckerInternalError):
    def __init__(self, variable_name: str):
        super().__init__(f"Unknown variable: '{variable_name}' (no unit supplied)")
        self.variable_name = variable_name


class DivisionByZeroError(CheckerInternalError):
    def __init__(self, message: str = "Division by zero detected"):
        super().__init__(message)


class IncompatibleDimensionsError(CheckerInternalError):
    def __init__(self, message: str, evidence: Dict[str, Any]):
        super().__init__(message)
        self.evidence = evidence


class RestrictedASTEvaluator:
    """Evaluates an AST subtree using strictly whitelisted nodes and Pint quantities."""

    ALLOWED_MAX_EXPONENT = 100.0

    def __init__(self, variables: Dict[str, pint.Quantity], ureg: pint.UnitRegistry):
        self.variables = variables
        self.ureg = ureg

    def evaluate(self, node: ast.AST) -> pint.Quantity:
        """Recursively evaluate an AST node into a Pint Quantity."""
        # 1. Numeric constants
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                raise UnsupportedSyntaxError(
                    f"Boolean literal '{node.value}' is not supported in physical expressions",
                    node_type="Constant(bool)",
                )
            if not isinstance(node.value, (int, float)):
                raise UnsupportedSyntaxError(
                    f"Non-numeric literal of type '{type(node.value).__name__}' is not supported",
                    node_type=f"Constant({type(node.value).__name__})",
                )
            if math.isnan(node.value) or math.isinf(node.value):
                raise UnsupportedSyntaxError(
                    f"Non-finite numeric constant '{node.value}' is not supported",
                    node_type="Constant(non-finite)",
                )
            return self.ureg.Quantity(float(node.value), "")

        # 2. Named variables
        elif isinstance(node, ast.Name):
            if node.id not in self.variables:
                raise UnknownVariableError(node.id)
            return self.variables[node.id]

        # 3. Unary operators (+, -)
        elif isinstance(node, ast.UnaryOp):
            operand = self.evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            elif isinstance(node.op, ast.USub):
                return -operand
            else:
                raise UnsupportedSyntaxError(
                    f"Unsupported unary operator: {type(node.op).__name__}",
                    node_type=type(node.op).__name__,
                )

        # 4. Binary operators (+, -, *, /, **)
        elif isinstance(node, ast.BinOp):
            return self._evaluate_binop(node)

        # Anything else is completely disallowed (Call, Attribute, Subscript, etc.)
        else:
            raise UnsupportedSyntaxError(
                f"Unsupported syntax or AST node: {type(node).__name__}",
                node_type=type(node).__name__,
            )

    def _evaluate_binop(self, node: ast.BinOp) -> pint.Quantity:
        left = self.evaluate(node.left)
        right = self.evaluate(node.right)

        # Addition: operands must have identical dimensionality
        if isinstance(node.op, ast.Add):
            if left.dimensionality != right.dimensionality:
                raise IncompatibleDimensionsError(
                    f"Incompatible addition: cannot add {left.units} (dim: {left.dimensionality}) "
                    f"and {right.units} (dim: {right.dimensionality})",
                    evidence={
                        "operation": "addition",
                        "left_units": str(left.units),
                        "right_units": str(right.units),
                        "left_dimensionality": str(left.dimensionality),
                        "right_dimensionality": str(right.dimensionality),
                        "left_quantity": str(left),
                        "right_quantity": str(right),
                    },
                )
            return left + right

        # Subtraction: operands must have identical dimensionality
        elif isinstance(node.op, ast.Sub):
            if left.dimensionality != right.dimensionality:
                raise IncompatibleDimensionsError(
                    f"Incompatible subtraction: cannot subtract {right.units} (dim: {right.dimensionality}) "
                    f"from {left.units} (dim: {left.dimensionality})",
                    evidence={
                        "operation": "subtraction",
                        "left_units": str(left.units),
                        "right_units": str(right.units),
                        "left_dimensionality": str(left.dimensionality),
                        "right_dimensionality": str(right.dimensionality),
                        "left_quantity": str(left),
                        "right_quantity": str(right),
                    },
                )
            return left - right

        # Multiplication
        elif isinstance(node.op, ast.Mult):
            return left * right

        # Division
        elif isinstance(node.op, ast.Div):
            if right.magnitude == 0:
                raise DivisionByZeroError("Division by zero in expression")
            return left / right

        # Restricted Powers
        elif isinstance(node.op, ast.Pow):
            # Exponent must be dimensionless
            if not right.dimensionless:
                raise IncompatibleDimensionsError(
                    f"Power exponent must be dimensionless, got {right.units} (dim: {right.dimensionality})",
                    evidence={
                        "operation": "power",
                        "base_units": str(left.units),
                        "exponent_units": str(right.units),
                        "exponent_dimensionality": str(right.dimensionality),
                    },
                )

            # Exponent magnitude must be finite and within safe bounds
            exp_val = right.to_base_units().magnitude
            if not isinstance(exp_val, (int, float)) or math.isnan(exp_val) or math.isinf(exp_val):
                raise UnsupportedSyntaxError(
                    f"Exponent must evaluate to a finite real number, got {exp_val}",
                    node_type="Pow",
                )

            if abs(exp_val) > self.ALLOWED_MAX_EXPONENT:
                raise UnsupportedSyntaxError(
                    f"Exponent magnitude |{exp_val}| exceeds safe limit of {self.ALLOWED_MAX_EXPONENT}",
                    node_type="Pow",
                )

            if left.magnitude == 0 and exp_val < 0:
                raise DivisionByZeroError("Zero cannot be raised to a negative power")

            try:
                result = left ** exp_val
                if isinstance(result.magnitude, complex):
                    raise UnsupportedSyntaxError(
                        "Complex power results are not supported in physical expressions",
                        node_type="Pow(complex)",
                    )
                return result
            except Exception as e:
                raise IncompatibleDimensionsError(
                    f"Power operation failed: {e}",
                    evidence={"operation": "power", "error": str(e)},
                )

        else:
            raise UnsupportedSyntaxError(
                f"Unsupported binary operator: {type(node.op).__name__}",
                node_type=type(node.op).__name__,
            )


def _prepare_variables(
    variable_units: Optional[Dict[str, Any]],
    ureg: pint.UnitRegistry,
    assumptions_base: Dict[str, Any],
) -> Union[Dict[str, pint.Quantity], CheckResult]:
    """Validate and parse user-supplied variable units into Pint quantities."""
    if variable_units is None:
        return {}

    if not isinstance(variable_units, dict):
        return CheckResult(
            status=CheckStatus.UNSUPPORTED,
            check_performed="Variable units dictionary validation",
            evidence={"provided_type": type(variable_units).__name__},
            assumptions=assumptions_base,
            message="variable_units must be a dictionary mapping variable names to units",
        )

    parsed_vars: Dict[str, pint.Quantity] = {}
    for var_name, unit_spec in variable_units.items():
        if not isinstance(var_name, str) or not var_name.isidentifier():
            return CheckResult(
                status=CheckStatus.UNSUPPORTED,
                check_performed="Variable identifier validation",
                evidence={"invalid_identifier": str(var_name)},
                assumptions=assumptions_base,
                message=f"Variable name '{var_name}' is not a valid Python identifier",
            )

        try:
            if isinstance(unit_spec, ureg.Quantity):
                parsed_vars[var_name] = unit_spec
            elif isinstance(unit_spec, str):
                cleaned_unit = unit_spec.strip()
                if not cleaned_unit:
                    return CheckResult(
                        status=CheckStatus.UNSUPPORTED,
                        check_performed=f"Unit validation for variable '{var_name}'",
                        evidence={"variable": var_name, "raw_unit": unit_spec},
                        assumptions=assumptions_base,
                        message=f"Empty unit string provided for variable '{var_name}'",
                    )
                unit_obj = ureg.parse_units(cleaned_unit)
                parsed_vars[var_name] = ureg.Quantity(1.0, unit_obj)
            elif isinstance(unit_spec, (tuple, list)) and len(unit_spec) == 2:
                mag, u_str = unit_spec
                unit_obj = ureg.parse_units(str(u_str))
                parsed_vars[var_name] = ureg.Quantity(float(mag), unit_obj)
            else:
                return CheckResult(
                    status=CheckStatus.UNSUPPORTED,
                    check_performed=f"Unit specification format for '{var_name}'",
                    evidence={"variable": var_name, "unit_spec_type": type(unit_spec).__name__},
                    assumptions=assumptions_base,
                    message=f"Unsupported unit specification type for variable '{var_name}'",
                )
        except Exception as e:
            return CheckResult(
                status=CheckStatus.UNSUPPORTED,
                check_performed=f"Unit parsing for variable '{var_name}'",
                evidence={"variable": var_name, "raw_unit": str(unit_spec), "error": str(e)},
                assumptions=assumptions_base,
                message=f"Invalid unit '{unit_spec}' for variable '{var_name}': {e}",
            )

    return parsed_vars


def check_expression(
    expression: str,
    variable_units: Optional[Dict[str, Any]] = None,
    expected_unit: Optional[str] = None,
    ureg: Optional[pint.UnitRegistry] = None,
) -> CheckResult:
    """Deterministically parse and check physical expression dimensional compatibility.

    Args:
        expression: Expression string or equation (e.g. 'v = d / t', 'v == d / t', 'a + b').
        variable_units: Dict mapping variable names to their unit strings or Quantities.
        expected_unit: Optional target unit for single expression checks.
        ureg: Optional Pint UnitRegistry. Uses singleton if omitted.

    Returns:
        CheckResult with status PASS, FAIL, or UNSUPPORTED.
    """
    if ureg is None:
        ureg = get_unit_registry()

    raw_units_record = {k: str(v) for k, v in (variable_units or {}).items()}
    assumptions: Dict[str, Any] = {
        "disclaimer": PASS_DISCLAIMER,
        "variable_units": raw_units_record,
    }
    if expected_unit is not None:
        assumptions["expected_unit"] = str(expected_unit)

    # Validate input expression string
    if not isinstance(expression, str) or not expression.strip():
        return CheckResult(
            status=CheckStatus.UNSUPPORTED,
            check_performed="Expression input validation",
            evidence={"expression": expression},
            assumptions=assumptions,
            message="Expression must be a non-empty string",
        )

    cleaned_expr = expression.strip()

    # Validate and parse variable units
    vars_or_err = _prepare_variables(variable_units, ureg, assumptions)
    if isinstance(vars_or_err, CheckResult):
        return vars_or_err
    variables: Dict[str, pint.Quantity] = vars_or_err

    # Validate expected_unit if provided
    expected_quantity: Optional[pint.Quantity] = None
    if expected_unit is not None:
        try:
            exp_unit_obj = ureg.parse_units(expected_unit)
            expected_quantity = ureg.Quantity(1.0, exp_unit_obj)
        except Exception as e:
            return CheckResult(
                status=CheckStatus.UNSUPPORTED,
                check_performed="Expected unit parsing",
                evidence={"expected_unit": expected_unit, "error": str(e)},
                assumptions=assumptions,
                message=f"Invalid expected unit '{expected_unit}': {e}",
            )

    # Parse AST safely (never eval or exec)
    try:
        tree = ast.parse(cleaned_expr, mode="exec")
    except SyntaxError as e:
        return CheckResult(
            status=CheckStatus.UNSUPPORTED,
            check_performed="AST syntax parsing",
            evidence={"syntax_error": str(e), "line": e.lineno, "offset": e.offset},
            assumptions=assumptions,
            message=f"Syntax error in expression: {e.msg}",
        )

    # Strictly disallow multiple statements or empty trees
    if len(tree.body) != 1:
        return CheckResult(
            status=CheckStatus.UNSUPPORTED,
            check_performed="AST statement structure validation",
            evidence={"statement_count": len(tree.body)},
            assumptions=assumptions,
            message="Only single expressions or single assignment equations are supported",
        )

    stmt = tree.body[0]
    evaluator = RestrictedASTEvaluator(variables, ureg)

    # Case 1: Assignment equation (e.g. 'v = d / t')
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            return CheckResult(
                status=CheckStatus.UNSUPPORTED,
                check_performed="Equation assignment target check",
                evidence={"targets": [type(t).__name__ for t in stmt.targets]},
                assumptions=assumptions,
                message="Equation target must be a single variable name",
            )
        target_name = stmt.targets[0].id
        if target_name not in variables:
            return CheckResult(
                status=CheckStatus.UNSUPPORTED,
                check_performed=f"Target variable unit check: '{target_name}'",
                evidence={"unknown_variable": target_name},
                assumptions=assumptions,
                message=f"Unknown variable: '{target_name}' (no unit supplied)",
            )

        lhs_quantity = variables[target_name]
        try:
            rhs_quantity = evaluator.evaluate(stmt.value)
        except UnsupportedSyntaxError as e:
            return CheckResult(
                status=CheckStatus.UNSUPPORTED,
                check_performed="Equation RHS AST evaluation",
                evidence={"unsupported_syntax": str(e), "node_type": e.node_type},
                assumptions=assumptions,
                message=str(e),
            )
        except UnknownVariableError as e:
            return CheckResult(
                status=CheckStatus.UNSUPPORTED,
                check_performed="Equation RHS variable lookup",
                evidence={"unknown_variable": e.variable_name},
                assumptions=assumptions,
                message=str(e),
            )
        except DivisionByZeroError as e:
            return CheckResult(
                status=CheckStatus.UNSUPPORTED,
                check_performed="Equation RHS arithmetic evaluation",
                evidence={"error": "Division by zero"},
                assumptions=assumptions,
                message=str(e),
            )
        except IncompatibleDimensionsError as e:
            return CheckResult(
                status=CheckStatus.FAIL,
                check_performed="Equation RHS dimensional compatibility",
                evidence=e.evidence,
                assumptions=assumptions,
                message=str(e),
            )

        # Check LHS vs RHS dimensional compatibility
        check_desc = (
            f"Dimensional equality check: '{target_name}' "
            f"({lhs_quantity.dimensionality}) == RHS ({rhs_quantity.dimensionality})"
        )
        if lhs_quantity.dimensionality == rhs_quantity.dimensionality:
            return CheckResult(
                status=CheckStatus.PASS,
                check_performed=check_desc,
                evidence={
                    "lhs_variable": target_name,
                    "lhs_unit": str(lhs_quantity.units),
                    "lhs_dimensionality": str(lhs_quantity.dimensionality),
                    "rhs_inferred_unit": str(rhs_quantity.units),
                    "rhs_dimensionality": str(rhs_quantity.dimensionality),
                    "dimensionality_match": True,
                },
                assumptions=assumptions,
                message=f"Dimensional check passed: '{target_name}' matches RHS dimensionality",
            )
        else:
            return CheckResult(
                status=CheckStatus.FAIL,
                check_performed=check_desc,
                evidence={
                    "lhs_variable": target_name,
                    "lhs_unit": str(lhs_quantity.units),
                    "lhs_dimensionality": str(lhs_quantity.dimensionality),
                    "rhs_inferred_unit": str(rhs_quantity.units),
                    "rhs_dimensionality": str(rhs_quantity.dimensionality),
                    "dimensionality_match": False,
                },
                assumptions=assumptions,
                message=(
                    f"Dimensional mismatch in equation: '{target_name}' has dimension "
                    f"{lhs_quantity.dimensionality}, but RHS evaluates to {rhs_quantity.dimensionality}"
                ),
            )

    # Case 2: Expression statement (single expression, or comparison equation like 'v == d / t')
    elif isinstance(stmt, ast.Expr):
        val_node = stmt.value

        # Handle comparison equation (e.g. 'v == d / t')
        if isinstance(val_node, ast.Compare):
            if len(val_node.ops) != 1 or not isinstance(val_node.ops[0], ast.Eq) or len(val_node.comparators) != 1:
                return CheckResult(
                    status=CheckStatus.UNSUPPORTED,
                    check_performed="Comparison equation check",
                    evidence={"operators": [type(op).__name__ for op in val_node.ops]},
                    assumptions=assumptions,
                    message="Only single equality comparisons ('==') are supported in equation checks",
                )
            try:
                lhs_quantity = evaluator.evaluate(val_node.left)
                rhs_quantity = evaluator.evaluate(val_node.comparators[0])
            except UnsupportedSyntaxError as e:
                return CheckResult(
                    status=CheckStatus.UNSUPPORTED,
                    check_performed="Comparison AST evaluation",
                    evidence={"unsupported_syntax": str(e), "node_type": e.node_type},
                    assumptions=assumptions,
                    message=str(e),
                )
            except UnknownVariableError as e:
                return CheckResult(
                    status=CheckStatus.UNSUPPORTED,
                    check_performed="Comparison variable lookup",
                    evidence={"unknown_variable": e.variable_name},
                    assumptions=assumptions,
                    message=str(e),
                )
            except DivisionByZeroError as e:
                return CheckResult(
                    status=CheckStatus.UNSUPPORTED,
                    check_performed="Comparison arithmetic evaluation",
                    evidence={"error": "Division by zero"},
                    assumptions=assumptions,
                    message=str(e),
                )
            except IncompatibleDimensionsError as e:
                return CheckResult(
                    status=CheckStatus.FAIL,
                    check_performed="Comparison internal dimensional compatibility",
                    evidence=e.evidence,
                    assumptions=assumptions,
                    message=str(e),
                )

            check_desc = (
                f"Dimensional equality check: LHS ({lhs_quantity.dimensionality}) "
                f"== RHS ({rhs_quantity.dimensionality})"
            )
            if lhs_quantity.dimensionality == rhs_quantity.dimensionality:
                return CheckResult(
                    status=CheckStatus.PASS,
                    check_performed=check_desc,
                    evidence={
                        "lhs_unit": str(lhs_quantity.units),
                        "lhs_dimensionality": str(lhs_quantity.dimensionality),
                        "rhs_unit": str(rhs_quantity.units),
                        "rhs_dimensionality": str(rhs_quantity.dimensionality),
                        "dimensionality_match": True,
                    },
                    assumptions=assumptions,
                    message="Dimensional equality check passed between LHS and RHS",
                )
            else:
                return CheckResult(
                    status=CheckStatus.FAIL,
                    check_performed=check_desc,
                    evidence={
                        "lhs_unit": str(lhs_quantity.units),
                        "lhs_dimensionality": str(lhs_quantity.dimensionality),
                        "rhs_unit": str(rhs_quantity.units),
                        "rhs_dimensionality": str(rhs_quantity.dimensionality),
                        "dimensionality_match": False,
                    },
                    assumptions=assumptions,
                    message=(
                        f"Dimensional mismatch in equation: LHS has dimension {lhs_quantity.dimensionality}, "
                        f"but RHS has dimension {rhs_quantity.dimensionality}"
                    ),
                )

        # Standard expression (e.g. 'a + b', 'd / t', 'm * a')
        try:
            res_quantity = evaluator.evaluate(val_node)
        except UnsupportedSyntaxError as e:
            return CheckResult(
                status=CheckStatus.UNSUPPORTED,
                check_performed="Expression AST evaluation",
                evidence={"unsupported_syntax": str(e), "node_type": e.node_type},
                assumptions=assumptions,
                message=str(e),
            )
        except UnknownVariableError as e:
            return CheckResult(
                status=CheckStatus.UNSUPPORTED,
                check_performed="Expression variable lookup",
                evidence={"unknown_variable": e.variable_name},
                assumptions=assumptions,
                message=str(e),
            )
        except DivisionByZeroError as e:
            return CheckResult(
                status=CheckStatus.UNSUPPORTED,
                check_performed="Expression arithmetic evaluation",
                evidence={"error": "Division by zero"},
                assumptions=assumptions,
                message=str(e),
            )
        except IncompatibleDimensionsError as e:
            return CheckResult(
                status=CheckStatus.FAIL,
                check_performed="Expression internal dimensional compatibility",
                evidence=e.evidence,
                assumptions=assumptions,
                message=str(e),
            )

        # Check against expected_unit if requested
        if expected_quantity is not None:
            check_desc = (
                f"Dimensional compatibility check: inferred ({res_quantity.dimensionality}) "
                f"against expected ({expected_quantity.dimensionality})"
            )
            if res_quantity.dimensionality == expected_quantity.dimensionality:
                return CheckResult(
                    status=CheckStatus.PASS,
                    check_performed=check_desc,
                    evidence={
                        "inferred_unit": str(res_quantity.units),
                        "inferred_dimensionality": str(res_quantity.dimensionality),
                        "expected_unit": str(expected_quantity.units),
                        "expected_dimensionality": str(expected_quantity.dimensionality),
                        "dimensionality_match": True,
                    },
                    assumptions=assumptions,
                    message="Expression dimensionality matches expected unit",
                )
            else:
                return CheckResult(
                    status=CheckStatus.FAIL,
                    check_performed=check_desc,
                    evidence={
                        "inferred_unit": str(res_quantity.units),
                        "inferred_dimensionality": str(res_quantity.dimensionality),
                        "expected_unit": str(expected_quantity.units),
                        "expected_dimensionality": str(expected_quantity.dimensionality),
                        "dimensionality_match": False,
                    },
                    assumptions=assumptions,
                    message=(
                        f"Dimensional mismatch: expression has dimension {res_quantity.dimensionality}, "
                        f"but expected {expected_quantity.dimensionality}"
                    ),
                )

        # Standalone expression consistency passed
        check_desc = f"Dimensional consistency check for expression: '{cleaned_expr}'"
        return CheckResult(
            status=CheckStatus.PASS,
            check_performed=check_desc,
            evidence={
                "inferred_unit": str(res_quantity.units),
                "inferred_dimensionality": str(res_quantity.dimensionality),
                "inferred_quantity": str(res_quantity),
            },
            assumptions=assumptions,
            message="All operations within the expression are dimensionally consistent",
        )

    # Disallowed top-level AST node
    else:
        return CheckResult(
            status=CheckStatus.UNSUPPORTED,
            check_performed="Top-level AST node validation",
            evidence={"node_type": type(stmt).__name__},
            assumptions=assumptions,
            message=f"Unsupported top-level statement type: {type(stmt).__name__}",
        )


def check_equation(
    lhs: str,
    rhs: str,
    variable_units: Optional[Dict[str, Any]] = None,
    ureg: Optional[pint.UnitRegistry] = None,
) -> CheckResult:
    """Check dimensional equality between LHS and RHS expressions."""
    return check_expression(f"({lhs}) == ({rhs})", variable_units=variable_units, ureg=ureg)


class DeterministicChecker:
    """Class interface for PhysCheck deterministic expression verification."""

    def __init__(self, ureg: Optional[pint.UnitRegistry] = None):
        self.ureg = ureg or get_unit_registry()

    def check(
        self,
        expression: str,
        variable_units: Optional[Dict[str, Any]] = None,
        expected_unit: Optional[str] = None,
    ) -> CheckResult:
        return check_expression(expression, variable_units=variable_units, expected_unit=expected_unit, ureg=self.ureg)

    def check_equation(
        self,
        lhs: str,
        rhs: str,
        variable_units: Optional[Dict[str, Any]] = None,
    ) -> CheckResult:
        return check_equation(lhs, rhs, variable_units=variable_units, ureg=self.ureg)
