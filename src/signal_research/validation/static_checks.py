"""AST and runtime checks for safe, point-in-time signal functions."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable

import numpy as np
import pandas as pd


class StaticCheckError(ValueError):
    pass


class RuntimeSignalError(ValueError):
    pass


ALLOWED_NODES = {
    ast.Add,
    ast.And,
    ast.Assign,
    ast.arguments,
    ast.arg,
    ast.Attribute,
    ast.BinOp,
    ast.Call,
    ast.Compare,
    ast.Constant,
    ast.Div,
    ast.Eq,
    ast.Expr,
    ast.FloorDiv,
    ast.FunctionDef,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Load,
    ast.Is,
    ast.IsNot,
    ast.Mod,
    ast.Module,
    ast.Mult,
    ast.Name,
    ast.NotEq,
    ast.Not,
    ast.NotIn,
    ast.Or,
    ast.Pow,
    ast.Return,
    ast.Store,
    ast.Subscript,
    ast.Sub,
    ast.Tuple,
    ast.In,
    ast.BoolOp,
    ast.USub,
    ast.UAdd,
    ast.UnaryOp,
    ast.keyword,
}

ALLOWED_NUMPY_ATTRIBUTES = {"arange", "isfinite", "log", "log1p", "nan", "ndarray", "sqrt", "sum"}
ALLOWED_PANDAS_ATTRIBUTES = {"DataFrame", "Series"}
ALLOWED_METHOD_ATTRIBUTES = {
    "apply",
    "groupby",
    "index",
    "median",
    "mean",
    "reindex",
    "rename",
    "rolling",
    "size",
    "shift",
    "stack",
    "std",
}
ALLOWED_CALL_NAMES = {"_trend_quality_array", "_wide", "float", "max"}
ALLOWED_KEYWORDS = {"level", "min_periods", "raw"}


def _simple_subscript_key(node: ast.AST, allowed_fields: set[str]) -> bool:
    """Return whether a subscript key is literal and field-safe."""
    if isinstance(node, ast.Constant):
        return not isinstance(node.value, str) or node.value in allowed_fields
    if isinstance(node, ast.Tuple):
        return all(_simple_subscript_key(element, allowed_fields) for element in node.elts)
    return False


def _attribute_is_allowed(node: ast.Attribute) -> bool:
    if isinstance(node.value, ast.Name) and node.value.id == "np":
        return node.attr in ALLOWED_NUMPY_ATTRIBUTES
    if isinstance(node.value, ast.Name) and node.value.id == "pd":
        return node.attr in ALLOWED_PANDAS_ATTRIBUTES
    return node.attr in ALLOWED_METHOD_ATTRIBUTES


def check_signal_function(function: Callable, allowed_fields: set[str]) -> None:
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    bound_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    bound_names.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if len(function_defs) != 1:
        raise StaticCheckError("signal functions may not contain nested function definitions")
    function_def = function_defs[0]
    arguments = function_def.args
    if arguments.vararg or arguments.kwarg or arguments.kwonlyargs:
        raise StaticCheckError("signal functions may not use *args, **kwargs, or keyword-only parameters")
    positional = list(arguments.posonlyargs) + list(arguments.args)
    if not positional:
        raise StaticCheckError("signal functions must accept the panel as their first parameter")
    # Only the panel argument itself may be forwarded to `_wide`; every other
    # parameter or variable must resolve to declared string-literal fields.
    panel_parameter = positional[0].arg
    protected_names = ALLOWED_CALL_NAMES | {"np", "pd", "EPS", "int"} | {panel_parameter}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in protected_names:
            raise StaticCheckError(f"protected name may not be rebound: {node.id}")
    for parameter in positional[1:]:
        if parameter.arg in protected_names:
            raise StaticCheckError(f"protected name may not be a parameter: {parameter.arg}")
    string_assignments: dict[str, set[str]] = {}
    untrusted_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        string_assignments.setdefault(target.id, set()).add(node.value.value)
                    else:
                        # Any non-literal binding makes the variable untrusted as a field name.
                        untrusted_names.add(target.id)
    allowed_names = bound_names | {"EPS", "np", "pd", "int"} | ALLOWED_CALL_NAMES
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    for node in ast.walk(tree):
        # The panel may only be read through declared-field subscripts, the
        # `_wide` helper, or its index; whole-frame operations would otherwise
        # reach undeclared columns.
        if (
            isinstance(node, ast.Name)
            and node.id == panel_parameter
            and isinstance(node.ctx, ast.Load)
        ):
            parent = parents.get(node)
            subscript_ok = isinstance(parent, ast.Subscript) and parent.value is node
            index_ok = isinstance(parent, ast.Attribute) and parent.attr == "index"
            wide_ok = (
                isinstance(parent, ast.Call)
                and isinstance(parent.func, ast.Name)
                and parent.func.id == "_wide"
                and parent.args
                and parent.args[0] is node
            )
            if not (subscript_ok or index_ok or wide_ok):
                raise StaticCheckError(
                    "panel may only be used via declared-field subscripts, _wide, or .index"
                )
        if type(node) not in ALLOWED_NODES:
            raise StaticCheckError(f"unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise StaticCheckError(f"name is not allowlisted: {node.id}")
        if isinstance(node, ast.Attribute) and not _attribute_is_allowed(node):
            raise StaticCheckError(f"attribute is not allowlisted: {node.attr}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in ALLOWED_CALL_NAMES:
                raise StaticCheckError(f"call is not allowlisted: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and not _attribute_is_allowed(node.func):
                raise StaticCheckError(f"call is not allowlisted: {node.func.attr}")
            if any(keyword.arg not in ALLOWED_KEYWORDS for keyword in node.keywords):
                raise StaticCheckError("keyword is not allowlisted")
            if isinstance(node.func, ast.Name) and node.func.id == "_wide":
                field_args = list(node.args) + [keyword.value for keyword in node.keywords]
                for argument in field_args:
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        if argument.value not in allowed_fields:
                            raise StaticCheckError(
                                f"field access is not declared in required_fields: {argument.value}"
                            )
                    elif isinstance(argument, ast.Name):
                        if argument.id == panel_parameter:
                            continue
                        assigned = string_assignments.get(argument.id)
                        if (
                            argument.id in untrusted_names
                            or assigned is None
                            or not assigned
                            or not assigned <= allowed_fields
                        ):
                            raise StaticCheckError(
                                f"_wide field variables must resolve to declared fields: {argument.id}"
                            )
                    else:
                        raise StaticCheckError("_wide field arguments must be literals or names")
        if isinstance(node, ast.Subscript) and not _simple_subscript_key(node.slice, allowed_fields):
            raise StaticCheckError("subscript keys must be literal declared fields")


def validate_runtime_signal(signal: pd.Series, panel_index: pd.MultiIndex) -> None:
    if not isinstance(signal, pd.Series):
        raise RuntimeSignalError("signal must be a Series")
    if not signal.index.equals(panel_index):
        raise RuntimeSignalError("signal index is not exactly aligned to the panel index")
    if signal.index.has_duplicates:
        raise RuntimeSignalError("signal index contains duplicates")


def truncation_cutoffs(panel: pd.DataFrame, fractions: tuple[float, ...] = (0.3, 0.45, 0.6, 0.75, 0.9)) -> list[pd.Timestamp]:
    """Deterministic cutoffs spread through the sample for invariance checks."""
    dates = panel.index.get_level_values("date").unique()
    return sorted({dates[max(1, min(len(dates) - 2, int(len(dates) * fraction)))] for fraction in fractions})


def assert_truncation_invariant(
    function: Callable,
    panel: pd.DataFrame,
    cutoff: pd.Timestamp,
    parameters: dict[str, int | float] | None = None,
) -> None:
    """Verify that truncating future rows does not alter values through cutoff.

    The check must run with the candidate's actual parameters: a function can be
    invariant at its defaults yet look ahead for other parameter values. Callers
    should check several cutoffs (see ``truncation_cutoffs``); this guards
    against accidental look-ahead, not a deliberately adversarial author.
    """
    kwargs = dict(parameters or {})
    full = function(panel, **kwargs)
    truncated_panel = panel.loc[panel.index.get_level_values("date") <= cutoff]
    truncated = function(truncated_panel, **kwargs)
    full_before = full.loc[full.index.get_level_values("date") <= cutoff]
    if not full_before.index.equals(truncated.index):
        raise RuntimeSignalError("truncation changed the historical index")
    left = full_before.to_numpy(dtype=float)
    right = truncated.to_numpy(dtype=float)
    equal = np.isclose(left, right, atol=1e-10, rtol=0.0, equal_nan=True)
    if not bool(equal.all()):
        raise RuntimeSignalError("signal is not invariant under truncation through cutoff")
