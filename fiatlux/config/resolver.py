import ast
import math
import operator


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_SAFE_METHODS = {"max", "min", "mean", "sum", "sqrt", "item"}
_CONSTANTS = {"pi": math.pi}


def resolve_reference(value, objects):
    """
    Replace string references by real objects.

    Example:
        "pupil_grid" → objects["pupil_grid"]
    """

    if isinstance(value, str):

        if value in objects:
            return objects[value]

    return value


def evaluate_expression(value, context):
    """Evaluate a restricted configuration expression.

    Supported syntax consists of numeric constants, named configuration
    objects, public attribute access, arithmetic, and a small allowlist of
    zero-argument numerical methods. No Python builtins are available and
    arbitrary function calls, indexing, comprehensions, and private
    attributes are rejected.
    """

    if not isinstance(value, str):
        return value
    if "/" in value and value.rsplit("/", 1)[-1].count(".") == 1:
        # File paths such as ``data/pupil.fits`` are configuration literals,
        # not division expressions.
        return value

    try:
        expression = ast.parse(value, mode="eval")
    except SyntaxError:
        return value

    root = expression.body
    is_context_reference = (
        isinstance(root, (ast.Attribute, ast.Name, ast.Call))
        and _root_name(root) in context
    )
    is_arithmetic = isinstance(root, (ast.BinOp, ast.UnaryOp))
    if not is_context_reference and not is_arithmetic:
        if isinstance(root, (ast.Name, ast.Constant, ast.Attribute)):
            return value
        raise ValueError(f"Unsupported configuration expression: {value!r}.")

    try:
        return _evaluate_node(root, context)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Unsupported configuration expression {value!r}: {error}"
        ) from error


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Call)):
        node = node.value if isinstance(node, ast.Attribute) else node.func
    return node.id if isinstance(node, ast.Name) else None


def _evaluate_node(node: ast.AST, context):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("only numeric constants are allowed")

    if isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"unknown name '{node.id}'")

    if isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            raise ValueError("private attributes are not allowed")
        return getattr(_evaluate_node(node.value, context), node.attr)

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left, context)
        right = _evaluate_node(node.right, context)
        return _BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand, context))

    if isinstance(node, ast.Call):
        if node.args or node.keywords or not isinstance(node.func, ast.Attribute):
            raise ValueError("only allowlisted zero-argument methods are allowed")
        if node.func.attr not in _SAFE_METHODS:
            raise ValueError(f"method '{node.func.attr}' is not allowed")
        method = _evaluate_node(node.func, context)
        if not callable(method):
            raise ValueError(f"attribute '{node.func.attr}' is not callable")
        return method()

    raise ValueError(f"syntax '{type(node).__name__}' is not allowed")
