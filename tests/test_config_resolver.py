from types import SimpleNamespace

import pytest
import torch

from fiatlux.config.resolver import evaluate_expression, resolve_reference


def make_context():
    spectrum = SimpleNamespace(
        wavelengths=torch.tensor([1.0e-6, 1.2e-6], dtype=torch.float64)
    )
    return {"source": SimpleNamespace(spectrum=spectrum), "radius": 0.5}


def test_exact_object_reference_is_resolved():
    context = make_context()

    assert resolve_reference("source", context) is context["source"]


def test_public_attributes_arithmetic_and_safe_method_are_supported():
    context = make_context()

    result = evaluate_expression(
        "source.spectrum.wavelengths.max() / (2 * radius)", context
    )

    torch.testing.assert_close(result, torch.tensor(1.2e-6, dtype=torch.float64))


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('touch forbidden')",
        "source.__class__",
        "source.spectrum.wavelengths.tolist()",
        "source.spectrum.wavelengths[0]",
        "[value for value in source.spectrum.wavelengths]",
        "lambda: 1",
    ],
)
def test_unsafe_or_unsupported_python_is_rejected(expression):
    with pytest.raises(ValueError, match="Unsupported configuration expression"):
        evaluate_expression(expression, make_context())


@pytest.mark.parametrize("literal", ["HCM2", "data/file.fits", "file.fits"])
def test_plain_configuration_strings_remain_literals(literal):
    assert evaluate_expression(literal, make_context()) == literal


def test_unknown_name_in_arithmetic_has_actionable_error():
    with pytest.raises(ValueError, match="unknown name 'missing'"):
        evaluate_expression("missing * 2", make_context())
