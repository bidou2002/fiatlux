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
    """
    Evaluate expressions like:

        "source.spectrum.wavelength.max()"
    """

    if not isinstance(value, str):
        return value

    if "." not in value:
        return value

    try:
        return eval(value, {}, context)

    except Exception:
        return value
