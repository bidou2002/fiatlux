from fiatlux.config.registry import TYPE_REGISTRY
from fiatlux.config.resolver import (
    resolve_reference,
    evaluate_expression,
)


def get_class_from_name(name):

    # convert to PascalCase
    cls_name = name.split("_")[0]
    if cls_name not in TYPE_REGISTRY:
        raise ValueError(f"Unknown element type '{cls_name}'")

    return TYPE_REGISTRY[cls_name]


def build_nested(name, cfg, objects):
    """
    cfg: dict like {"actuator_grid": { ... }}
    """
    key = list(cfg.keys())[0]

    if key in TYPE_REGISTRY:
        cls = TYPE_REGISTRY[key]
        params = cfg[key]

        # recursively resolve params
        resolved_params = {}
        for k, v in params.items():
            if isinstance(v, dict):
                resolved_params[k] = build_nested(f"{name}.{k}", v, objects)
            else:
                resolved_params[k] = resolve_reference(v, objects)
                resolved_params[k] = evaluate_expression(resolved_params[k], objects)

        return cls(**resolved_params)

    else:
        # not a registered type, return raw dict
        return cfg


def build_serial_elements(config, objects):

    elements = []

    for name, params in config.items():

        cls = get_class_from_name(name)

        kwargs = {}

        for key, value in params.items():

            if isinstance(value, dict):

                kwargs[key] = build_nested(f"{name}.{key}", value, objects)

            else:

                value = resolve_reference(value, objects)
                value = evaluate_expression(value, objects)

                kwargs[key] = value

        obj = cls(**kwargs)

        objects[name] = obj

        elements.append(obj)

    return elements
