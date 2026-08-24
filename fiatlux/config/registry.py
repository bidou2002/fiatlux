TYPE_REGISTRY = {}


def register_type(name):
    """
    Decorator used to register classes
    that can be built from JSON.
    """

    def decorator(cls):
        TYPE_REGISTRY[name] = cls
        return cls

    return decorator
