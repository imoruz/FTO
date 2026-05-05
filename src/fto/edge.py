class EdgeSuppressor:

    def suppress(self, *args, **kwargs):
        return None

    def restore(self, token, *args, **kwargs):
        pass


class MethodSwapEdgeSuppressor(EdgeSuppressor):

    def __init__(self, instance_from_args, method_name: str):
        self._get_instance = instance_from_args
        self._method_name = method_name

    def suppress(self, *args, **kwargs):
        instance = self._get_instance(*args, **kwargs)
        original = getattr(instance, self._method_name)
        setattr(instance, self._method_name, lambda *a, **kw: None)
        return (instance, original)

    def restore(self, token, *args, **kwargs):
        instance, original = token
        setattr(instance, self._method_name, original)
