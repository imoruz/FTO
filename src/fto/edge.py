from typing import Any, Callable, Tuple


class EdgeSuppressor:
    def suppress(self, *args, **kwargs) -> Tuple[Any, Any]:
        pass

    def restore(self, token: Tuple[Any, Any], *args, **kwargs) -> None:
        pass


class MethodSwapEdgeSuppressor(EdgeSuppressor):
    def __init__(self, instance_from_args: Callable[..., Any], method_name: str) -> None:
        self._get_instance = instance_from_args
        self._method_name = method_name

    def suppress(self, *args, **kwargs) -> Tuple[Any, Any]:
        instance = self._get_instance(*args, **kwargs)
        original = getattr(instance, self._method_name)
        setattr(instance, self._method_name, lambda *a, **kw: None)
        return (instance, original)

    def restore(self, token: Tuple[Any, Any], *args, **kwargs) -> None:
        instance, original = token
        setattr(instance, self._method_name, original)
