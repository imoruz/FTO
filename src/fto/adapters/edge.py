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

    # def suppress(self, *args, **kwargs) -> Tuple[Any, Any]:
    #     instance = self._get_instance(*args, **kwargs)
    #     original = getattr(instance, self._method_name)
    #     setattr(instance, self._method_name, lambda *a, **kw: None)
    #     return (instance, original)

    # def restore(self, token: Tuple[Any, Any], *args, **kwargs) -> None:
    #     instance, original = token
    #     setattr(instance, self._method_name, original)
    def suppress(self, *args, **kwargs):
          instance = self._get_instance(*args, **kwargs)
          original = getattr(instance, self._method_name)
          captured, order = {}, []
          def _capture(edge_link, msg, from_node, *a, **kw):
              key = id(edge_link.target)
              if key not in captured:
                  order.append(key)
              captured[key] = (edge_link, msg, from_node, a, kw)  # keep last run's msgs
          setattr(instance, self._method_name, _capture)
          return (instance, original, captured, order)

    def restore(self, token, *args, **kwargs):
        instance, original, captured, order = token
        setattr(instance, self._method_name, original)
        for key in order:
            edge_link, msg, from_node, a, kw = captured[key]
            original(edge_link, msg, from_node, *a, **kw)
