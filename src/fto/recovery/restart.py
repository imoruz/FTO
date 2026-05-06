from copy import deepcopy
from typing import Any


class Restart:
    def __init__(self) -> None:
        pass

    def set_context(self, context: Any) -> None:
        self.context = deepcopy(context)

    def get_context(self) -> Any:
        pass


class RestartAllContext(Restart):
    def __init__(self) -> None:
        super().__init__()

    def get_context(self) -> Any:
        return self.context


class RestartNoContext(Restart):
    def __init__(self) -> None:
        super().__init__()

    def get_context(self) -> Any:
        return None


class RestartRefinedContext(Restart):
    def __init__(self) -> None:
        super().__init__()

    def get_context(self) -> Any:
        # TODO
        # based on fm_mode
        # based on just a summary.
        #
        pass
