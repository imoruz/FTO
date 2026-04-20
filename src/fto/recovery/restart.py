from copy import deepcopy
from typing import Callable

from aegis_core import FMErrorType


class Restart:
    def __init__(self, fault_mode: FMErrorType):
        self.fault_mode = fault_mode

    def set_context(self, context):
        self.context = deepcopy(context)

    def get_context(self):
        pass


class RestartAllContext(Restart):
    def __init__(self, fault_mode: FMErrorType):
        super().__init__(fault_mode)

    def get_context(self):
        return self.context


class RestartNoContext(Restart):
    def __init__(self, fault_mode: FMErrorType):
        super().__init__(fault_mode)

    def get_context(self):
        return None


class RestartRefinedContext(Restart):
    def __init__(self, fault_mode: FMErrorType):
        super().__init__(fault_mode)

    def get_context(self):
        # TODO
        pass
