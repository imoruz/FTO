from copy import deepcopy
from typing import Callable



class Restart:
    def __init__(self):
        pass
    
    def set_context(self, context):
        self.context = deepcopy(context)

    def get_context(self):
        pass


class RestartAllContext(Restart):
    def __init__(self):
        super().__init__()

    def get_context(self):
        return self.context


class RestartNoContext(Restart):
    def __init__(self):
        super().__init__()

    def get_context(self):
        return None


class RestartRefinedContext(Restart):
    def __init__(self):
        super().__init__()

    def get_context(self):
        # TODO
        # based on fm_mode 
        # based on just a summary. 
        # 
        pass
