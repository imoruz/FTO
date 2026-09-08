from copy import deepcopy
from typing import Any
from llmlingua import PromptCompressor


class Restart:
    def __init__(self, restart_count: int = 1) -> None:
        # Maximum number of times a single node may be restarted
        self.restart_count = restart_count

    def set_context(self, context: Any) -> None:
        self.context = deepcopy(context)

    def get_context(self) -> Any:
        pass


class RestartAllContext(Restart):
    def __init__(self, restart_count: int = 1) -> None:
        super().__init__(restart_count)

    def get_context(self) -> Any:
        return self.context


class RestartNoContext(Restart):
    def __init__(self, restart_count: int = 1) -> None:
        super().__init__(restart_count)

    def get_context(self) -> Any:
        return []


class RestartRefinedContext(Restart):
    def __init__(self, restart_count: int = 1) -> None:
        super().__init__(restart_count)

    def get_context(self) -> Any:
        # TODO
        # based on fm_mode
        # based on just a summary.
        #
        pass
    def _compress(self):

        llm_lingua = PromptCompressor()
        # compressed_prompt = llm_lingua.compress_prompt(
        #     prompt_list,
        #     question=question,
        #     rate=0.55,
        #     # Set the special parameter for LongLLMLingua
        #     condition_in_question="after_condition",
        #     reorder_context="sort",
        #     dynamic_context_compression_ratio=0.3, # or 0.4
        #     condition_compare=True,
        #     context_budget="+100",
        #     rank_method="longllmlingua",
        # )
